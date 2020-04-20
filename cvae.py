import argparse
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from psbody.mesh import Mesh, MeshViewers
import mesh_operations
from config_parser import read_config
from data import ComaDataset
from model_cvae import Coma
from transform import Normalize
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

def adjust_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay

def save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir):
    checkpoint = {}
    checkpoint['state_dict'] = coma.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_'+ str(epoch)+'.pt'))

def main(args):
    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)

    config = read_config(args.conf)

    print('Initializing parameters')
    template_file_path = config['template_fname']
    template_mesh = Mesh(filename=template_file_path)

    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    visualize = config['visualize']
    output_dir = config['visual_output_dir']
    if visualize is True and not output_dir:
        print('No visual output directory is provided. Checkpoint directory will be used to store the visual results')
        output_dir = checkpoint_dir

    if not output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    eval_flag = config['eval']
    lr = config['learning_rate']
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    total_epochs = config['epoch']
    workers_thread = config['workers_thread']
    opt = config['optimizer']
    batch_size = config['batch_size']
    val_losses, accs, durations = [], [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Generating transforms')
    M, A, D, U = mesh_operations.generate_transform_matrices(template_mesh, config['downsampling_factors'])

    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    print('Loading Dataset')
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = config['data_dir']

    normalize_transform = Normalize()
    dataset = ComaDataset(data_dir, dtype='train', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
    dataset_val = ComaDataset(data_dir, dtype='val', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
    dataset_test = ComaDataset(data_dir, dtype='test', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers_thread)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=workers_thread)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=workers_thread)

    print('Loading model')
    start_epoch = 1
    coma = Coma(dataset, config, D_t, U_t, A_t, num_nodes)
    if opt == 'adam':
        optimizer = torch.optim.Adam(coma.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(coma.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise Exception('No optimizer provided')

    checkpoint_file = config['checkpoint_file']
    print(checkpoint_file)
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch_num']
        coma.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #To find if this is fixed in pytorch
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    coma.to(device)

    if eval_flag:
        val_loss = evaluate(coma, output_dir, test_loader, dataset_test, template_mesh, device, visualize)
        print('val loss', val_loss)
        return

    best_val_loss = float('inf')
    val_loss_history = []

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs/cvae', current_time)
    writer = SummaryWriter(log_dir+'_dp_z2')
    print(coma.z)

    for epoch in range(start_epoch, total_epochs + 1):
        print("Training for epoch ", epoch)
        recon_loss, kld_loss, mu, var = train(coma, train_loader, len(dataset), optimizer, device)
        val_loss = evaluate(coma, output_dir, val_loader, dataset_val, template_mesh, device, visualize=visualize)
        train_loss = recon_loss + kld_loss
        writer.add_scalar('loss/train_loss', recon_loss + kld_loss, epoch)
        writer.add_scalar('train_loss/recon_loss', recon_loss, epoch)
        writer.add_scalar('train_loss/kld_loss', kld_loss, epoch)
        writer.add_scalar('loss/val_loss', val_loss, epoch)
        writer.add_histogram('hist/mean', mu, epoch)
        writer.add_histogram('hist/variance', var, epoch)

        print('epoch ', epoch,' Recon loss ', recon_loss, ' KLD loss ', kld_loss,' Val loss ', val_loss)
        if val_loss < best_val_loss:
            save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir)
            best_val_loss = val_loss

        if epoch == total_epochs or epoch % 100 == 0:
            save_model(coma, optimizer, epoch, train_loss, val_loss, checkpoint_dir)

        val_loss_history.append(val_loss)
        val_losses.append(best_val_loss)

        if opt=='sgd':
            adjust_learning_rate(optimizer, lr_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    writer.close()

def train(coma, train_loader, len_dataset, optimizer, device):
    coma.train()
    total_loss = 0
    recon_loss = 0
    kld_loss = 0
    mu_total = np.asarray([])
    var_total = np.asarray([])
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out, mu, logvar = coma(data)
        loss = F.l1_loss(out, data.x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld /= coma.z
        kld /= data.num_graphs
        recon_loss += data.num_graphs * loss.item()
        kld_loss += data.num_graphs * kld.item()
        mu_total = np.r_[mu_total, torch.reshape(mu,
            (1, data.num_graphs * coma.z)).detach().numpy()[0]]
        var_total = np.r_[var_total, torch.reshape(logvar.exp(),
            (1, data.num_graphs * coma.z)).detach().numpy()[0]]
        loss = loss + 1 * kld
        loss.backward()
        optimizer.step()
    return recon_loss / len_dataset, kld_loss / len_dataset, mu_total, var_total

def evaluate(coma, output_dir, test_loader, dataset, template_mesh, device, visualize=False):
    coma.eval()
    total_loss = 0
    if visualize :
        meshviewer = MeshViewers(shape=(1, 2))
    for i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            out, _, _ = coma(data)
        loss = F.l1_loss(out, data.x)
        total_loss += data.num_graphs * loss.item()

        if visualize and i % 100 == 0:
            save_out = out.detach().cpu().numpy()
            save_out = save_out*dataset.std.numpy()+dataset.mean.numpy()
            expected_out = (data.x.detach().cpu().numpy())*dataset.std.numpy()+dataset.mean.numpy()
            result_mesh = Mesh(v=save_out, f=template_mesh.f)
            expected_mesh = Mesh(v=expected_out, f=template_mesh.f)
            meshviewer[0][0].set_dynamic_meshes([result_mesh])
            meshviewer[0][1].set_dynamic_meshes([expected_mesh])
            meshviewer[0][0].save_snapshot(os.path.join(output_dir, 'file'+str(i)+'.png'), blocking=False)

    return total_loss/len(dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')
    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-s', '--split', default='gnrt', help='split can be gnrt, clsf, or lgtd')
    parser.add_argument('-st', '--split_term', default='gnrt', help='split can be gnrt, clsf, or lgtd')
    parser.add_argument('-d', '--data_dir', help='path where the downloaded data is stored')
    parser.add_argument('-cp', '--checkpoint_dir', help='path where checkpoints file need to be stored')

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    main(args)
