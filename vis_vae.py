import argparse
import os
import torch
import numpy as np
from torch_geometric.data import DataLoader
from psbody.mesh import Mesh, MeshViewers
import mesh_operations
from config_parser import read_config
from data import ComaDataset
from model_vae import Coma
from transform import Normalize
import readchar

def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Trainer for Convolutional Mesh Autoencoders')
    parser.add_argument('-c', '--conf', default='cfgs/vae.cfg', help='path of config file')
    parser.add_argument('-s', '--split', default='gnrt', help='split can be gnrt, clsf, or lgtd')
    parser.add_argument('-st', '--split_term', default='gnrt', help='split can be gnrt, clsf, or lgtd')
    parser.add_argument('-d', '--data_dir', help='path where the downloaded data is stored')
    parser.add_argument('-cp', '--checkpoint_dir', help='path where checkpoints file need to be stored')
    cols = 7

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'default.cfg')
        print('configuration file not specified, trying to load '
              'it from current directory', args.conf)

    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)

    config = read_config(args.conf)

    print('Initializing parameters')
    template_file_path = config['template_fname']
    template_mesh = Mesh(filename=template_file_path)

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
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    print('Loading model')
    coma = Coma(dataset, config, D_t, U_t, A_t, num_nodes)

    checkpoint_file = config['checkpoint_file']
    print(checkpoint_file)
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        coma.load_state_dict(checkpoint['state_dict'])
    coma.to(device)

    meshviewer = MeshViewers(shape=(1, cols))
    coma.eval()

    exit = 0
    cnt = 0
    mu = torch.zeros([1, coma.z])
    logvar = torch.zeros([1, coma.z])
    std = torch.exp(0.5*logvar)
    eps = np.asarray([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    while(1) :
        for i, e in enumerate(eps) :
            x = mu + e * std
            print(i, x.size())
            with torch.no_grad():
                out = coma.decoder(x)
            save_out = out.detach().cpu().numpy()
            save_out = save_out*dataset.std.numpy()+dataset.mean.numpy()
            result_mesh = Mesh(v=save_out, f=template_mesh.f)
            meshviewer[0][i].set_dynamic_meshes([result_mesh])
        while(1) :
            input_key = readchar.readchar()
            if input_key == "\x1b":
                exit = 1
                break
            elif input_key == "u" :
                eps = 1.1 * eps
                break
            elif input_key == "d" :
                eps = 0.9 * eps
                break
        if exit :
            break

