import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
from psbody.mesh import Mesh, MeshViewers
import mesh_operations
from config_parser import read_config
from data import ComaDataset
from model_cvae import Coma
from transform import Normalize
import readchar
from numpy import linalg as LA

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
    parser.add_argument('-c', '--conf', default='cfgs/cvae_dx.cfg', help='path of config file')
    parser.add_argument('-s', '--split', default='gnrtdx', help='split can be gnrt, clsf, or lgtd')
    parser.add_argument('-st', '--split_term', default='gnrtdx', help='split can be gnrt, clsf, or lgtd')
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
    dataset = ComaDataset(data_dir, dtype='test', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    print('Loading model')
    coma = Coma(dataset, config, D_t, U_t, A_t, num_nodes)

    checkpoint_file = config['checkpoint_file']
    print(checkpoint_file)
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        coma.load_state_dict(checkpoint['state_dict'])
    coma.to(device)

    meshviewer = MeshViewers(shape=(3, cols))
    for row in meshviewer :
        for window in row :
            window.set_background_color(np.asarray([1.0, 1.0, 1.0]))
    coma.eval()

    exit = 0
    cnt = 0
    mu = torch.zeros([1, coma.z])
    logvar = torch.zeros([1, coma.z])
    ad_cond = torch.Tensor([1, 0])
    ad_cond = torch.reshape(ad_cond, (1,2))
    cn_cond = torch.Tensor([0, 1])
    cn_cond = torch.reshape(cn_cond, (1,2))
    cond = [ad_cond, cn_cond]
    std = torch.exp(0.5*logvar)
    eps = np.asarray([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    #eps = np.asarray([-3, -2, -1, 0, 1, 2, 3])
    while(1) :
        with torch.no_grad():
            cn_zero_out = coma.decoder(torch.cat((mu + eps[3] * std, cn_cond), dim=1))
            ad_zero_out = coma.decoder(torch.cat((mu + eps[3] * std, ad_cond), dim=1))
        cn_zero_out = cn_zero_out[0].detach().cpu().numpy()
        cn_zero_out = cn_zero_out*dataset.std.numpy()+dataset.mean.numpy()
        ad_zero_out = ad_zero_out[0].detach().cpu().numpy()
        ad_zero_out = ad_zero_out*dataset.std.numpy()+dataset.mean.numpy()
        for i, e in enumerate(eps) :
            x = mu + e * std
            cn_x = torch.cat((x, cn_cond), dim=1)
            ad_x = torch.cat((x, ad_cond), dim=1)
            with torch.no_grad():
                cn_out = coma.decoder(cn_x)
                ad_out = coma.decoder(ad_x)
            #weights = F.l1_loss(cn_out, ad_out, reduction='none')
            #weights = torch.mean(weights, dim=[0, 2])
            #weights = torch.reshape(weights, (732, 3))
            cn_save_out = cn_out.detach().cpu().numpy()[0]
            cn_save_out = cn_save_out*dataset.std.numpy()+dataset.mean.numpy()
            ad_save_out = ad_out.detach().cpu().numpy()[0]
            ad_save_out = ad_save_out*dataset.std.numpy()+dataset.mean.numpy()
            ad_delta = LA.norm(ad_save_out - ad_zero_out, ord=2, axis=1)
            cn_delta = LA.norm(cn_save_out - cn_zero_out, ord=2, axis=1)
            mse_delta = LA.norm(ad_save_out - cn_save_out, ord=2, axis=1)
            cn_result_mesh = Mesh(v=cn_save_out, f=template_mesh.f)
            ad_result_mesh = Mesh(v=ad_save_out, f=template_mesh.f)
            ad_mse_mesh = Mesh(v=ad_save_out, f=template_mesh.f)
            #weights = weights.detach().cpu().numpy()
            if e != 0 :
                ad_result_mesh.set_vertex_colors_from_weights(20*ad_delta, scale_to_range_1=False, color=True)
                cn_result_mesh.set_vertex_colors_from_weights(20*cn_delta, scale_to_range_1=False, color=True)
            ad_mse_mesh.set_vertex_colors_from_weights(20*mse_delta, scale_to_range_1=False, color=True)
            meshviewer[0][i].set_dynamic_meshes([ad_mse_mesh])
            meshviewer[1][i].set_dynamic_meshes([ad_result_mesh])
            meshviewer[2][i].set_dynamic_meshes([cn_result_mesh])
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
            elif input_key == 'z' :
                eps = np.asarray([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
                break
        if exit :
            break

