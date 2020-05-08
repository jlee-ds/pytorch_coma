import argparse
import os
import torch
import numpy as np
from torch_geometric.data import DataLoader
from psbody.mesh import Mesh, MeshViewers
import mesh_operations
from config_parser import read_config
from data import ComaDataset
from model import Coma
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
    parser.add_argument('-c', '--conf', default='cfgs/ae.cfg', help='path of config file')
    parser.add_argument('-s', '--split', default='gnrtdx', help='split can be gnrt, clsf, or lgtd')
    parser.add_argument('-st', '--split_term', default='gnrtdxb', help='split can be gnrt, clsf, or lgtd')
    parser.add_argument('-d', '--data_dir', help='path where the downloaded data is stored')
    parser.add_argument('-cp', '--checkpoint_dir', help='path where checkpoints file need to be stored')
    cols = 8

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
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    print('Loading model')
    coma = Coma(dataset, config, D_t, U_t, A_t, num_nodes)

    checkpoint_file = config['checkpoint_file']
    print(checkpoint_file)
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        coma.load_state_dict(checkpoint['state_dict'])
    coma.to(device)

    meshviewer = MeshViewers(shape=(3, cols))
    for row in meshviewer :
        for window in row :
            window.set_background_color(np.asarray([1.0, 1.0, 1.0]))
    coma.eval()

    exit = 0
    cnt = 0
    for i, data in enumerate(data_loader) :
        data = data.to(device)
        print(cnt, data.y)
        with torch.no_grad():
            out = coma(data)
        save_out = out.detach().cpu().numpy()
        expected_out = data.x.detach().cpu().numpy()
        if dataset.pre_transform is not None :
            save_out = save_out*dataset.std.numpy()+dataset.mean.numpy()
            expected_out = (data.x.detach().cpu().numpy())*dataset.std.numpy()+dataset.mean.numpy()

        result_delta = LA.norm(expected_out - save_out, ord=2, axis=1)
        delta_mesh = Mesh(v=save_out, f=template_mesh.f)
        result_mesh = Mesh(v=save_out, f=template_mesh.f)
        expected_mesh = Mesh(v=expected_out, f=template_mesh.f)

        delta_mesh.set_vertex_colors_from_weights(result_delta*10, scale_to_range_1=False, color=True)
        meshviewer[2][cnt].set_dynamic_meshes([expected_mesh])
        meshviewer[1][cnt].set_dynamic_meshes([result_mesh])
        meshviewer[0][cnt].set_dynamic_meshes([delta_mesh])
        cnt += 1
        if cnt == 8 :
            while(1) :
                input_key = readchar.readchar()
                if input_key == "\x1b":
                    exit = 1
                    break
                elif input_key == "n" :
                    cnt = 0
                    break
        if exit :
            break
