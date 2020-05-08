import argparse
import os
import torch
import numpy as np
from torch_geometric.data import DataLoader
from psbody.mesh import Mesh, MeshViewers
import mesh_operations
from config_parser import read_config
from data import ComaDataset
from model_cae_vc import Coma
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
    parser.add_argument('-c', '--conf', default='cfgs/lcae_vc_de2.cfg', help='path of config file')
    parser.add_argument('-s', '--split', default='lgtdvc', help='split can be gnrt, clsf, or lgtd')
    parser.add_argument('-st', '--split_term', default='lgtdvc', help='split can be gnrt, clsf, or lgtd')
    parser.add_argument('-d', '--data_dir', default='data/ADNI2_data_lgtd', help='path where the downloaded data is stored')
    parser.add_argument('-cp', '--checkpoint_dir', help='path where checkpoints file need to be stored')


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
    print(data_dir)
    dataset = ComaDataset(data_dir, dtype='test', split=args.split, split_term=args.split_term, pre_transform=normalize_transform)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)

    print('Loading model')
    coma = Coma(dataset, config, D_t, U_t, A_t, num_nodes)

    checkpoint_file = config['checkpoint_file']
    print(checkpoint_file)
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        coma.load_state_dict(checkpoint['state_dict'])
    coma.to(device)

    cols = 5
    meshviewer = MeshViewers(shape=(5, 4))
    for row in meshviewer :
        for window in row :
            window.set_background_color(np.asarray([1.0, 1.0, 1.0]))
    coma.eval()

    exit = 0
    cnt = 0
    for i, data in enumerate(data_loader) :
        print(data.period)
        data = data.to(device)
        # print(cnt, data.y)
        with torch.no_grad():
            result_delta = coma(data)
        result_delta = result_delta.reshape(2, -1, 3)
        y = data.y.reshape(2, -1, 3)
        x = data.x.reshape(2, -1, 3)[0]
        m12_y, m24_y = y[0], y[1]
        m12_out = result_delta[0] + x
        m24_out = result_delta[1] + x
        m12_save_out = m12_out.detach().cpu().numpy()
        m12_expected_out = m12_y.detach().cpu().numpy()
        m24_save_out = m24_out.detach().cpu().numpy()
        m24_expected_out = m24_y.detach().cpu().numpy()
        base_input = x.detach().cpu().numpy()
        if dataset.pre_transform is not None :
            m12_save_out = m12_save_out*dataset.std.numpy()+dataset.mean.numpy()
            m12_expected_out = m12_y*dataset.std.numpy()+dataset.mean.numpy()
            m24_save_out = m24_save_out*dataset.std.numpy()+dataset.mean.numpy()
            m24_expected_out = m24_y*dataset.std.numpy()+dataset.mean.numpy()
            base_input = base_input*dataset.std.numpy()+dataset.mean.numpy()
        m12_result_delta = LA.norm(m12_save_out - base_input, ord=2, axis=1)
        m12_expected_delta = LA.norm(m12_expected_out - base_input, ord=2, axis=1)
        m24_result_delta = LA.norm(m24_save_out - base_input, ord=2, axis=1)
        m24_expected_delta = LA.norm(m24_expected_out - base_input, ord=2, axis=1)

        m12_result_mesh = Mesh(v=m12_save_out, f=template_mesh.f)
        m12_expected_mesh = Mesh(v=m12_expected_out, f=template_mesh.f)
        m24_result_mesh = Mesh(v=m24_save_out, f=template_mesh.f)
        m24_expected_mesh = Mesh(v=m24_expected_out, f=template_mesh.f)
        base_mesh = Mesh(v=base_input, f=template_mesh.f)

        max_dist = np.max([np.max(m12_result_delta), np.max(m12_expected_delta), np.max(m24_result_delta), np.max(m24_expected_delta)])
        m12_result_mesh.set_vertex_colors_from_weights(m12_result_delta*10, scale_to_range_1=False, color=True)
        m12_expected_mesh.set_vertex_colors_from_weights(m12_expected_delta*10, scale_to_range_1=False, color=True)
        m24_result_mesh.set_vertex_colors_from_weights(m24_result_delta*10, scale_to_range_1=False, color=True)
        m24_expected_mesh.set_vertex_colors_from_weights(m24_expected_delta*10, scale_to_range_1=False, color=True)

        meshviewer[4][cnt].set_dynamic_meshes([base_mesh])
        meshviewer[3][cnt].set_dynamic_meshes([m12_expected_mesh])
        meshviewer[2][cnt].set_dynamic_meshes([m24_expected_mesh])
        meshviewer[1][cnt].set_dynamic_meshes([m12_result_mesh])
        meshviewer[0][cnt].set_dynamic_meshes([m24_result_mesh])
        cnt += 1
        if cnt == 4 :
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

