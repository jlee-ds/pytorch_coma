import argparse
import os
import torch
import numpy as np
from torch_geometric.data import DataLoader
from psbody.mesh import Mesh, MeshViewers
import mesh_operations
from config_parser import read_config
from data import ComaDataset
from model_cae_dx import Coma
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
    parser.add_argument('-c', '--conf', default='cfgs/lcae_dxf_de2.cfg', help='path of config file')
    parser.add_argument('-s', '--split', default='lgtddxo', help='split can be lgtddxb or lgtddxf')
    parser.add_argument('-st', '--split_term', default='lgtddxo', help='split can be lgtddxb or lgtddxf')
    parser.add_argument('-d', '--data_dir', default='data/ADNI2_data_all', help='path where the downloaded data is stored')
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
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        coma.load_state_dict(checkpoint['state_dict'])
    coma.to(device)

    meshviewer = MeshViewers(shape=(4, 4))
    for row in meshviewer :
        for window in row :
            window.set_background_color(np.asarray([1.0, 1.0, 1.0]))
    coma.eval()

    exit = 0
    cnt = 0
    for i, data in enumerate(data_loader) :

        if i not in [10, 14, 16, 27] :
            continue
        print(data.label)
        data = data.to(device)
        # print(cnt, data.y)
        data.label = torch.Tensor([1, 0])
        with torch.no_grad():
            ad_result_delta = coma(data)
        data.label = torch.Tensor([0, 1])
        with torch.no_grad():
            cn_result_delta = coma(data)
        ad_out = ad_result_delta + data.x
        ad_save_out = ad_out.detach().cpu().numpy()
        cn_out = cn_result_delta + data.x
        cn_save_out = cn_out.detach().cpu().numpy()
        expected_out = data.y.detach().cpu().numpy()
        base_input = data.x.detach().cpu().numpy()
        if dataset.pre_transform is not None :
            ad_save_out = ad_save_out*dataset.std.numpy()+dataset.mean.numpy()
            cn_save_out = cn_save_out*dataset.std.numpy()+dataset.mean.numpy()
            expected_out = (data.y.detach().cpu().numpy())*dataset.std.numpy()+dataset.mean.numpy()
            base_input = base_input*dataset.std.numpy()+dataset.mean.numpy()
        ad_result_delta = LA.norm(ad_save_out - base_input, ord=2, axis=1)
        cn_result_delta = LA.norm(cn_save_out - base_input, ord=2, axis=1)
        expected_delta = LA.norm(expected_out - base_input, ord=2, axis=1)
        #print(np.max(result_delta), np.max(expected_delta))

        ad_result_mesh = Mesh(v=ad_save_out, f=template_mesh.f)
        cn_result_mesh = Mesh(v=cn_save_out, f=template_mesh.f)
        expected_mesh = Mesh(v=expected_out, f=template_mesh.f)
        base_mesh = Mesh(v=base_input, f=template_mesh.f)

        max_dist = np.max([np.max(ad_result_delta), np.max(cn_result_delta),np.max(expected_delta)])
        ad_result_mesh.set_vertex_colors_from_weights(ad_result_delta*10, scale_to_range_1=False, color=True)
        cn_result_mesh.set_vertex_colors_from_weights(cn_result_delta*10, scale_to_range_1=False, color=True)
        expected_mesh.set_vertex_colors_from_weights(expected_delta*10, scale_to_range_1=False, color=True)
        meshviewer[3][cnt].set_dynamic_meshes([base_mesh])
        meshviewer[2][cnt].set_dynamic_meshes([expected_mesh])
        meshviewer[1][cnt].set_dynamic_meshes([ad_result_mesh])
        meshviewer[0][cnt].set_dynamic_meshes([cn_result_mesh])
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
