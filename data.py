import argparse
from glob import glob
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from psbody.mesh import Mesh
from utils import get_vert_connectivity
from transform import Normalize

class ComaDataset(InMemoryDataset):
    def __init__(self, root_dir, dtype='train', split='sliced', split_term='sliced', nVal = 100, transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.split_term = split_term
        self.nVal = nVal
        self.transform = transform
        self.pre_tranform = pre_transform
        # Downloaded data is present in following format root_dir/*/*/*.py
        if not osp.exists(osp.join(root_dir, 'processed', self.split_term)):
            os.makedirs(osp.join(root_dir, 'processed', self.split_term))
        self.data_file = self.gather_paths(self.split)
        super(ComaDataset, self).__init__(root_dir, transform, pre_transform)
        if dtype == 'train':
            data_path = self.processed_paths[0]
        elif dtype == 'val':
            data_path = self.processed_paths[1]
        elif dtype == 'test':
            data_path = self.processed_paths[2]
        else:
            raise Exception("train, val and test are supported data types")

        norm_path = self.processed_paths[3]
        norm_dict = torch.load(norm_path)
        self.mean, self.std = norm_dict['mean'], norm_dict['std']
        self.data, self.slices = torch.load(data_path)
        if self.transform:
            self.data = [self.transform(td) for td in self.data]

    @property
    def raw_file_names(self):
        all_data_files = []
        for key in self.data_file.keys() :
            all_data_files += self.data_file[key]
        return all_data_files

    @property
    def processed_file_names(self):
        processed_files = ['training.pt', 'val.pt', 'test.pt', 'norm.pt']
        processed_files = [self.split_term+'/'+pf for pf in processed_files]
        return processed_files

    def gather_paths(self, split):
        datapaths = dict()
        if split == 'gnrt' :
            datapaths['all'] = []
            obj_list = glob(osp.join(self.root_dir, '*.obj'))
            print('data length: ', len(obj_list))
            datapaths['all'] = datapaths['all'] + obj_list
        elif split == 'gnrtdx' :
            datapaths['ad'] = []
            datapaths['cn'] = []
            ad_ids = pd.read_csv('/Users/jlee/Desktop/JONG/tum/thesis/data/adni2/AD_PTID_IMGID.csv')
            ad_ids = ad_ids['mesh_filename']
            ad_ids = [i.split('_')[0] for i in ad_ids]
            cn_ids = pd.read_csv('/Users/jlee/Desktop/JONG/tum/thesis/data/adni2/CN_PTID_IMGID.csv')
            cn_ids = cn_ids['mesh_filename']
            cn_ids = [i.split('_')[0] for i in cn_ids]

            for i in ad_ids :
                datapaths['ad'].append(self.root_dir+'/'+i+'-L_Hipp_first.obj')
            for n, i in enumerate(cn_ids) :
                if self.split_term == 'gnrtdxb' and n >= len(ad_ids) :
                    break
                datapaths['cn'].append(self.root_dir+'/'+i+'-L_Hipp_first.obj')
            print('AD ', len(datapaths['ad']), ', CN ', len(datapaths['cn']))
        elif split == 'lgtd' :
            print('lgtd dataset file paths...')
            datapaths['lgtd'] = []
            bl_m24_ids = pd.read_csv('/Users/jlee/Desktop/JONG/tum/thesis/data/adni2/bl_m24_imgid.csv')
            bl_ids = bl_m24_ids['bl']; m24_ids = bl_m24_ids['m24'];
            print('length of dataset is %d'%(len(bl_m24_ids)))
            for i in range(len(bl_m24_ids)):
                blm24 = bl_m24_ids.loc[i]
                bl_path = self.root_dir+'/'+str(blm24['bl'])+'-L_Hipp_first.obj'
                m24_path = self.root_dir+'/'+str(blm24['m24'])+'-L_Hipp_first.obj'
                datapaths['lgtd'].append([bl_path, m24_path])
        elif split == 'lgtddxi' :
            print('lgtdc dataset file paths...')
            ad_count, cn_count = 0, 0;
            datapaths['lgtddx'] = []
            bl_m24_ids = pd.read_csv('/Users/jlee/Desktop/JONG/tum/thesis/data/adni2/bl_m24_imgid.csv')
            adni_info = pd.read_csv('/Users/jlee/Desktop/JONG/tum/thesis/data/adni2/adni_info.csv')
            bl_ids = bl_m24_ids['bl']; m24_ids = bl_m24_ids['m24'];

            for i in range(len(bl_m24_ids)):
                blm24 = bl_m24_ids.loc[i]
                # for lgtddxo, change 'bl' to 'm24' in the below code line.
                bl_DX = adni_info[adni_info['ImageUID'] == blm24['bl']].iloc[0]['DX']
                #print(bl_DX)
                if bl_DX in ['CN', 'Dementia'] :
                    bl_path = self.root_dir+'/'+str(blm24['bl'])+'-L_Hipp_first.obj'
                    m24_path = self.root_dir+'/'+str(blm24['m24'])+'-L_Hipp_first.obj'
                    if bl_DX == 'CN' :
                        bl_class = [0, 1];  cn_count += 1
                    elif bl_DX == 'Dementia' :
                        bl_class = [1, 0];  ad_count += 1
                    datapaths['lgtddx'].append([bl_path, m24_path, bl_class])
            print('length of dataset is %d'%(ad_count+cn_count))
            print(ad_count, cn_count)
        elif split == 'lgtddxo' :
            print('lgtdc dataset file paths...')
            ad_count, cn_count = 0, 0;
            datapaths['lgtddx'] = []
            bl_m24_ids = pd.read_csv('/Users/jlee/Desktop/JONG/tum/thesis/data/adni2/bl_m24_imgid.csv')
            adni_info = pd.read_csv('/Users/jlee/Desktop/JONG/tum/thesis/data/adni2/adni_info.csv')
            bl_ids = bl_m24_ids['bl']; m24_ids = bl_m24_ids['m24'];

            for i in range(len(bl_m24_ids)):
                blm24 = bl_m24_ids.loc[i]
                m24_DX = adni_info[adni_info['ImageUID'] == blm24['m24']].iloc[0]['DX']
                #print(bl_DX)
                if m24_DX in ['CN', 'Dementia'] :
                    bl_path = self.root_dir+'/'+str(blm24['bl'])+'-L_Hipp_first.obj'
                    m24_path = self.root_dir+'/'+str(blm24['m24'])+'-L_Hipp_first.obj'
                    if m24_DX == 'CN' :
                        m24_class = [0, 1];  cn_count += 1
                    elif m24_DX == 'Dementia' :
                        m24_class = [1, 0];  ad_count += 1
                    datapaths['lgtddx'].append([bl_path, m24_path, m24_class])
            print('length of dataset is %d'%(ad_count+cn_count))
            print(ad_count, cn_count)
        elif split == 'lgtdvc' :
            print('lgtdvc dataset file paths...')
            ad_count, cn_count = 0, 0;
            bl_count, m12_count, m24_count = 0, 0, 0;
            datapaths['lgtdvc'] = []
            imgids = pd.read_csv('/Users/jlee/Desktop/JONG/tum/thesis/data/adni2/blm12m24_imgids.csv')
            adni_info = pd.read_csv('/Users/jlee/Desktop/JONG/tum/thesis/data/adni2/adni_info.csv')
            for i in range(len(imgids)):
                imgid = imgids.iloc[i]['ImageUID']
                #print(imgid, type(imgid))
                if i % 3 == 0 : # BL
                    bl_count += 1
                    # period = [1, 0, 0]
                    bl_path = self.root_dir+'/'+str(imgid)+'-L_Hipp_first.obj'
                    bl_DX = adni_info[adni_info['ImageUID'] == imgid].iloc[0]['DX']
                    if bl_DX == 'CN' :
                        bl_class = [0, 1];  cn_count += 1
                    elif bl_DX == 'Dementia' :
                        bl_class = [1, 0];  ad_count += 1
                    # datapaths['lgtdp'].append([bl_path, bl_path, bl_class, period])
                elif i % 3 == 1 : # M12
                    m12_count += 1
                    period = [1, 0]
                    m12_path = self.root_dir+'/'+str(imgid)+'-L_Hipp_first.obj'
                    #print([bl_path, m12_path, bl_class, period])
                    datapaths['lgtdvc'].append([bl_path, m12_path, bl_class, period])
                elif i % 3 == 2 : # M24
                    m24_count += 1
                    period = [0, 1]
                    m24_path = self.root_dir+'/'+str(imgid)+'-L_Hipp_first.obj'
                    #print([bl_path, m24_path, bl_class, period])
                    datapaths['lgtdvc'].append([bl_path, m24_path, bl_class, period])
            #print('length of dataset is %d'%((ad_count+cn_count)*3))
            print(bl_count + m12_count + m24_count)
            print(bl_count, m12_count, m24_count)
            #print(ad_count, cn_count)

        return datapaths

    def process(self):
        train_data, val_data, test_data = [], [], []
        train_vertices = []
        for key in self.data_file :
            for idx, data_file in tqdm(enumerate(self.data_file[key])):
                if key == 'lgtd' :
                    mesh = Mesh(filename=data_file[0])
                    mesh_verts = torch.Tensor(mesh.v)
                    adjacency = get_vert_connectivity(mesh.v, mesh.f).tocoo()
                    edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col)))
                    mesh_m24 = Mesh(filename=data_file[1])
                    data = Data(x=mesh_verts, y=torch.Tensor(mesh_m24.v), edge_index=edge_index)
                elif key == 'lgtddx' :
                    mesh = Mesh(filename=data_file[0])
                    mesh_verts = torch.Tensor(mesh.v)
                    adjacency = get_vert_connectivity(mesh.v, mesh.f).tocoo()
                    edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col)))
                    mesh_m24 = Mesh(filename=data_file[1])
                    data = Data(x=mesh_verts, y=torch.Tensor(mesh_m24.v), edge_index=edge_index,
                        label=torch.Tensor(data_file[2]))
                elif key == 'lgtdvc' :
                    #print(data_file)
                    mesh = Mesh(filename=data_file[0])
                    mesh_verts = torch.Tensor(mesh.v)
                    adjacency = get_vert_connectivity(mesh.v, mesh.f).tocoo()
                    edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col)))
                    mesh_fu = Mesh(filename=data_file[1])
                    data = Data(x=mesh_verts, y=torch.Tensor(mesh_fu.v), edge_index=edge_index,
                        label=torch.Tensor(data_file[2]), period=torch.Tensor(data_file[3]))
                else :
                    mesh = Mesh(filename=data_file)
                    mesh_verts = torch.Tensor(mesh.v)
                    adjacency = get_vert_connectivity(mesh.v, mesh.f).tocoo()
                    edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col)))
                    if key == 'ad' :
                        data = Data(x=mesh_verts, y=mesh_verts, label=torch.Tensor([1,0]), edge_index=edge_index)
                    elif key == 'cn' :
                        data = Data(x=mesh_verts, y=mesh_verts, label=torch.Tensor([0,1]), edge_index=edge_index)
                    elif key == 'all' :
                        data = Data(x=mesh_verts, y=mesh_verts, edge_index=edge_index)

                if idx % 100 < 10:
                    test_data.append(data)
                    #print(data.period)
                elif idx % 100 < 20:
                    val_data.append(data)
                else:
                    train_data.append(data)
                    train_vertices.append(mesh.v)
        print(len(train_data), len(val_data), len(test_data))

        mean_train = torch.Tensor(np.mean(train_vertices, axis=0))
        std_train = torch.Tensor(np.std(train_vertices, axis=0))
        norm_dict = {'mean': mean_train, 'std': std_train}
        if self.pre_transform is not None:
            if hasattr(self.pre_transform, 'mean') and hasattr(self.pre_transform, 'std'):
                if self.pre_tranform.mean is None:
                    self.pre_tranform.mean = mean_train
                if self.pre_transform.std is None:
                    self.pre_tranform.std = std_train
            train_data = [self.pre_transform(td) for td in train_data]
            val_data = [self.pre_transform(td) for td in val_data]
            test_data = [self.pre_transform(td) for td in test_data]

        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(val_data), self.processed_paths[1])
        torch.save(self.collate(test_data), self.processed_paths[2])
        torch.save(norm_dict, self.processed_paths[3])

def prepare_gnrt_dataset(path):
    ComaDataset(path, split='gnrt', split_term='gnrt', pre_transform=Normalize())

def prepare_gnrtdx_dataset(path):
    ComaDataset(path, split='gnrtdx', split_term='gnrtdx', pre_transform=Normalize())

def prepare_lgtd_dataset(path):
    ComaDataset(path, split='lgtd', split_term='lgtd', pre_transform=Normalize())

def prepare_lgtddxi_dataset(path):
    ComaDataset(path, split='lgtddxi', split_term='lgtddxi', pre_transform=Normalize())

def prepare_lgtddxo_dataset(path):
    ComaDataset(path, split='lgtddxo', split_term='lgtddxo', pre_transform=Normalize())

def prepare_lgtdvc_dataset(path):
    ComaDataset(path, split='lgtdvc', split_term='lgtdvc', pre_transform=Normalize())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ADNI2 Data preparation for Convolutional Mesh Autoencoders')
    parser.add_argument('-s', '--split', default='gnrt', help='split can be gnrt, clsf, lgtd, lgtdvc, or lgtddx')
    parser.add_argument('-d', '--data_dir', help='path where the downloaded data is stored')

    args = parser.parse_args()
    split = args.split
    data_dir = args.data_dir
    if split == 'gnrt':
        prepare_gnrt_dataset(data_dir)
    elif split == 'gnrtdx':
        prepare_gnrtdx_dataset(data_dir)
    elif split == 'lgtd':
        prepare_lgtd_dataset(data_dir)
    elif split == 'lgtddxi':
        prepare_lgtddxi_dataset(data_dir)
    elif split == 'lgtddxo':
        prepare_lgtddxo_dataset(data_dir)
    elif split == 'lgtdvc':
        prepare_lgtdvc_dataset(data_dir)
    else:
        raise Exception("Only gnrt, clsf, and lgtd split are supported")

