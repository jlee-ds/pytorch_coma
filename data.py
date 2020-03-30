import argparse
import glob
import numpy as np
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
        processed_files = [self.split_term+'_'+pf for pf in processed_files]
        return processed_files

    def gather_paths(self, split):
        datapaths = dict()
        if split == 'gnrt' :
            datapaths['all'] = []
            datapaths['all'] += glob.glob(self.root_dir+'/*.obj')
        elif split == 'clsf' :
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
            for i in cn_ids :
                datapaths['cn'].append(self.root_dir+'/'+i+'-L_Hipp_first.obj')
        elif split == 'lgtd' :
            datapaths['bl'] = []
            datapaths['m24'] = []
            bl_m24_ids = pd.read_csv('/Users/jlee/Desktop/JONG/tum/thesis/data/adni2/bl_m24_imgid.csv')
            bl_ids = bl_m24_ids['bl']; m24_ids = bl_m24_ids['m24'];
            for i in bl_ids :
                datapaths['bl'].append(self.root_dir+'/'+str(i)+'-L_Hipp_first.obj')
            for i in m24_ids :
                datapaths['m24'].append(self.root_dir+'/'+str(i)+'-L_Hipp_first.obj')

        return datapaths

    def process(self):
        train_data, val_data, test_data = [], [], []
        train_vertices = []
        for key in self.data_file :
            for idx, data_file in tqdm(enumerate(self.data_file[key])):
                mesh = Mesh(filename=data_file)
                mesh_verts = torch.Tensor(mesh.v)
                adjacency = get_vert_connectivity(mesh.v, mesh.f).tocoo()
                edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col)))
                data = Data(x=mesh_verts, y=mesh_verts, edge_index=edge_index)

                if idx % 100 <= 10:
                    test_data.append(data)
                elif idx % 100 <= 20:
                    val_data.append(data)
                else:
                    train_data.append(data)
                    train_vertices.append(mesh.v)

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

def prepare_clsf_dataset(path):
    ComaDataset(path, split='clsf', split_term='clsf', pre_transform=None)

def prepare_lgtd_dataset(path):
    ComaDataset(path, split='lgtd', split_term='lgtd', pre_transform=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ADNI2 Data preparation for Convolutional Mesh Autoencoders')
    parser.add_argument('-s', '--split', default='gnrt', help='split can be gnrt, clsf, or lgtd')
    parser.add_argument('-d', '--data_dir', help='path where the downloaded data is stored')

    args = parser.parse_args()
    split = args.split
    data_dir = args.data_dir
    if split == 'gnrt':
        prepare_gnrt_dataset(data_dir)
    elif split == 'clsf':
        prepare_clsf_dataset(data_dir)
    elif split == 'lgtd':
        prepare_lgtd_dataset(data_dir)
    else:
        raise Exception("Only gnrt, clsf, and lgtd split are supported")

