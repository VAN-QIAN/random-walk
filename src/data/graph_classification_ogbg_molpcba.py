from typing import List, Dict
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.datasets import TUDataset
from typing import Callable, List, Optional
from torch_geometric.io import fs, read_tu_data
from torch_geometric.utils import remove_self_loops, to_undirected
from ogb.graphproppred import PygGraphPropPredDataset
import pandas as pd
import shutil, os
import os.path as osp
import ogb

import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from torch_geometric.data import InMemoryDataset
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_graph_pyg

from .walker import Walker

from sklearn.metrics import roc_auc_score, average_precision_score

class GraphCLS_OGBGPCBA_Walker(Walker):
    def __init__(self, config):
        super().__init__(config)
        self.out_dim = 128
        self.metric_name = 'ap'
    

    def criterion(self, y_hat: Tensor, y: Tensor) -> Tensor:
        loss_fn = nn.BCEWithLogitsLoss()
        is_labeled = y == y  # Filter our nans.
        return loss_fn(y_hat[is_labeled], y[is_labeled].float())

    def evaluator(self, y_hat: Tensor, y: Tensor) -> Dict[str, float]:
        """
        Compute Average Precision (AP) averaged across tasks.

        Args:
            y_hat (Tensor): Logits output from the model, shape (batch_size, num_labels)
            y (Tensor): Ground truth labels, shape (batch_size, num_labels)

        Returns:
            Dict[str, float]: Dictionary containing the average precision score.
        """
        # 将张量移动到 CPU 并转换为 NumPy 数组
        y_pred = torch.sigmoid(y_hat).detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()
        batch_size = y_true.shape[0]
        # print(f"y_ture: {y_true.shape}")

        ap_list = []

        for i in range(y_true.shape[1]):
            # 仅在该标签至少有一个正例和一个负例时计算 AP
            
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                ap = average_precision_score(y_true[is_labeled,i], y_pred[is_labeled,i])
                
                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError(
                'No positively labeled data available. Cannot compute Average Precision.')

        average_ap = np.mean(ap_list)
        metric_val = average_ap
        
        return {
            'metric_sum': metric_val * batch_size,
            'metric_count': batch_size
        }
    

import os.path as osp
from torch_geometric.data import InMemoryDataset


class GraphCLS_OGBGPCBA_Dataset(InMemoryDataset):
    def __init__(self, root, config, split: str = 'train' ,transform=None, pre_transform=None, meta_dict=None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
            - split (str): the dataset split to load, can be 'train', 'valid', or 'test'
        '''
        self.name = 'ogbg-molpcba'  # original name, e.g., ogbg-molhiv
        self.split = split  # new argument for dataset split
        
        if meta_dict is None:
            self.dir_name = '_'.join(self.name.split('-')) 
            # check if previously-downloaded folder exists
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'
            self.original_root = root
            self.root = osp.join(root, self.dir_name)
            print(self.root)
            ogb_root = os.path.dirname(ogb.graphproppred.__file__)
        
            # 修改这里的路径，使用OGB下载路径下的master.csv
            master_csv_path = os.path.join(ogb_root, 'master.csv') #os.path.join(os.path.dirname(__file__), 'master.csv')
            master = pd.read_csv(master_csv_path, index_col=0, keep_default_na=False)
            if not self.name in master:
                error_mssg = f'Invalid dataset name {self.name}.\n'
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]
            
        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict
        
        # check version
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(f'{self.name} has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name']  # name of downloaded file, e.g., tox21
        self.num_tasks = int(self.meta_info['num tasks'])
        self.eval_metric = self.meta_info['eval metric']
        self.task_type = self.meta_info['task type']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.binary = self.meta_info['binary'] == 'True'

        super(GraphCLS_OGBGPCBA_Dataset, self).__init__(self.root, transform, pre_transform)
        # super().__init__(root, transform, pre_transform, pre_filter,
        #                  force_reload=force_reload)

        # Load the processed data corresponding to the selected split
        
        if split == 'val':
            split = 'valid'
        self.processed_file_path = osp.join(self.root, 'processed', f'geometric_data_processed_{split}.pt')
        if not osp.exists(self.processed_file_path):
            raise RuntimeError(f"Processed file for {split} split does not exist: {self.processed_file_path}")
        print(f"##########")
        print(f'Loading {split} split data from {self.processed_file_path}...')
        print(f"##########")
        self.data, self.slices = torch.load(self.processed_file_path)

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info['split']
            
        path = osp.join(self.root, 'split', split_type)

        # Short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]

        return {'train': torch.tensor(train_idx, dtype=torch.long), 'valid': torch.tensor(valid_idx, dtype=torch.long), 'test': torch.tensor(test_idx, dtype=torch.long)}

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            return ['data.npz']
        else:
            file_names = ['edge']
            if self.meta_info['has_node_attr'] == 'True':
                file_names.append('node-feat')
            if self.meta_info['has_edge_attr'] == 'True':
                file_names.append('edge-feat')
            return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return f'geometric_data_processed_{self.split}.pt'

    def download(self):
        url = self.meta_info['url']
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)
        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        # process the data according to the specified split
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'
        
        # Reading the raw data once
        additional_node_files = [] if self.meta_info['additional node files'] == 'None' else self.meta_info['additional node files'].split(',')
        additional_edge_files = [] if self.meta_info['additional edge files'] == 'None' else self.meta_info['additional edge files'].split(',')
        
        data_list = read_graph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge, additional_node_files=additional_node_files, additional_edge_files=additional_edge_files, binary=self.binary)
        
        # Add graph labels
        if self.task_type == 'subtoken prediction':
            graph_label_notparsed = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values
            graph_label = [str(graph_label_notparsed[i][0]).split(' ') for i in range(len(graph_label_notparsed))]

            for i, g in enumerate(data_list):
                g.y = graph_label[i]

        else:
            if self.binary:
                graph_label = np.load(osp.join(self.raw_dir, 'graph-label.npz'))['graph_label']
            else:
                graph_label = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values
        
        # Convert graph labels to torch.Tensor
        has_nan = np.isnan(graph_label).any()

        for i, g in enumerate(data_list):
            if 'classification' in self.task_type:
                if has_nan:
                    g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.float32)
                else:
                    g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.long)
            else:
                g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.float32)
        
        # Get split indices
        split_indices = self.get_idx_split()
        
        # Dictionary to hold data splits
        splits_data = {
            'train': [],
            'valid': [],
            'test': []
        }
        
        # Split the data based on the split indices
        for split_name, indices in split_indices.items():
            for idx in indices:
                splits_data[split_name].append(data_list[idx])
        
        # Process each split individually
        for split_name, split_data_list in splits_data.items():
            if self.pre_transform is not None:
                split_data_list = [self.pre_transform(data) for data in split_data_list]
            
            data, slices = self.collate(split_data_list)
            
            # Save the processed data for the current split
            print(f'Saving {split_name} split data...')
            torch.save((data, slices), osp.join(self.processed_dir, f'geometric_data_processed_{split_name}.pt'))

    def __repr__(self):
        return f'{self.name}({self.split} split, {len(self)} graphs)'
