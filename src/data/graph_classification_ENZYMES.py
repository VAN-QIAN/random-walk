from typing import List, Dict
import os
import os.path as osp
import pickle
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.datasets import TUDataset
from typing import Callable, List, Optional
from torch_geometric.io import fs, read_tu_data
from torch_geometric.utils import remove_self_loops, to_undirected

from .walker import Walker


class GraphCLSENZYMESWalker(Walker):
    def __init__(self, config):
        super().__init__(config)
        self.out_dim = 6
        self.metric_name = 'accuracy'

    def criterion(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return torch.nn.functional.cross_entropy(y_hat, y)

    def evaluator(self, y_hat: Tensor, y: Tensor) -> Dict:
        preds, target = y_hat, y
        # compute metrics
        preds = preds.argmax(dim=-1)
        metric_val = (preds == target).float().mean()
        batch_size = target.size(0)
        return {
            'metric_sum': metric_val * batch_size,
            'metric_count': batch_size
        }

import os.path as osp
from torch_geometric.data import InMemoryDataset


class GraphCLSENZYMESDataset(InMemoryDataset):
    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(
        self,
        root,
        config,
        split: str = 'train',  # 'train', 'val', or 'test'
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        use_node_attr: bool = False,
        use_edge_attr: bool = False,
        cleaned: bool = False,
        train_val_test_split=(0.8, 0.1, 0.1),  # Default split ratio
    ) -> None:
        self.name = 'ENZYMES'
        self.cleaned = cleaned
        self.split = split
        self.train_val_test_split = train_val_test_split
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        # Load the appropriate dataset split based on `split`
        self.load_processed_split()

    def load_processed_split(self):
        """Load the processed data based on the split."""
        if self.split == 'train':
            data_path = self.processed_paths[0]
        elif self.split == 'val':
            data_path = self.processed_paths[1]
        elif self.split == 'test':
            data_path = self.processed_paths[2]
        else:
            raise ValueError(f"Invalid split: {self.split}. Choose from 'train', 'val', or 'test'.")

        out = torch.load(data_path)
        if not isinstance(out, tuple) or len(out) < 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility.
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_labels(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_file_names(self) -> List[str]:
        """Processed file names for train, val, and test splits."""
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']
    
    def download(self) -> None:
        url = self.url
        fs.cp(f'{url}/{self.name}.zip', self.raw_dir, extract=True)
        for filename in fs.ls(osp.join(self.raw_dir, self.name)):
            fs.mv(filename, osp.join(self.raw_dir, osp.basename(filename)))
        fs.rm(osp.join(self.raw_dir, self.name))
        
    def load_or_create_split_indices(self):
        """Load or generate new split indices for train/val/test splits."""
        split_path = f"./split/{self.name}.pt"
        if os.path.exists(split_path):
            # Load predefined indices if available
            indices = torch.load(split_path)
        else:
            # Create new split indices if not available
            torch.manual_seed(0)
            indices = torch.randperm(len(self))  # Shuffle the dataset
            train_size = int(self.train_val_test_split[0] * len(self))
            val_size = int(self.train_val_test_split[1] * len(self))
            test_size = len(self) - train_size - val_size

            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

            # Save the indices for future use
            # indices_dict = {
            #     'train': train_indices,
            #     'val': val_indices,
            #     'test': test_indices
            # }
            # torch.save(indices_dict, split_path)
            # indices = indices_dict
            torch.save(indices, split_path)
            print("New indices generated and saved.")

        return indices
        
    def process(self) -> None:
        # Step 1: Read the raw data
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)

        # Step 2: Load all data points into a data list
        data_list = [self.get(idx) for idx in range(len(self))]

        # Step 3: Apply pre_filter if specified
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        # Step 4: Apply pre_transform if specified
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # Step 5: Get indices for train, val, and test splits
        indices = self.load_or_create_split_indices()
        train_size = int(self.train_val_test_split[0] * len(self))
        val_size = int(self.train_val_test_split[1] * len(self))
        test_size = len(self) - train_size - val_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Step 6: Split data using predefined or generated indices
        train_data = [data_list[idx] for idx in train_indices]
        val_data = [data_list[idx] for idx in val_indices] #indices['val']]
        test_data = [data_list[idx] for idx in test_indices]#indices['test']]

        # Step 7: Collate and save the train set
        train_data, train_slices = self.collate(train_data)
        torch.save((train_data.to_dict(), train_slices, sizes, train_data.__class__), self.processed_paths[0])

        # Step 8: Collate and save the validation set
        val_data, val_slices = self.collate(val_data)
        torch.save((val_data.to_dict(), val_slices, sizes, val_data.__class__), self.processed_paths[1])

        # Step 9: Collate and save the test set
        test_data, test_slices = self.collate(test_data)
        torch.save((test_data.to_dict(), test_slices, sizes, test_data.__class__), self.processed_paths[2])

        # Reset cache (if applicable)
        self._data_list = None

    def __repr__(self) -> str:
        return f'{self.name}({self.split} set, {len(self)} graphs)'
