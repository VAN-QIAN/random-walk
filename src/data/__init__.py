# pylint: disable=line-too-long
from typing import Tuple
from .ds_builder import DatasetBuilder
from .walker import Walker
from .graph_separation_csl import GraphSeparationCSLDataset, GraphSeparationCSLWalker
from .graph_separation_sr16 import GraphSeparationSR16Dataset, GraphSeparationSR16Walker
from .graph_separation_sr25 import GraphSeparationSR25Dataset, GraphSeparationSR25Walker
from .regression_counting import RegressionCountingDataset, RegressionCountingWalker
from .graph_classification_reddit_treads import GraphCLSRedditDataset,GraphCLSRedditWalker
from .graph_classification_IMDBM import GraphCLSIMDBMDataset,GraphCLSIMDBMWalker
from .graph_classification_ENZYMES import GraphCLSENZYMESDataset,GraphCLSENZYMESWalker
from .graph_classification_github_stargazers import GraphCLSGitStarDataset,GraphCLSGitStarWalker
from .graph_classification_ogbg_molhiv import GraphCLS_OGBGHIV_Dataset,GraphCLS_OGBGHIV_Walker
from .graph_classification_ogbg_molpcba import GraphCLS_OGBGPCBA_Dataset,GraphCLS_OGBGPCBA_Walker
from .graph_classification_ogbg_molppa import GraphCLS_OGBGPPA_Dataset,GraphCLS_OGBGPPA_Walker
from torch_geometric.datasets import TUDataset


def setup_data_and_walker(dataset: str, root_dir: str, config) -> Tuple[DatasetBuilder, Walker]:
    # pyg datasets
    is_pyg = True
    if dataset == 'graph_separation_csl':
        walker = GraphSeparationCSLWalker(config)
        ds_builder = DatasetBuilder(dataset, is_pyg, GraphSeparationCSLDataset, root_dir, config)
        walker.register_ds_builder(ds_builder)
        return ds_builder, walker
    if dataset == 'graph_separation_sr16':
        walker = GraphSeparationSR16Walker(config)
        ds_builder = DatasetBuilder(dataset, is_pyg, GraphSeparationSR16Dataset, root_dir, config)
        walker.register_ds_builder(ds_builder)
        return ds_builder, walker
    if dataset == 'graph_separation_sr25':
        walker = GraphSeparationSR25Walker(config)
        ds_builder = DatasetBuilder(dataset, is_pyg, GraphSeparationSR25Dataset, root_dir, config)
        walker.register_ds_builder(ds_builder)
        return ds_builder, walker
    if dataset == 'regression_counting':
        walker = RegressionCountingWalker(config)
        ds_builder = DatasetBuilder(dataset, is_pyg, RegressionCountingDataset, root_dir, config)
        walker.register_ds_builder(ds_builder)
        return ds_builder, walker
    
    if dataset == 'graph_classification_reddit_threads':
        walker = GraphCLSRedditWalker(config)
        data_dir = config.data_dir
        ds_builder = DatasetBuilder(dataset, is_pyg, GraphCLSRedditDataset, root_dir, config)
        walker.register_ds_builder(ds_builder)
        return ds_builder, walker
    if dataset == 'graph_classification_IMDB_MULTI':
        walker = GraphCLSIMDBMWalker(config)
        data_dir = config.data_dir
        ds_builder = DatasetBuilder(dataset, is_pyg, GraphCLSIMDBMDataset, root_dir, config)
        walker.register_ds_builder(ds_builder)
        return ds_builder, walker
    if dataset == 'graph_classification_ENZYMES':
        walker = GraphCLSENZYMESWalker(config)
        data_dir = config.data_dir
        ds_builder = DatasetBuilder(dataset, is_pyg, GraphCLSENZYMESDataset, root_dir, config)
        walker.register_ds_builder(ds_builder)
        return ds_builder, walker
    if dataset == 'graph_classification_github_stargazers':
        walker = GraphCLSGitStarWalker(config)
        data_dir = config.data_dir
        ds_builder = DatasetBuilder(dataset, is_pyg, GraphCLSGitStarDataset, root_dir, config)
        walker.register_ds_builder(ds_builder)
        return ds_builder, walker
    if dataset == 'graph_classification_ogbg_molhiv':
        walker = GraphCLS_OGBGHIV_Walker(config)
        data_dir = config.data_dir
        ds_builder = DatasetBuilder(dataset, is_pyg, GraphCLS_OGBGHIV_Dataset, root_dir, config)
        walker.register_ds_builder(ds_builder)
        return ds_builder, walker
    if dataset == 'graph_classification_ogbg_molpcba':
        walker = GraphCLS_OGBGPCBA_Walker(config)
        data_dir = config.data_dir
        ds_builder = DatasetBuilder(dataset, is_pyg, GraphCLS_OGBGPCBA_Dataset, root_dir, config)
        walker.register_ds_builder(ds_builder)
        return ds_builder, walker
    if dataset == 'graph_classification_ogbg_ppa':
        walker = GraphCLS_OGBGPPA_Walker(config)
        data_dir = config.data_dir
        ds_builder = DatasetBuilder(dataset, is_pyg, GraphCLS_OGBGPPA_Dataset, root_dir, config)
        walker.register_ds_builder(ds_builder)
        return ds_builder, walker
    # non-pyg datasets
    is_pyg = False
    raise NotImplementedError(f"Dataset ({dataset}) not supported!")
