import torch
import random
import numpy as np

from .events import fetch_event_representation
from .utils import fetch_preprocessing

from .dsec_dataset import load_datasets as load_datasets_dsec, DSEC_HEIGHT, DSEC_WIDTH
from .mvsec_dataset import load_datasets as load_datasets_mvsec, MVSEC_HEIGHT, MVSEC_WIDTH


ALL_HEIGHTS = {
    "dsec": DSEC_HEIGHT,
    "mvsec": MVSEC_HEIGHT,
}

ALL_WIDTHS = {
    "dsec": DSEC_WIDTH,
    "mvsec": MVSEC_WIDTH,
}

def worker_init_fn(worker_id):                                                          
    torch_seed = torch.randint(0, 2**30, (1,)).item()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)

def fetch_dataloader(config_dataloader, args, test=False):
    """ Create the data loader for the corresponding trainign set """

    dataset = config_dataloader['dataset']
    
    if 'datapath' in config_dataloader:
        datapath = config_dataloader['datapath']
    else:
        datapath = args.datapath
    
    if datapath is None:
        raise ValueError("No datapath provided")    

    datasplit = config_dataloader['split']

    slicing = config_dataloader['slicing']

    batch_size = config_dataloader['batch_size'] if 'batch_size' in config_dataloader else 1
    numworkers = config_dataloader['num_workers'] if 'num_workers' in config_dataloader else 1
    load_images = config_dataloader['load_images'] if 'load_images' in config_dataloader else False
    use_voxels = config_dataloader['use_voxels'] if 'use_voxels' in config_dataloader else False
    concatenate_sequences = config_dataloader['concatenate_sequences'] if 'concatenate_sequences' in config_dataloader else False
    sequence_window = config_dataloader['sequence_window'] if 'sequence_window' in config_dataloader else 1
    sequence_step = config_dataloader['sequence_step'] if 'sequence_step' in config_dataloader else 1
    all_events = config_dataloader['all_events'] if 'all_events' in config_dataloader else False
    self_supervised = config_dataloader['self_supervised'] if 'self_supervised' in config_dataloader else False
    hybrid = config_dataloader['hybrid'] if 'hybrid' in config_dataloader else False
    postfix = config_dataloader['postfix'] if 'postfix' in config_dataloader else ""

    overfit = args.overfit

    augmentator = fetch_preprocessing(config_dataloader['preprocessing'])

    datasets = {}

    if dataset == 'mix1':
        datasets_names = ['eventscape', 'mvsec']
        datasets_paths = datapath.split(';')
    elif dataset == 'mix2':
        datasets_names = ['eventscape', 'dense']
        datasets_paths = datapath.split(';')
    elif dataset == 'mix3':
        datasets_names = ['eventscape', 'dsec']
        datasets_paths = datapath.split(';')
    else:
        datasets_names = [dataset]
        datasets_paths = [datapath]

    for dataset_name, dataset_path in zip(datasets_names, datasets_paths):
        _ev_config = config_dataloader['event_representation'].copy()
        _ev_config['height'] = ALL_HEIGHTS[dataset_name] if 'height' not in _ev_config else _ev_config['height']
        _ev_config['width'] = ALL_WIDTHS[dataset_name] if 'width' not in _ev_config else _ev_config['width']
        event_representation = fetch_event_representation(_ev_config)

        if dataset_name == 'mvsec':
            _datasets = load_datasets_mvsec(dataset_path, datasplit, slicing, event_representation, augmentator, load_images, use_voxels, overfit, sequence_window, sequence_step, self_supervised, hybrid, postfix)
        elif dataset_name == 'dsec':
            _datasets = load_datasets_dsec(dataset_path, datasplit, slicing, event_representation, augmentator, load_images, overfit, sequence_window, sequence_step, self_supervised, hybrid, postfix)
        else:
            raise ValueError(f"Dataset {dataset} not supported")
        
        for key in _datasets:
            print(f"Dataset {key} has {len(_datasets[key])} samples")
        
        datasets.update(_datasets)
    
    if concatenate_sequences:
        datasets = {"concatenated": torch.utils.data.ConcatDataset(list(datasets.values()))}

    dataloaders = {}

    for seq in datasets:
        loader = torch.utils.data.DataLoader(datasets[seq], batch_size=batch_size, 
            pin_memory=True, shuffle=not test, num_workers=numworkers, drop_last=True, worker_init_fn = worker_init_fn, persistent_workers=numworkers>0)
        dataloaders[seq] = loader

    return dataloaders