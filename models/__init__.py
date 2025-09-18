import torch

from .dav2 import get_depth_anything_v2
from .rec_dav2 import get_rec_depth_anything_v2

def fetch_model(model_config, args, device=None, test=False, _state_dict=None):
    """ Fetch the model to be used """
    model_name = model_config['model_type']
    model_config = {k:v for k,v in model_config.items() if k != 'model_type'}
    nopretrain = model_config.get('nopretrain', False) if not test else False
    model_config['nopretrain'] = nopretrain

    if 'checkpoint_path' in model_config:
        checkpoint_path = model_config['checkpoint_path']
        model_config = {k:v for k,v in model_config.items() if k != 'checkpoint_path'}
    else:
        checkpoint_path = args.loadmodel

    if model_name == 'DAv2':
        if 'encoder' not in model_config:
            raise ValueError("Encoder not defined in the config file")

        model = get_depth_anything_v2(checkpoint_path, _state_dict=_state_dict, **model_config)
    elif model_name == 'RecDAv2':
        if 'encoder' not in model_config:
            raise ValueError("Encoder not defined in the config file")
        
        model = get_rec_depth_anything_v2(checkpoint_path, _state_dict=_state_dict, **model_config)
    else:
        raise ValueError(f"Model {model_name} not recognized")

    if device is not None:
        model = model.to(device)
    
    if test:
        model.eval()

    return model