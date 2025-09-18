from .dpt import RecurrentDepthAnythingV2
import torch


def get_rec_depth_anything_v2(checkpoint_path = '', encoder = None, activation = 'relu', scale_factor = 1.0, inv_prediction = False, input_size_width = 518, input_size_height = 518, freeze_encoder = False, input_channels=3, recurrent_block_type='convlstm', nopretrain=False, _state_dict = None):
    assert encoder in [None, 'vits', 'vitb', 'vitl', 'vitg'], "Select a valid ViT encoder"

    if encoder is None:
        # Try to infer the encoder from the checkpoint name
        if 'vits' in checkpoint_path:
            encoder = 'vits'
        elif 'vitb' in checkpoint_path:
            encoder = 'vitb'
        elif 'vitl' in checkpoint_path:
            encoder = 'vitl'
        elif 'vitg' in checkpoint_path:
            encoder = 'vitg'
        else:
            raise ValueError("Could not infer the ViT encoder from the checkpoint path")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    rec_depth_anything = RecurrentDepthAnythingV2(**model_configs[encoder], activation=activation, scale_factor=scale_factor, inv_prediction=inv_prediction, input_size_width=input_size_width, input_size_height=input_size_height, freeze_encoder=freeze_encoder, input_channels=input_channels, recurrent_block_type=recurrent_block_type)
    if not nopretrain:
        state_dict = torch.load(checkpoint_path, map_location='cpu') if _state_dict is None else _state_dict
        rec_depth_anything.load_state_dict(state_dict['state_dict'] if 'state_dict' in state_dict else state_dict, strict=False)
    
    return rec_depth_anything
