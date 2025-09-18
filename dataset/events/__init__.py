from .events_representations import EventRepresentation, VoxelGrid, Histogram, Tencode

def fetch_event_representation(config):
    if config['representation_type'] == 'voxel_grid':
        return VoxelGrid.from_configuration(config)
    elif config['representation_type'] == 'histogram':
        return Histogram.from_configuration(config)
    elif config['representation_type'] == 'tencode':
        return Tencode.from_configuration(config)
    else:
        raise ValueError(f"Unknown event representation: {config['event_representation']}")