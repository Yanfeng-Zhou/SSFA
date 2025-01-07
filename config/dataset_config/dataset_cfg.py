import numpy as np

def dataset_cfg(dataet_name):

    config = {
        'SNEMI3D':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'MEAN': [0.501840],
                'STD': [0.189555],
                'SIZE': 256,
                'PALETTE': list(np.array([
                    [255, 255, 255],  # background
                    [0, 0, 0],  # membrane
                ]).flatten()),
                'TRAIN_RATIO': 8,
                'VAL_RATIO': 4,
            },
        'CREMI':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'MEAN': [0.503733],
                'STD': [0.111173],
                'SIZE': 256,
                'PALETTE': list(np.array([
                    [255, 255, 255],
                    [0, 0, 0],
                ]).flatten()),
                'TRAIN_RATIO': 16,
                'VAL_RATIO': 8,
            },
        'CBMI_ER':
            {
                'IN_CHANNELS': 1,
                'NUM_CLASSES': 2,
                'MEAN': [0.083338],
                'STD': [0.112200],
                'SIZE': 256,
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten()),
                'TRAIN_RATIO': 1,
                'VAL_RATIO': 1,
            },
        'STARE_DRIVE':
            {
                'IN_CHANNELS': 3,
                'NUM_CLASSES': 2,
                'MEAN': [0.553565, 0.306110, 0.114089],
                'STD': [0.351092, 0.191511, 0.106024],
                'SIZE': 256,
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten()),
                'TRAIN_RATIO': 2,
                'VAL_RATIO': 1,
            },
    }

    return config[dataet_name]
