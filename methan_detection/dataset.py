import os
import rasterio
import torch
from .visionDataTransformer import VisionDataTransformer
import numpy as np

class Dataset:
    def __init__(self, config, labels, training_type, transform = False , test = False):
        self.labels = labels
        self.transform = transform
        self.visionAugmentation = VisionDataTransformer()
        self.config = config
        self.training_type = training_type
        self.test = test

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        event_id = self.labels['id'].values[idx]
        qplume = self.labels['qplume'].values[idx]
        rgb,mag1c,gt = self.image_preprocessing(event_id)

        if self.transform:
            rgb, mag1c, gt = self.visionAugmentation.transform(image=rgb, mag1c=mag1c, mask=gt)
        
        else:
            rgb, mag1c, gt = self.visionAugmentation.transform_for_validation(image=rgb, mag1c=mag1c, mask=gt)
        return rgb,mag1c,gt,qplume
    

    def image_preprocessing(self, event_id): 
        if self.test:
            folder = os.path.join('..', self.config['storage']['local_raw_path'], 'STARCOP_test')
        else:
            folder = os.path.join('..', self.config['storage']['local_raw_path'], f'STARCOP_train_{self.training_type}')
        size_read = 300
        
        ft = os.path.join(folder, event_id)
        aviris_r = os.path.join(ft, "TOA_AVIRIS_640nm.tif")
        aviris_g = os.path.join(ft, "TOA_AVIRIS_550nm.tif")
        aviris_b = os.path.join(ft, "TOA_AVIRIS_460nm.tif")
        magic_path = os.path.join(ft, "mag1c.tif")
        # Ground truth:
        gt_path = os.path.join(ft, "labelbinary.tif")

        with rasterio.open(gt_path) as src:

            # Compute shape to read to from pyramids and speed up plotting
            shape = src.shape
            if (size_read >= shape[0]) and (size_read >= shape[1]):
                out_shape = shape
            elif shape[0] > shape[1]:
                out_shape = (size_read, int(round(shape[1]/shape[0] * size_read)))
            else:
                out_shape = (int(round(shape[0] / shape[1] * size_read)), size_read)
            gt = src.read(1, out_shape=out_shape)

        with rasterio.open(magic_path) as src:
            mag1c = src.read(1, out_shape=out_shape)
        with rasterio.open(aviris_r) as src:
            r = src.read(1, out_shape=out_shape)
        with rasterio.open(aviris_g) as src:
            g = src.read(1, out_shape=out_shape)
        with rasterio.open(aviris_b) as src:
            b = src.read(1, out_shape=out_shape)


        rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
        mag1c = np.asarray(mag1c).astype(np.float32)
        gt = np.asarray(gt).astype(np.float32)
        
        return rgb,mag1c,gt