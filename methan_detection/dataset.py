import os
import rasterio
import torch

class Dataset:
    def __init__(self, labels, transform = False):
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        event_id = label['id']
        qplume = label['qplume']
        r,g,b,magic,gt = self.image_preprocessing(event_id, idx)

        if self.transform:
            None #later
        
        return r,g,b,magic,gt,qplume
    

    def image_preprocessing(self, event_id): 
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
            gt = torch.tensor(src.read(1, out_shape=out_shape), dtype=torch.float32)

        with rasterio.open(magic_path) as src:
            magic = torch.tensor(src.read(1, out_shape=out_shape), dtype=torch.float32)
        with rasterio.open(aviris_r) as src:
            r = torch.tensor(src.read(1, out_shape=out_shape), dtype=torch.float32)
        with rasterio.open(aviris_g) as src:
            g = torch.tensor(src.read(1, out_shape=out_shape), dtype=torch.float32)
        with rasterio.open(aviris_b) as src:
            b = torch.tensor(src.read(1, out_shape=out_shape), dtype=torch.float32) 
        
        return r,g,b,magic,gt