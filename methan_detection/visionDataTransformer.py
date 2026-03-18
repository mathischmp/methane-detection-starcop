import albumentations as A
from albumentations.pytorch import ToTensorV2
from .utils import load_config

class VisionDataTransformer:
    def __init__(self):
        config = load_config()
        self.n_swir = config['training']['n_swir']
        

    def data_augmentation_pipeline(self):
        transform = A.Compose([A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5), 
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                ToTensorV2()], additional_targets={'mag1c': 'image', 'swir': 'image'})
        
        return transform 
    
    
    def transform(self, image, mag1c, swir=None, mask=None):

        kwargs = {
        'image': image,
        'mag1c': mag1c,
        'mask': mask
    }
    
        if swir is not None:
            kwargs['swir'] = swir

        transform = self.data_augmentation_pipeline()
        augmented = transform(**kwargs)

        return augmented['image'], augmented['mag1c'], augmented.get('swir'), augmented['mask']
    
    
    def transform_for_validation(self, image, mag1c, swir=None,  mask=None):
        
        val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ToTensorV2()
        ], additional_targets={'mag1c': 'image', 'swir': 'image'})
        
        kwargs = {
                'image': image,
                'mag1c': mag1c,
                'mask': mask
            }
            
        if swir is not None:
            kwargs['swir'] = swir
                
        augmented = val_transform(**kwargs)
                
        return augmented['image'], augmented['mag1c'], augmented.get('swir'), augmented['mask']