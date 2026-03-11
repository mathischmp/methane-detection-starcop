import albumentations as A
from albumentations.pytorch import ToTensorV2


class VisionDataTransformer:
    def __init__(self):
        None

    def data_augmentation_pipeline(self):
    
        transform = A.Compose([A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5), 
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(
        p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ToTensorV2()], additional_targets={'mag1c': 'image'})
        
        return transform 
    
    def transform(self, image, mag1c, mask=None):
        transform = self.data_augmentation_pipeline()
        
        augmented = transform(image=image, mag1c=mag1c, mask=mask)
        
        return augmented['image'], augmented['mag1c'], augmented['mask']