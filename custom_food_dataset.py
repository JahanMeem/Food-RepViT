import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class FoodDataset(Dataset):
    def __init__(self, csv_path, img_root, transform=None):
        """
        Args:
            csv_path: Path to the CSV file (train.csv or test.csv)
            img_root: Root directory containing the images
            transform: Optional transform to be applied on images
        """
        self.data = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform
        
        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.data['Food_Label'].unique()))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get image path - already includes subdirectory
        img_path = os.path.join(self.img_root, row['Image_Path'])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get label
        label = self.label_to_idx[row['Food_Label']]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def build_food_dataset(is_train, args):
    """
    Build custom food dataset
    """
    transform = build_food_transform(is_train, args)
    
    if is_train:
        csv_path = '/kaggle/working/food_dataset/train.csv'
    else:
        csv_path = '/kaggle/working/food_dataset/test.csv'
    
    img_root = '/kaggle/working/food_dataset/images'
    
    dataset = FoodDataset(csv_path, img_root, transform=transform)
    nb_classes = dataset.num_classes
    
    return dataset, nb_classes


def build_food_transform(is_train, args):
    """
    Build transforms for food dataset
    """
    resize_im = args.input_size > 32
    
    if is_train:
        # Training transforms with augmentation
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform
    
    # Validation/test transforms
    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(transforms.Resize(size, interpolation=3))
        t.append(transforms.CenterCrop(args.input_size))
    
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    
    return transforms.Compose(t)
