# Brain Tumor Dataset for SimMIM pre-training and classification
import os
import random
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class RGBConverter:
    """Convert image to RGB mode if it's not already in RGB."""
    def __call__(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img


class MaskGenerator:
    def __init__(self, input_size=224, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask


class BrainTumorSimMIMDataset(Dataset):
    """Dataset for SimMIM pre-training on brain tumor images"""
    
    def __init__(self, root_dir, transform=None, img_size=224, mask_patch_size=32, 
                 model_patch_size=4, mask_ratio=0.6):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        # Collect all images from Training and Testing directories
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(subdir, file))
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = T.Compose([
                RGBConverter(),
                T.RandomResizedCrop(img_size, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
            ])
        
        self.mask_generator = MaskGenerator(
            input_size=img_size,
            mask_patch_size=mask_patch_size,
            model_patch_size=model_patch_size,
            mask_ratio=mask_ratio,
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        # Apply transformations
        img = self.transform(image)
        mask = self.mask_generator()
        
        return img, mask, img_path


class BrainTumorClassificationDataset(Dataset):
    """Dataset for brain tumor classification"""
    
    def __init__(self, root_dir, split='train', transform=None, val_ratio=0.2):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train' or 'val' to specify the dataset split
            transform (callable, optional): Optional transform to be applied on a sample.
            val_ratio (float): Ratio of validation set size
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.val_ratio = val_ratio
        
        # Define class mapping
        self.classes = sorted([d for d in os.listdir(os.path.join(root_dir, 'Training')) 
                              if os.path.isdir(os.path.join(root_dir, 'Training', d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Collect all images and labels
        self.image_paths = []
        self.labels = []
        
        # Process Training directory
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, 'Training', class_name)
            if os.path.isdir(class_dir):
                for file in os.listdir(class_dir):
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_dir, file))
                        self.labels.append(self.class_to_idx[class_name])
        
        # Process Testing directory if available
        test_dir = os.path.join(root_dir, 'Testing')
        if os.path.exists(test_dir):
            for class_name in self.classes:
                class_dir = os.path.join(test_dir, class_name)
                if os.path.isdir(class_dir):
                    for file in os.listdir(class_dir):
                        if file.endswith(('.jpg', '.jpeg', '.png')):
                            self.image_paths.append(os.path.join(class_dir, file))
                            self.labels.append(self.class_to_idx[class_name])
        
        # Create a deterministic train/val split
        indices = list(range(len(self.image_paths)))
        random.seed(42)  # For reproducibility
        random.shuffle(indices)
        split_idx = int(len(indices) * (1 - self.val_ratio))
        
        if split == 'train':
            self.indices = indices[:split_idx]
        else:  # 'val'
            self.indices = indices[split_idx:]
            
        # Default transform if none provided
        if self.transform is None:
            if split == 'train':
                self.transform = T.Compose([
                    RGBConverter(),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
                ])
            else:
                self.transform = T.Compose([
                    RGBConverter(),
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
                ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        true_idx = self.indices[idx]
        img_path = self.image_paths[true_idx]
        image = Image.open(img_path)
        label = self.labels[true_idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def collate_fn(batch):
    """Custom collate function for SimMIM pre-training"""
    imgs = torch.stack([item[0] for item in batch])
    masks = torch.tensor(np.stack([item[1] for item in batch])).float()
    paths = [item[2] for item in batch]
    
    return imgs, masks, paths


def build_simmim_loader(root_dir, img_size=224, mask_patch_size=32, model_patch_size=4, 
                        mask_ratio=0.6, batch_size=32, num_workers=8):
    """Build data loader for SimMIM pre-training"""
    dataset = BrainTumorSimMIMDataset(
        root_dir=root_dir,
        img_size=img_size,
        mask_patch_size=mask_patch_size,
        model_patch_size=model_patch_size,
        mask_ratio=mask_ratio
    )
    
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
        
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    return dataloader


def build_classification_loader(root_dir, batch_size=32, num_workers=8):
    """Build data loaders for classification fine-tuning"""
    transform_train = T.Compose([
        RGBConverter(),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
    ])
    
    transform_val = T.Compose([
        RGBConverter(),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
    ])
    
    train_dataset = BrainTumorClassificationDataset(
        root_dir=root_dir,
        split='train',
        transform=transform_train
    )
    
    val_dataset = BrainTumorClassificationDataset(
        root_dir=root_dir,
        split='val',
        transform=transform_val
    )
    
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_dataset, val_dataset, train_loader, val_loader
