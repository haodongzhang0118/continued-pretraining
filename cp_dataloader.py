# Data loading utilities for continued pretraining
import torch
import stable_pretraining as spt
from stable_pretraining.data import transforms
from stable_pretraining.data.transforms import MultiViewTransform
from cp_datasets import get_dataset


def create_transforms(ds_cfg, n_views=1, strong_aug=False):
    """Create training and validation transforms.
    
    Args:
        ds_cfg: Dataset configuration from get_dataset_config() containing:
            - input_size: Target image size
            - normalization: Normalization parameters (mean, std)
        n_views: Number of views for multi-view learning (default: 1)
        strong_aug: If True, use stronger augmentation (default: False)
        
    Returns:
        tuple: (train_transform, val_transform)
            - train_transform: Transform for training (with augmentation)
            - val_transform: Transform for validation/test (no augmentation)
    """
    if strong_aug:
        # Strong augmentation for contrastive learning
        base_aug = transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop(
                (ds_cfg["input_size"], ds_cfg["input_size"]), scale=(0.2, 1.0)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=0.5),
            transforms.ToImage(**ds_cfg["normalization"]),
        )
    else:
        # Standard augmentation
        base_aug = transforms.Compose(
            transforms.RGB(),
            transforms.RandomResizedCrop((ds_cfg["input_size"], ds_cfg["input_size"])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.3
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 2.0), p=0.2),
            transforms.ToImage(**ds_cfg["normalization"]),
        )
    
    # Validation transform (no augmentation)
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.Resize((ds_cfg["input_size"], ds_cfg["input_size"])),
        transforms.ToImage(**ds_cfg["normalization"]),
    )
    
    # Multi-view support for contrastive learning
    train_transform = (
        MultiViewTransform({f"view_{i}": base_aug for i in range(n_views)})
        if n_views > 1
        else base_aug
    )
    
    return train_transform, val_transform


class CPSubset(torch.utils.data.Dataset):
    """Dataset subset for continued pretraining.
    
    Wraps a dataset and provides access to a subset of indices,
    with proper sample_idx tracking.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.dataset[self.indices[idx]]
        if isinstance(sample, dict):
            sample["sample_idx"] = idx
        return sample


def create_data_loaders(args, ds_cfg, train_transform, val_transform, data_dir):
    """Create data loaders using stable-datasets.
    
    Args:
        args: Command-line arguments containing:
            - dataset: Dataset name
            - seed: Random seed for reproducibility
            - n_samples: Number of training samples to use
            - batch_size: Batch size for data loaders
            - num_workers: Number of workers for data loading
        ds_cfg: Dataset configuration from get_dataset_config() containing:
            - splits: List of [train_split, val_split, test_split]
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        data_dir: Cache directory for datasets
        
    Returns:
        tuple: (data_module, test_loader, eval_train_loader, indices)
            - data_module: spt.data.DataModule with train and val loaders
            - test_loader: DataLoader for test set
            - eval_train_loader: DataLoader for evaluation on train set (with val transform)
            - indices: List of training sample indices used
    """
    # Use splits from dataset config to handle datasets without validation split
    splits = ds_cfg.get("splits", ["train", "validation", "test"])
    train_split, val_split, test_split = splits
    
    # Load datasets using stable-datasets through get_dataset()
    # get_dataset returns HFDatasetWrapper with transform already applied
    full_train = get_dataset(
        args.dataset,
        split=train_split,
        transform=train_transform,
        cache_dir=data_dir,
        seed=args.seed,
    )
    val_data = get_dataset(
        args.dataset,
        split=val_split,
        transform=val_transform,
        cache_dir=data_dir,
        seed=args.seed,
    )
    test_data = get_dataset(
        args.dataset,
        split=test_split,
        transform=val_transform,
        cache_dir=data_dir,
        seed=args.seed,
    )

    # Create training subset
    torch.manual_seed(args.seed)
    indices = torch.randperm(len(full_train))[: args.n_samples].tolist()
    train_subset = CPSubset(full_train, indices)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
    )
    data = spt.data.DataModule(train=train_loader, val=val_loader)

    # Create evaluation train loader (with val transform, same indices as training)
    eval_train = get_dataset(
        args.dataset,
        split=train_split,
        transform=val_transform,
        cache_dir=data_dir,
        seed=args.seed,
    )
    eval_train_loader = torch.utils.data.DataLoader(
        CPSubset(eval_train, indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Data: train={len(train_subset)} val={len(val_data)} test={len(test_data)}")
    return data, test_loader, eval_train_loader, indices
