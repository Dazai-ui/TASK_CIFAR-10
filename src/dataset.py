import torch
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_transforms(augment: bool = True):
    test_tf=transforms.Compose([ transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    if not augment:
        return test_tf, test_tf
    train_tf=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        transforms.RandomErasing(p=0.5,)])
    return train_tf, test_tf

def get_dataloaders( data_dir: str = "./data", batch_size: int = 128, augment: bool = True, num_workers: int = 2):
    train_tf, test_tf = get_transforms(augment=augment)
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
    