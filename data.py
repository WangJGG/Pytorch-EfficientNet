import os
import torch
from torchvision import datasets, transforms


def loaddata(data_dir, batch_size, set_name, input_size, shuffle=True):
    """
    加载数据集并返回相应的数据加载器。
    支持 train、val 和 test 数据集。
    
    - data_dir: 数据集根目录路径。
    - batch_size: 批大小。
    - set_name: 数据集类型，'train'、'val' 或 'test'。
    - net_name: EfficientNet 模型名称，用于确定 input_size。
    - shuffle: 是否打乱数据。
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 使用 ImageFolder 通过目录名自动加载和映射类别
    image_datasets = datasets.ImageFolder(os.path.join(data_dir, set_name), transform=data_transforms[set_name])
    
    # 自动根据目录结构映射类别名称到标签
    dataset_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    
    data_set_size = len(image_datasets)
    
    return dataset_loader, data_set_size
