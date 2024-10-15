import os
import shutil
import random

class_map = {
    'buildings': 0,
    'forest': 1,
    'glacier': 2,
    'mountain': 3,
    'sea': 4,
    'street': 5
}

def create_val_set(data_dir, val_split=0.2):
    """
    从训练集中分割出一部分作为验证集，并将其移动到新的验证集文件夹。
    
    参数:
    - data_dir: 包含训练集数据的根目录。
    - val_split: 验证集占训练集的比例，默认是 0.2。
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # 遍历每个类别目录，并根据 class_map 映射
    for class_name, class_label in class_map.items():
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        
        if not os.path.exists(val_class_dir):
            os.makedirs(val_class_dir)
        
        # 获取当前类别下所有图像文件
        images = os.listdir(train_class_dir)
        num_val_images = int(len(images) * val_split)
        
        # 随机选择一部分图像用于验证集
        val_images = random.sample(images, num_val_images)
        
        # 将选中的图像移动到验证集文件夹
        for image in val_images:
            src_path = os.path.join(train_class_dir, image)
            dest_path = os.path.join(val_class_dir, image)
            shutil.move(src_path, dest_path)
        
        print(f"Moved {num_val_images} images from {class_name} to validation set.")

if __name__ == '__main__':
    data_dir = './dataset'  # 替换为数据集的根目录
    val_split = 0.2  # 验证集占训练集比例（20%）
    
    create_val_set(data_dir, val_split)
