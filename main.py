import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
from model.efficientnet import EfficientNet
from train import train_model,exp_lr_scheduler
from test import test_model


use_gpu = torch.cuda.is_available()
pth_map = {
        'efficientnet-b0': 'efficientnet-b0-355c32eb.pth',
        'efficientnet-b1': 'efficientnet-b1-f1951068.pth',
        'efficientnet-b2': 'efficientnet-b2-8bb594d6.pth',
        'efficientnet-b3': 'efficientnet-b3-5fb5a3c3.pth',
        'efficientnet-b4': 'efficientnet-b4-6ed6700e.pth',
        'efficientnet-b5': 'efficientnet-b5-b6417697.pth',
        'efficientnet-b6': 'efficientnet-b6-c76e70fd.pth',
        'efficientnet-b7': 'efficientnet-b7-dcc49843.pth',
    }
# 根据 EfficientNet 的版本动态设置输入大小
input_size_map = {
    'efficientnet-b0': 224,
    'efficientnet-b1': 240,
    'efficientnet-b2': 260,
    'efficientnet-b3': 300,
    'efficientnet-b4': 380,
    'efficientnet-b5': 456,
    'efficientnet-b6': 528,
    'efficientnet-b7': 600
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default='./dataset/', help='path of /dataset/')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--class-num', type=int, default=6, help='class num')
    parser.add_argument('--weights-loc', type=str, default=None, help='path of weights')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--net-name', type=str, default='efficientnet-b0', help='efficientnet type')
    parser.add_argument('--resume-epoch', type=int, default=0, help='what epoch to start from')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--log-dir', type=str, default='./logs/exp1', help='path of /logs/')
    parser.add_argument('--patience',type=int,default=7, help='earlystopping counts')
    opt = parser.parse_args()
    print(opt.__dict__)
    print('-' * 50)
    
    data_dir = opt.data_dir
    num_epochs = opt.num_epochs
    batch_size = opt.batch_size
    class_num = opt.class_num
    weights_loc = opt.weights_loc
    lr = opt.lr
    net_name = opt.net_name
    epoch_to_resume_from = opt.resume_epoch
    momentum = opt.momentum
    log_dir = opt.log_dir
    patience=opt.patience
    input_size = input_size_map[net_name]
    '''------run------'''
    if weights_loc is not None:
        model_ft = torch.load(weights_loc)
    else:
        if net_name in pth_map:
            model_ft = EfficientNet.from_name(net_name)
            weights_path = os.path.join('./weights/backbone/', pth_map[net_name])
            state_dict = torch.load(weights_path)
            model_ft.load_state_dict(state_dict)
        else:
            raise ValueError(f"Invalid net_name: {net_name}")

    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, class_num)

    criterion = nn.CrossEntropyLoss()
    if use_gpu:
        model_ft = model_ft.cuda()
        criterion = criterion.cuda()

    optimizer = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum, weight_decay=0.0004)

    # 调用 train_model 返回模型和保存的路径
    model_ft, model_out_path = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs,
                                           data_dir=data_dir, batch_size=batch_size, epoch_to_resume_from=epoch_to_resume_from,
                                           log_dir=log_dir,input_size=input_size,patience=patience)

    print('-' * 50)
    print('Test Accuracy:')

    # 使用保存的最优模型权重进行测试
    model_ft.load_state_dict(torch.load(model_out_path))  # 加载最优权重
    test_model(model_ft, criterion, data_dir=data_dir, batch_size=4,input_size=input_size)
