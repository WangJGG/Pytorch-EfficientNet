import torch
from torch.autograd import Variable
from data import loaddata  # 引入数据加载模块

use_gpu = torch.cuda.is_available()

def test_model(model, criterion, data_dir='', batch_size=4,input_size=224):

    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    running_corrects = 0
    dset_loader, dset_size = loaddata(data_dir=data_dir, batch_size=batch_size, set_name='test',input_size=input_size, shuffle=False)

    for data in dset_loader:
        inputs, labels = data
        labels = torch.squeeze(labels.type(torch.LongTensor))
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs.data, 1)
        running_corrects += torch.sum(preds == labels.data)

    loss = running_loss / dset_size
    accuracy = running_corrects.double() / dset_size
    print(f'Final Test Loss: {loss:.4f} Acc: {accuracy:.4f}')
