import torch
import torch.optim as optim
from torch.autograd import Variable
import time
import os
from model.efficientnet import EfficientNet
from torch.utils.tensorboard import SummaryWriter  # 引入 TensorBoard
from early_stopping import EarlyStopping  # 引入 EarlyStopping
from data import loaddata

use_gpu = torch.cuda.is_available()

def exp_lr_scheduler(optimizer, init_lr=0.01):
    lr = init_lr * 0.1  # 学习率衰减系数 0.8
    print(f'Learning rate is set to {lr}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def train_model(model_ft, criterion, optimizer, lr_scheduler, num_epochs=50, data_dir='', batch_size=2, 
                epoch_to_resume_from=0, patience=7,log_dir='',input_size=224):
    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    best_acc = 0.0
    best_model_wts = model_ft.state_dict()  # 保存最佳模型的权重
    save_dir = os.path.join('weights', 'model')  # 创建保存模型的目录
    os.makedirs(save_dir, exist_ok=True)
    #model_out_path = os.path.join(save_dir, "best_model.pth")  # 模型保存路径
    
    since = time.time()
    
    # 初始化 EarlyStopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epoch_to_resume_from, num_epochs):
        print('-' * 50)
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 50)

        # 训练阶段
        model_ft.train(True)
        dset_loader, dset_size = loaddata(data_dir=data_dir, batch_size=batch_size, set_name='train',input_size=input_size, shuffle=True)

        running_loss = 0.0
        running_corrects = 0

        for data in dset_loader:
            inputs, labels = data
            labels = torch.squeeze(labels.type(torch.LongTensor))
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dset_size
        epoch_acc = running_corrects.double() / dset_size

        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', epoch_acc, epoch)

        # 验证阶段
        model_ft.train(False)
        dset_loader_val, dset_size_val = loaddata(data_dir=data_dir, batch_size=batch_size, set_name='val',input_size=input_size, shuffle=False)
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for data in dset_loader_val:
                inputs, labels = data
                labels = torch.squeeze(labels.type(torch.LongTensor))
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                outputs = model_ft(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs.data, 1)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / dset_size_val
        epoch_val_acc = val_corrects.double() / dset_size_val

        print(f'Validation Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
        writer.add_scalar('Validation Loss', epoch_val_loss, epoch)
        writer.add_scalar('Validation Accuracy', epoch_val_acc, epoch)

        # 更新最佳模型权重
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = model_ft.state_dict()  # 保存最佳模型的权重
            model_out_path = os.path.join(save_dir, f"best_model_epoch{epoch}.pth")  # 更新模型保存路径
            torch.save(best_model_wts, model_out_path)  # 保存最佳模型到文件
            print(f'Saving model and best weights at epoch {epoch}')

        # 调用 EarlyStopping 检查是否需要停止训练
        early_stopping(epoch_val_acc, model_ft)

        if early_stopping.early_stop:
            lr = optimizer.param_groups[0]['lr']
            optimizer = lr_scheduler(optimizer,init_lr=lr)  # 更新学习率
            if optimizer.param_groups[0]['lr']<0.00001:
                print("Early stopping")
                break
            else:
                pass

    # 恢复最优模型权重
    model_ft.load_state_dict(best_model_wts)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    writer.close()
    return model_ft, model_out_path  # 返回训练好的模型和保存的模型路径
