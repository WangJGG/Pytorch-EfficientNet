import numpy as np
import torch

class EarlyStopping:
    """早停机制类，监控验证集上的损失，并在损失未下降时提前停止训练"""
    def __init__(self, patience=7, verbose=False, delta=0.0001, path='ckpt/checkpoint.pt'):
        """
        Args:
            patience (int): 当验证集损失未提升的连续次数达到 patience 时停止训练.
            verbose (bool): 是否打印早停信息.
            delta (float): 损失改变量阈值，用于判断是否有显著提升.
            path (str): 最佳模型保存路径.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_acc, model):
        score = val_acc
        self.early_stop=False
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.counter=0
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        '''保存模型'''
        torch.save(model.state_dict(), self.path)
        self.val_acc_max = val_acc
