"""
数据集与 DataLoader 定义文件。

这个文件负责读取已经完成向量化的文本分类数据：
- traindata_vec.txt: 训练集
- valdata_vec.txt: 验证集

其中每一行的格式为：
类别编号,词索引1,词索引2,...,词索引20,

第一列是标签，后面固定长度的 20 个整数表示文本的词索引序列。
"""

from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np


# 默认训练集与验证集文件路径。
trainDataFile = 'traindata_vec.txt'
valDataFile = 'valdata_vec.txt'


def get_valdata(file=valDataFile):
    """
    读取验证集原始行数据。

    参数：
    - file: 验证集文件路径，默认使用 valDataFile

    返回：
    - 打乱顺序后的字符串列表，每个元素对应文件中的一行
    """
    valData = open(valDataFile, 'r').read().split('\n')
    valData = list(filter(None, valData))
    random.shuffle(valData)

    return valData


class textCNN_data(Dataset):
    """
    自定义训练数据集。

    该类会把 traindata_vec.txt 中的每一行解析为：
    - cla: 整数类别编号
    - sentence: 词索引组成的一维 numpy 数组
    """
    def __init__(self):
        # 直接把整个训练文件读入内存。
        # 对于当前这个小型教学项目，这种实现足够简单直接。
        trainData = open(trainDataFile, 'r').read().split('\n')
        trainData = list(filter(None, trainData))
        # 在数据集初始化时先随机打乱一次。
        random.shuffle(trainData)
        self.trainData = trainData

    def __len__(self):
        """返回训练样本总数。"""
        return len(self.trainData)

    def __getitem__(self, idx):
        """
        读取指定索引的一条样本。

        参数：
        - idx: 样本下标

        返回：
        - cla: 标签编号
        - sentence: 文本的词索引序列
        """
        data = self.trainData[idx]
        # 每一行以逗号分隔，行尾也带逗号，因此这里要先过滤空字符串。
        data = list(filter(None, data.split(',')))
        data = [int(x) for x in data]
        # 第一项是标签。
        cla = data[0]
        # 后续所有项是固定长度的词索引序列。
        sentence = np.array(data[1:])

        return cla, sentence



def textCNN_dataLoader(param):
    """
    构造训练用 DataLoader。

    参数：
    - param: 包含 batch_size 与 shuffle 的字典

    返回：
    - PyTorch DataLoader 对象
    """
    dataset = textCNN_data()
    batch_size = param['batch_size']
    shuffle = param['shuffle']
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # 这个测试入口仅用于快速验证数据集读取是否正常。
    dataset = textCNN_data()
    cla, sen = dataset.__getitem__(0)

    print(cla)
    print(sen)
