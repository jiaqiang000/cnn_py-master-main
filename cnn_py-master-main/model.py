"""
TextCNN 模型定义文件。

这里实现的是一个典型的文本卷积神经网络：
1. 先把词索引映射为词向量；
2. 使用多个不同窗口大小的卷积核提取局部 n-gram 特征；
3. 对每一路卷积结果执行最大池化，保留最显著特征；
4. 将多路特征拼接后送入全连接层完成分类。
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class textCNN(nn.Module):
    """
    TextCNN 文本分类模型。

    参数 param 是一个字典，至少需要包含：
    - vocab_size: 词表大小
    - embed_dim: 词向量维度
    - class_num: 类别数
    - kernel_num: 每种卷积核的输出通道数
    - kernel_size: 卷积窗口大小列表，例如 [3, 4, 5]
    - dropout: dropout 概率
    """
    def __init__(self, param):
        super(textCNN, self).__init__()
        # TextCNN 把一句话表示成二维张量后再做二维卷积，
        # 因此输入通道数固定为 1。
        ci = 1  # input chanel size
        # 每种卷积核对应多少个输出通道，也就是能学习多少组局部特征。
        kernel_num = param['kernel_num'] # output chanel size
        kernel_size = param['kernel_size']
        vocab_size = param['vocab_size']
        embed_dim = param['embed_dim']
        dropout = param['dropout']
        class_num = param['class_num']
        self.param = param

        # 词嵌入层：把词编号映射为稠密向量。
        # padding_idx=1 表示索引为 1 的词在训练时不会更新梯度。
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)

        # 三组不同尺寸的卷积核：
        # (3, embed_dim) 表示一次看连续 3 个词；
        # (4, embed_dim) 表示一次看连续 4 个词；
        # (5, embed_dim) 表示一次看连续 5 个词。
        # 卷积核宽度等于 embed_dim，意味着每次卷积都覆盖词向量的全部维度。
        self.conv11 = nn.Conv2d(ci, kernel_num, (kernel_size[0], embed_dim))
        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_size[1], embed_dim))
        self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_size[2], embed_dim))

        # Dropout 用于减少过拟合。
        self.dropout = nn.Dropout(dropout)
        # 全连接层：将三路卷积特征拼接后映射到类别空间。
        self.fc1 = nn.Linear(len(kernel_size) * kernel_num, class_num)

    def init_embed(self, embed_matrix):
        """
        用外部预训练词向量初始化嵌入层。

        参数：
        - embed_matrix: 形状通常为 [vocab_size, embed_dim] 的二维矩阵
        """
        self.embed.weight = nn.Parameter(torch.Tensor(embed_matrix))

    @staticmethod
    def conv_and_pool(x, conv):
        """
        对某一路卷积执行 “卷积 -> ReLU -> 最大池化”。

        参数：
        - x: 输入张量，形状约为 (batch, 1, sentence_length, embed_dim)
        - conv: 某一个卷积层

        返回：
        - 池化后的张量，形状为 (batch, kernel_num)
        """
        # x: (batch, 1, sentence_length,  )
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        """
        前向传播。

        参数：
        - x: 形状为 (batch, sentence_length) 的词索引张量

        返回：
        - 每个类别的对数概率，形状为 (batch, class_num)
        """
        # x: (batch, sentence_length)
        x = self.embed(x)
        # x: (batch, sentence_length, embed_dim)
        # TODO init embed matrix with pre-trained

        # Conv2d 期望的输入是四维，因此在通道维上扩展一维。
        x = x.unsqueeze(1)
        # x: (batch, 1, sentence_length, embed_dim)

        # 分别用不同尺度的卷积核提取特征。
        x1 = self.conv_and_pool(x, self.conv11)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv12)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv13)  # (batch, kernel_num)

        # 将三路特征按特征维拼接起来。
        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)

        # dropout 后接分类层。
        x = self.dropout(x)
        logit = F.log_softmax(self.fc1(x), dim=1)
        return logit

    def init_weight(self):
        """
        自定义参数初始化。

        卷积层使用与 He 初始化类似的正态分布；
        线性层使用较小方差的正态分布；
        偏置统一初始化为 0。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层按感受野大小和输出通道数估算初始化方差。
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # 当前模型没有显式使用 BatchNorm2d，这里保留了通用初始化逻辑。
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # 线性层使用较小标准差初始化，避免初始输出过大。
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
