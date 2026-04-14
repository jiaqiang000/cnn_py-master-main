"""
模型测试脚本。

这个文件负责：
1. 构造与训练阶段一致的 TextCNN 网络；
2. 加载已经训练好的模型参数；
3. 逐条读取验证集样本；
4. 输出累计分类准确率。

该脚本采用逐条推理的方式，逻辑直观，便于教学演示。
"""

import os

import numpy as np
import torch

from model import textCNN
import sen2inds

# 读取词表与标签映射，确保测试阶段的编码方式与训练阶段一致。
word2ind, ind2word = sen2inds.get_worddict('wordLabel.txt')
label_w2n, label_n2w = sen2inds.read_labelFile('label.txt')

# 测试时必须与训练时保持完全一致的模型超参数。
textCNN_param = {
    'vocab_size': len(word2ind),
    'embed_dim': 60,
    'class_num': len(label_w2n),
    "kernel_num": 16,
    "kernel_size": [3, 4, 5],
    "dropout": 0.5,
}


def get_valData(file):
    """
    读取验证集向量文件。

    返回值是原始字符串列表，后续在主函数中逐行解析。
    """
    datas = open(file, 'r').read().split('\n')
    datas = list(filter(None, datas))

    return datas


def parse_net_result(out):
    """
    解析网络输出，得到预测标签与对应分数。

    参数：
    - out: 模型输出的一维 numpy 数组，表示各类别的对数概率

    返回：
    - label: 最大分数对应的类别编号
    - score: 该类别的对数概率值
    """
    score = max(out)
    label = np.where(out == score)[0][0]
    
    return label, score


def main():
    """测试入口函数。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化网络结构。
    print('init net...')
    net = textCNN(textCNN_param)

    # 优先读取训练过程中保存的最佳权重，其次读取最新权重。
    weight_candidates = ['best_weight.pkl', 'weight.pkl', 'textCNN.pkl']
    weightFile = None
    for file_name in weight_candidates:
        if os.path.exists(file_name):
            weightFile = file_name
            break

    if weightFile:
        print('load weight')
        # 加载训练好的模型参数。
        net.load_state_dict(torch.load(weightFile, map_location=device))
    else:
        print('No weight file!')
        exit()
    print(net)

    # 测试阶段切换到目标设备并开启 eval 模式。
    # eval 模式会关闭 dropout 等训练时特有的行为。
    net.to(device)
    net.eval()

    # numAll 统计总样本数，numRight 统计预测正确的样本数。
    numAll = 0
    numRight = 0
    confusion = np.zeros((len(label_n2w), len(label_n2w)), dtype=np.int32)
    testData = get_valData('valdata_vec.txt')
    with torch.no_grad():
        for data in testData:
            numAll += 1
            # 每一行格式是：label,idx1,idx2,...,idx20,
            data = list(filter(None, data.split(',')))
            label = int(data[0])
            # 只取后面 20 个位置作为句子输入。
            sentence = np.array([int(x) for x in data[1:21]])
            sentence = torch.from_numpy(sentence)

            # 增加 batch 维后送入模型，并把结果转成 numpy 便于后续处理。
            predict = net(
                sentence.unsqueeze(0).type(torch.LongTensor).to(device)
            ).cpu().detach().numpy()[0]
            label_pre, score = parse_net_result(predict)

            confusion[label][label_pre] += 1
            # score > -100 这个判断在当前逻辑下几乎总是成立，
            # 这里保持原样，仅补充说明。
            if label_pre == label and score > -100:
                numRight += 1
            if numAll % 100 == 0:
                # 每处理 100 条样本输出一次当前累计准确率。
                print('acc:{}({}/{})'.format(numRight / numAll, numRight, numAll))

    print('final_acc:{}({}/{})'.format(numRight / numAll, numRight, numAll))
    print('per_class_acc:')
    for idx in sorted(label_n2w):
        total = confusion[idx].sum()
        right = confusion[idx][idx]
        acc = right / total if total else 0.0
        print('{}: {:.6f} ({}/{})'.format(label_n2w[idx], acc, right, total))

    print('confusion_matrix:')
    print('labels:', [label_n2w[idx] for idx in sorted(label_n2w)])
    print(confusion)


if __name__ == "__main__":
    main()
