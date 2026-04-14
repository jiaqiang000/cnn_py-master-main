"""
训练脚本。

这个文件负责把已经向量化好的训练数据送入 TextCNN 模型进行迭代训练，
并周期性输出损失、保存权重和记录日志。

整体流程如下：
1. 读取词表和标签表，得到模型需要的词表大小与类别数量。
2. 初始化 TextCNN 网络；如果已经存在权重文件，则直接加载继续训练。
3. 构造训练集 DataLoader，并额外读取验证集文本（当前版本仅读取，未参与评估）。
4. 使用 Adam 优化器和 NLLLoss 损失函数进行训练。
5. 每个 epoch 结束后保存最新模型参数。
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from model import textCNN
import sen2inds
import textCNN_data

# 从词表文件中读取 “词 -> 索引” 与 “索引 -> 词” 的映射。
# 这里的词表是在预处理阶段由训练数据统计得到的。
word2ind, ind2word = sen2inds.get_worddict('wordLabel.txt')
# 读取标签与数字编号之间的映射关系，例如 “教育 -> 0”。
label_w2n, label_n2w = sen2inds.read_labelFile('label.txt')
# DataLoader 相关参数。
# batch_size 决定每次训练喂入多少条样本；
# shuffle=True 表示每个 epoch 都打乱数据顺序。
dataLoader_param = {
    'batch_size': 128,
    'shuffle': True,
}


def load_vectorized_data(file_path):
    """
    读取已经向量化好的数据文件。

    每一行格式：
    label,idx1,idx2,...,idx20,
    """
    rows = open(file_path, 'r').read().split('\n')
    rows = list(filter(None, rows))
    labels = []
    sentences = []
    for row in rows:
        items = list(filter(None, row.split(',')))
        items = [int(x) for x in items]
        labels.append(items[0])
        sentences.append(items[1:])
    return labels, sentences


def evaluate(net, device, file_path='valdata_vec.txt'):
    """
    在验证集上计算准确率。
    """
    labels, sentences = load_vectorized_data(file_path)
    total = len(labels)
    right = 0

    net.eval()
    with torch.no_grad():
        for label, sentence in zip(labels, sentences):
            sentence = torch.tensor(sentence, dtype=torch.long, device=device).unsqueeze(0)
            predict = net(sentence).argmax(dim=1).item()
            if predict == label:
                right += 1
    net.train()

    return right / total if total else 0.0


def build_model_params(kernel_num=16, dropout=0.5):
    """
    根据实验参数构造 TextCNN 配置。
    """
    return {
        'vocab_size': len(word2ind),
        'embed_dim': 60,
        'class_num': len(label_w2n),
        'kernel_num': kernel_num,
        'kernel_size': [3, 4, 5],
        'dropout': dropout,
    }


def load_experiment_config(config_path):
    """
    如果输出目录中已经存在实验配置，则优先复用，便于继续训练。
    """
    if not os.path.exists(config_path):
        return None
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_experiment_config(config_path, config):
    """
    保存本次训练的关键超参数和数据文件路径，便于后续测试复现。
    """
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def parse_args():
    """
    解析命令行参数，便于控制训练轮数和是否从头开始训练。
    """
    parser = argparse.ArgumentParser(description='Train TextCNN for text classification.')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--kernel-num', type=int, default=16, help='number of kernels for each filter size')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--train-file', default='traindata_vec.txt', help='vectorized training file')
    parser.add_argument('--val-file', default='valdata_vec.txt', help='vectorized validation file')
    parser.add_argument(
        '--output-dir',
        default='outputs',
        help='directory for logs and saved weights',
    )
    parser.add_argument(
        '--from-scratch',
        action='store_true',
        help='ignore existing weight.pkl and initialize a new model',
    )
    parser.add_argument(
        '--save-checkpoints',
        action='store_true',
        help='save an extra checkpoint file after every epoch',
    )
    return parser.parse_args()


def main():
    """
    训练入口函数。

    该函数不接收外部参数，直接依赖当前目录下已经准备好的词表、标签和向量化数据文件。
    """
    args = parse_args()

    # 根据当前环境自动选择 GPU 或 CPU。
    # 如果机器支持 CUDA，则优先在 GPU 上训练；否则退回 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # add 2024.11.10
    print('init net...')

    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, 'config.json')
    old_config = load_experiment_config(config_path) if not args.from_scratch else None
    if old_config:
        experiment_config = old_config
    else:
        experiment_config = {
            'epochs': args.epochs,
            'lr': args.lr,
            'kernel_num': args.kernel_num,
            'dropout': args.dropout,
            'train_file': args.train_file,
            'val_file': args.val_file,
        }
        save_experiment_config(config_path, experiment_config)

    textCNN_param = build_model_params(
        kernel_num=experiment_config['kernel_num'],
        dropout=experiment_config['dropout'],
    )
    # 创建模型实例。模型的网络结构由 textCNN_param 决定。
    net = textCNN(textCNN_param)

    # 默认权重文件名。
    # 如果该文件存在，说明之前已经训练过，可以直接继续训练。
    weightFile = os.path.join(args.output_dir, 'weight.pkl')
    bestWeightFile = os.path.join(args.output_dir, 'best_weight.pkl')
    load_candidates = [weightFile, bestWeightFile, 'weight.pkl', 'best_weight.pkl', 'textCNN.pkl']
    load_weight = None
    if not args.from_scratch:
        for candidate in load_candidates:
            if os.path.exists(candidate):
                load_weight = candidate
                break

    if load_weight:
        print('load weight')
        # 加载已有模型参数。
        net.load_state_dict(torch.load(load_weight, map_location=device))
    else:
        # 如果不存在历史权重，则按照 model.py 中定义的方式初始化参数。
        net.init_weight()
    print(net)

    # 将模型移动到前面选定的计算设备上。
    net.to(device)

    # 初始化训练数据。
    # textCNN_dataLoader 会从 traindata_vec.txt 中读取样本，并封装成 DataLoader。
    print('init dataset...')
    dataLoader = textCNN_data.textCNN_dataLoader(
        dataLoader_param,
        train_file=experiment_config['train_file'],
    )

    # 选择优化器与损失函数。
    # 模型输出的是 log_softmax，因此这里使用 NLLLoss 与之对应。
    optimizer = torch.optim.Adam(net.parameters(), lr=experiment_config['lr'])
    criterion = nn.NLLLoss()

    # 训练损失日志文件。
    run_tag = time.strftime('%y%m%d%H')
    log_path = os.path.join(args.output_dir, 'log_{}.txt'.format(run_tag))
    log = open(log_path, 'w')
    log.write('epoch step loss\n')

    # 预留的测试/验证日志文件。
    # 当前版本虽然创建了该文件，但未写入验证结果。
    log_test_path = os.path.join(args.output_dir, 'log_test_{}.txt'.format(run_tag))
    log_test = open(log_test_path, 'w')
    log_test.write('epoch val_acc\n')

    print("training...")
    best_acc = -1.0
    # 按指定轮数训练。
    for epoch in range(args.epochs):
        net.train()
        # DataLoader 每次返回一个 batch：
        # clas 是类别编号，sentences 是已经补齐长度的词索引序列。
        for i, (clas, sentences) in enumerate(dataLoader):
            # 清空上一轮反向传播累计的梯度。
            optimizer.zero_grad()

            # 将输入与标签转换为 LongTensor，再移动到目标设备。
            # 词嵌入层需要 LongTensor 类型的词索引。
            sentences = sentences.type(torch.LongTensor).to(device)  #20241110
            clas = clas.type(torch.LongTensor).to(device)   #20241110

            # 前向传播，得到每个类别的对数概率。
            out = net(sentences)

            # 计算当前 batch 的损失。
            loss = criterion(out, clas)

            # 反向传播并更新参数。
            loss.backward()
            optimizer.step()

            if (i + 1) % 1 == 0:
                # 当前脚本设置为每个 step 都打印一次损失。
                print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())
                data = str(epoch + 1) + ' ' + str(i + 1) + ' ' + str(loss.item()) + '\n'
                log.write(data)

        val_acc = evaluate(net, device, file_path=experiment_config['val_file'])
        log_test.write('{} {:.6f}\n'.format(epoch + 1, val_acc))

        # 每个 epoch 结束后保存最新模型。
        print("save model...")

        # 保存一个固定文件名，便于下次直接加载继续训练。
        torch.save(net.state_dict(), weightFile)
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), bestWeightFile)
        # 如需保留每轮快照，可显式开启该选项。
        if args.save_checkpoints:
            checkpoint_path = os.path.join(
                args.output_dir,
                "model_{}_epoch_{}_step_{}_loss_{:.2f}.pkl".format(
                    run_tag, epoch, i, loss.item()
                ),
            )
            torch.save(net.state_dict(), checkpoint_path)
        print(
            "epoch:",
            epoch + 1,
            "step:",
            i + 1,
            "loss:",
            loss.item(),
            "val_acc:",
            round(val_acc, 6),
            "best_val_acc:",
            round(best_acc, 6),
        )


if __name__ == "__main__":
    main()
