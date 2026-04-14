#-*- coding: utf_8 -*-

"""
文本预处理与向量化脚本。

这个文件承担从原始 json 文本到数字化训练样本的关键转换工作：
1. 读取标签映射和词表映射；
2. 对标题文本进行 jieba 分词；
3. 去掉停用词与词表外单词；
4. 将文本映射为固定长度的词索引序列；
5. 写入 traindata_vec.txt 或 valdata_vec.txt。

这里默认处理的是 trainFile 指向的数据文件。
如果想处理验证集，需要手动切换上方的文件路径配置。
"""

import json
import sys, io
import random

try:
    import jieba
except ImportError:
    jieba = None

# 将标准输出编码调整为 gb18030。
# 这是原作者为了兼容某些 Windows 终端环境所做的设置。
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')

# 当前默认处理的原始数据文件。
trainFile = 'baike_qa2019/my_traindata.json'
#trainFile = 'baike_qa2019/my_validdata.json'
stopwordFile = 'stopword.txt'
wordLabelFile = 'wordLabel.txt'
# 当前默认输出的向量化文件。
trainDataVecFile = 'traindata_vec.txt'
#trainDataVecFile = 'valdata_vec.txt'
# 文本最大长度。由于第一位还要存标签，所以实际每行总长度是 maxLen + 1。
maxLen = 20

labelFile = 'label.txt'


def ensure_jieba():
    """
    只有在需要重新分词与向量化时才要求 jieba 可用。
    """
    if jieba is None:
        raise ImportError(
            'jieba is required for preprocessing. Install it before running json2txt().'
        )


def read_labelFile(file):
    """
    读取标签文件。

    标签文件格式：
    类别名 类别编号

    返回：
    - label_w2n: 类别名 -> 编号
    - label_n2w: 编号 -> 类别名
    """
    data = open(file, 'r', encoding='utf_8').read().split('\n')
    label_w2n = {}
    label_n2w = {}
    for line in data:
        line = line.split(' ')
        name_w = line[0]
        name_n = int(line[1])
        label_w2n[name_w] = name_n
        label_n2w[name_n] = name_w

    return label_w2n, label_n2w


def read_stopword(file):
    """
    读取停用词表并按行返回。
    """
    data = open(file, 'r', encoding='utf_8').read().split('\n')

    return data


def get_worddict(file):
    """
    读取词表文件，构造词和编号之间的双向映射。

    词表文件格式：
    词 索引 词频
    """
    datas = open(file, 'r', encoding='utf_8').read().split('\n')
    datas = list(filter(None, datas))
    word2ind = {}
    for line in datas:
        line = line.split(' ')
        word2ind[line[0]] = int(line[1])
    
    ind2word = {word2ind[w]:w for w in word2ind}
    return word2ind, ind2word


def json2txt():
    """
    将原始 json 行文本转成模型训练用的整数序列文件。

    输出文件中每一行的格式为：
    标签编号,词索引1,词索引2,...,词索引20,
    """
    ensure_jieba()

    # 读取标签映射与词表映射。
    label_dict, label_n2w = read_labelFile(labelFile)
    word2ind, ind2word = get_worddict(wordLabelFile)

    # 打开输出文件。
    traindataTxt = open(trainDataVecFile, 'w')
    stoplist = read_stopword(stopwordFile)

    # 读取全部原始数据行并打乱顺序。
    datas = open(trainFile, 'r', encoding='utf_8').read().split('\n')
    datas = list(filter(None, datas))
    random.shuffle(datas)

    for line in datas:
        line = json.loads(line)
        # 当前项目仅使用标题 title 字段做分类。
        title = line['title']
        # 取 category 前两个字符作为一级类别，例如 “教育”“健康”。
        cla = line['category'][0:2]
        cla_ind = label_dict[cla]

        # 对标题进行分词。
        title_seg = jieba.cut(title, cut_all=False)
        # 每条样本的第一个位置先放入类别编号。
        title_ind = [cla_ind]
        for w in title_seg:
            # 跳过停用词。
            if w in stoplist:
                continue
            # 如果词不在词表中，直接丢弃，避免索引越界。
            if word2ind.get(w, -1) == -1:   #add 2024 1110  for new words error
                continue      #add 2024 1110
            # 原始作者额外排除了编号为 35451 的词。
            # 这里保持原有逻辑，不改变其行为。
            if word2ind[w] == 35451:
                continue
            title_ind.append(word2ind[w])

        # length 包含了最前面的标签位。
        length = len(title_ind)
        if length > maxLen + 1:
            # 过长则截断，只保留前 maxLen 个词以及标签位。
            title_ind = title_ind[0:21]
        if length < maxLen + 1:
            # 过短则补 0 到固定长度，便于后续批量训练。
            title_ind.extend([0] * (maxLen - length + 1))

        # 将整条样本按逗号分隔写回文本文件。
        for n in title_ind:
            traindataTxt.write(str(n) + ',')
        traindataTxt.write('\n')


def main():
    """脚本入口。"""
    json2txt()


if __name__ == "__main__":
    main()
