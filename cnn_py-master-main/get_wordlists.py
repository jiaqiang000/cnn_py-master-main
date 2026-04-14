# -*- coding: utf-8 -*-
'''
将训练数据使用jieba分词工具进行分词。并且剔除stopList中的词。
得到词表：
        词表的每一行的内容为：词 词的序号 词的频次
'''

"""
词表构建脚本。

功能：
1. 读取筛选后的训练集 my_traindata.json；
2. 对 title 字段进行分词；
3. 去除停用词；
4. 统计每个词的出现频次；
5. 生成词表文件 wordLabel.txt；
6. 同时统计文本长度分布并写入 length.txt。
"""


import json
import jieba
from tqdm import tqdm

# 输入输出文件配置。
trainFile = 'baike_qa2019/my_traindata.json'
stopwordFile = 'stopword.txt'
wordLabelFile = 'wordLabel.txt'
lengthFile = 'length.txt'


def read_stopword(file):
    """
    读取停用词列表。
    """
    data = open(file, 'r', encoding='utf_8').read().split('\n')

    return data


def main():
    """
    统计词频并写出词表。

    wordLabel.txt 每一行格式：
    词 索引 词频

    其中索引按照词频从高到低依次分配。
    """
    # worddict 用于统计每个词出现的次数。
    worddict = {}
    stoplist = read_stopword(stopwordFile)

    # 读取全部训练样本。
    datas = open(trainFile, 'r', encoding='utf_8').read().split('\n')
    datas = list(filter(None, datas))
    data_num = len(datas)

    # len_dic 用于统计“分词后长度 -> 占比”。
    len_dic = {}
    for line in datas:
        line = json.loads(line)
        title = line['title']
        # 使用精确模式分词。
        title_seg = jieba.cut(title, cut_all=False)
        length = 0
        for w in title_seg:
            # 停用词不参与词频统计。
            if w in stoplist:
                continue
            length += 1
            if w in worddict:
                worddict[w] += 1
            else:
                worddict[w] = 1

        # 统计当前样本的有效长度。
        if length in len_dic:
            len_dic[length] += 1
        else:
            len_dic[length] = 1

    # 按词频从高到低排序，得到最终词表。
    wordlist = sorted(worddict.items(), key=lambda item:item[1], reverse=True)
    f = open(wordLabelFile, 'w', encoding='utf_8')
    ind = 0
    for t in wordlist:
        # 每个词都写出：词、编号、词频。
        d = t[0] + ' ' + str(ind) + ' ' + str(t[1]) + '\n'
        ind += 1
        f.write(d)

    # 把长度统计转换成比例，便于后续确定 maxLen 等参数。
    for k, v in len_dic.items():
        len_dic[k] = round(v * 1.0 / data_num, 3)
    len_list = sorted(len_dic.items(), key=lambda item:item[0], reverse=True)
    f = open(lengthFile, 'w')
    for t in len_list:
        # 写出 “长度 占比”。
        d = str(t[0]) + ' ' + str(t[1]) + '\n'
        f.write(d)

if __name__ == "__main__":
    main()
