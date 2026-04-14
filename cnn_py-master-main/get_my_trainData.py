# -*- coding: utf-8 -*-
'''
从原数据中选取部分数据；
选取数据的title前两个字符在字典WantedClass中；
且各个类别的数量为WantedNum
'''

"""
子数据集抽取脚本。

这个版本主要从原始验证集文件中抽取指定类别、指定数量的样本，
生成当前项目使用的 my_validdata.json。

虽然文件名叫 get_my_trainData.py，
但当前脚本实际读取的是 ValidJsonFile，并输出到 MyValidJsonFile。
这里保留原作者命名与实现，不改动逻辑。
"""

import jieba
import json

# 原始训练集与验证集路径。
TrainJsonFile = 'baike_qa2019/baike_qa_train.json'
ValidJsonFile = 'baike_qa2019/baike_qa_valid.json'
# 抽样后生成的小规模数据集路径。
MyTainJsonFile = 'baike_qa2019/my_traindata.json'
MyValidJsonFile = 'baike_qa2019/my_validdata.json'
StopWordFile = 'stopword.txt'

# 目标类别及其当前已采样数量。
# 字典的键表示允许保留的类别，值用于统计各类别已经抽到多少条。
WantedClass = {'教育': 0, '健康': 0, '生活': 0, '娱乐': 0, '游戏': 0}
# 每个类别希望抽取的样本数。
WantedNum = 1000
# 总共希望抽取的样本数。
numWantedAll = WantedNum * 5


def main():
    """
    从原始验证集里筛选目标类别样本。

    筛选规则：
    - 类别前两个字符必须出现在 WantedClass 中；
    - 每个类别最多保留 WantedNum 条；
    - 累计达到总数量后提前结束。
    """
    # 读取原始验证集的全部 json 行。
    Datas = open(ValidJsonFile, 'r', encoding='utf_8').readlines()
    # 打开输出文件，写入筛选后的子集。
    f = open(MyValidJsonFile, 'w', encoding='utf_8')

    numInWanted = 0
    for line in Datas:
        data = json.loads(line)
        # 只截取 category 前两个字符作为一级标签。
        cla = data['category'][0:2]
        if cla in WantedClass and WantedClass[cla] < WantedNum:
            # 保持原始 json 结构不变，直接写出。
            json_data = json.dumps(data, ensure_ascii=False)
            f.write(json_data)
            f.write('\n')
            # 更新当前类别计数和全局计数。
            WantedClass[cla] += 1
            numInWanted += 1
            if numInWanted >= numWantedAll:
                # 当总数达到目标后，提前结束，减少不必要的遍历。
                break


if __name__ == '__main__':
    main()
