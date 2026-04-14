# -*- coding: utf-8 -*-
'''
从原数据中选取部分数据；
选取数据的title前两个字符在字典WantedClass中；
且各个类别的数量为WantedNum
'''

"""
训练集与验证集联合抽样脚本。

这个脚本的目标是从原始大数据中筛出当前实验使用的小规模数据集：
- 从原始验证集抽取一部分样本写入 my_validdata.json
- 从原始训练集抽取一部分样本写入 my_traindata.json

注意：
WantedClass 这个计数字典在验证集抽样和训练集抽样之间是连续复用的。
因此训练集阶段的计数是建立在验证集阶段结果之上的。
这里仅补充说明，不修改原始行为。
"""

import jieba
import json

# 原始数据路径。
TrainJsonFile = 'baike_qa2019/baike_qa_train.json'
ValidJsonFile = 'baike_qa2019/baike_qa_valid.json'
# 抽样后输出的数据路径。
MyTrainJsonFile = 'baike_qa2019/my_traindata.json'
MyValidJsonFile = 'baike_qa2019/my_validdata.json'
StopWordFile = 'stopword.txt'

# 目标类别及当前累计数量。
WantedClass = {'教育': 0, '健康': 0, '生活': 0, '娱乐': 0, '游戏': 0}



def main():
    """
    先抽验证集，再抽训练集。

    验证集阶段每类抽 1000 条；
    训练集阶段继续累计到每类 5000 条。
    """
    # ---------- 第一步：生成验证集 ----------
    Datas = open(ValidJsonFile, 'r', encoding='utf_8').readlines()
    f = open(MyValidJsonFile, 'w', encoding='utf_8')
    WantedNum = 1000
    numWantedAll = WantedNum * 5
    numInWanted = 0
    for line in Datas:
        data = json.loads(line)
        # 仅保留一级类别在目标列表中的样本。
        cla = data['category'][0:2]
        if cla in WantedClass and WantedClass[cla] < WantedNum:
            json_data = json.dumps(data, ensure_ascii=False)
            f.write(json_data)
            f.write('\n')
            WantedClass[cla] += 1
            numInWanted += 1
            if numInWanted >= numWantedAll:
                break

    # ---------- 第二步：生成训练集 ----------
    Datas = open(TrainJsonFile, 'r', encoding='utf_8').readlines()
    f = open(MyTrainJsonFile, 'w', encoding='utf_8')
    WantedNum = 5000
    numWantedAll = WantedNum * 5
    numInWanted = 0
    for line in Datas:
        data = json.loads(line)
        # 由于 WantedClass 没有重置，这里的比较是基于前一步的累计值。
        cla = data['category'][0:2]
        if cla in WantedClass and WantedClass[cla] < WantedNum:
            json_data = json.dumps(data, ensure_ascii=False)
            f.write(json_data)
            f.write('\n')
            # 继续更新类别计数，直到每类累计达到 WantedNum。
            WantedClass[cla] += 1
            numInWanted += 1
            if numInWanted >= numWantedAll:
                break

if __name__ == '__main__':
    main()
