自然语言处理作业-文本分类实践

### 这个项目就是：

先从百科问答里抽出教育、健康、生活、娱乐、游戏这 5 类数据，再对标题分词、去停用词、建立词表、把标题转成固定长度的数字序列，然后送进 model.py 里定义的 TextCNN 训练，最后用 test.py 在验证集上统计准确率。

第一层：数据怎么来
get_my_train_and_validData.py：从原始数据抽样
get_wordlists.py：建词表
sen2inds.py：把文本变数字
第二层：模型怎么学
textCNN_data.py：加载数字数据
model.py：定义 TextCNN
train.py：训练模型
test.py：测试模型


本地环境类

这些是你本地开发环境里的东西，不是项目算法主线。

__pycache__/：Python 运行后自动生成的缓存。
.idea/：PyCharm / IDEA 的项目配置。
venv/：Python 虚拟环境，放依赖包。
数据目录
baike_qa2019/：项目实际用的数据目录，里面现在有 my_traindata.json 和 my_validdata.json 两个抽样后的数据文件。
my_traindata.json：抽样后的训练集。
my_validdata.json：抽样后的验证集。
说明类文件
.gitignore：告诉 Git 哪些文件不要提交。仓库里有这个文件，但不参与模型逻辑。
外层 README.md：仓库总说明。
内层 README.md：项目简短说明，写的是 textCNN_pytorch。
数据准备脚本
get_my_trainData.py：从原始验证集里抽指定 5 类、指定数量的数据，生成 my_validdata.json。名字叫 trainData，但当前脚本实际在做 valid 数据抽样。
get_my_train_and_validData.py：一起抽训练集和验证集；先抽验证集，再从原始训练集继续抽，生成 my_validdata.json 和 my_traindata.json。
get_wordlists.py：读取训练集标题，分词、去停用词、统计词频，生成词表 wordLabel.txt。
标签 / 词表 / 停用词
label.txt：类别对照表，把“教育/健康/生活/娱乐/游戏”映射成数字标签。
stopword.txt：停用词表，分词后要过滤掉的词。get_wordlists.py 和 sen2inds.py 会读它。
wordLabel.txt：词表文件，保存“词 → 编号”的映射，给后续数字化和训练使用。
文本转数字
sen2inds.py：把训练集标题分词后转成词编号序列，统一补到固定长度 20，输出 traindata_vec.txt。
traindata_vec.txt：已经数字化后的训练数据，给训练直接读取。
valdata_vec.txt：已经数字化后的验证数据，给验证/测试读取。
数据加载 / 模型 / 训练 / 测试
textCNN_data.py：把 traindata_vec.txt / valdata_vec.txt 读进来，封装成训练要用的数据集和 DataLoader。
model.py：定义 TextCNN 模型本体，也就是 Embedding、卷积、池化、Dropout、全连接这些结构。
train.py：训练主程序，设置超参数，读数据，训练模型，并保存权重。
test.py：测试/验证脚本，读取验证数据，跑模型，统计准确率。
模型文件
weight.pkl：训练保存的模型权重文件，train.py 会读写它。
textCNN.pkl_bak：旧模型备份文件，仓库里有，但不是当前主训练流程默认使用的文件。
一句话串起来

这个项目主线就是：

抽样数据 → 建词表 → 文本转数字 → textCNN_data.py 加载数据 → model.py 定义 TextCNN → train.py 训练 → test.py 测试。