# coding: UTF-8
# 我当然第一步就是，打开model的文件夹，来瞄一瞄bert是怎么敲出来了，会不会有几十万行代码，结果打开一看：
# 这短短的几十行代码，告诉我，我又可以开心的调包了，难怪之前师兄说，bert感觉没啥厉害的，模型你又不能动，参数又不能随便改。果然就是这样，
# 貌似能动的就是有配置参数下面的几个东东了。

import torch
import torch.nn as nn
# 这两行，大家可能只能百度到被注释那一行的包，那个就是提供bert预训练模型的包，一般使用也都只是调用这两个包，一般使用以下两行代码，
# 来调用bert去做预训练（也就是去处理你下载好的预训练模型）
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert'
        # ./THUCNews/data文件夹下拿训练数据
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 3  # epoch数，默认为3轮
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './bert_pretrain'
        # 加载bert的分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)  # 到./bert_pretrain拿预训练好的数据
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # 加载bert模型，这个路径文件夹下有bert_config.json配置文件和model.bin模型权重文件
        # 关于bert模型的参数配置都在你下好的预训练文件中的json文件里面，这个一般是不能修改的。
        self.bert = BertModel.from_pretrained(config.bert_path)  # 加载模型
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度
        self.fc = nn.Linear(config.hidden_size, config.num_classes)   # 全连接层:768 -> 2

    def forward(self, x):
        context = x[0]  # 输入的句子   (ids, seq_len, mask)
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        # 控制是否输出所有encoder层的结果
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)  # 得到10分类
        return out
        # 可以发现，bert模型的定义由于高效简易的封装库存在，使得定义模型较为容易，如果想要在bert之后加入cnn/rnn等层，可在这里定义。
