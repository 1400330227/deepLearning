from unittest.mock import inplace

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import jieba

jieba.setLogLevel(jieba.logging.INFO)  # 屏蔽jieba分词时出现的提示信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------------
torch.manual_seed(123)
np.random.seed(10)


# 生成索引编码和词典库
def get_texts_vocab(fn):
    max_len = 0
    sentence_words = []
    vocab_dict = dict()
    with open(fn, encoding='UTF-8') as f:
        lines = list(f)
        for line in lines:
            line = line.strip()
            if line == '' or '---' in line:
                continue
            words = list(jieba.cut(line))
            words = ['<s>'] + words + ['<e>']
            if max_len < len(words):
                max_len = len(words)
            sentence_words.append(words)
            for word in words:
                vocab_dict[word] = vocab_dict.get(word, 0) + 1  # 统计词频
    f.close()
    sorted_vocab_dict = sorted(vocab_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)  # 按词频降序排列
    sorted_vocab_dict = sorted_vocab_dict[:-10]  # 减掉10个低频词
    print(sorted_vocab_dict)
    vocab_word2index = {'<unk>': 0, '<pad>': 1, '<s>': 2, '<e>': 3}
    for word, _ in sorted_vocab_dict:  # 构建词汇的整数编号，从0，1开始
        if not word in vocab_word2index:
            vocab_word2index[word] = len(vocab_word2index)
    return sentence_words, vocab_word2index


def enOneTxt(en_ws):
    ln = len(en_ws)
    texts, labels = [], []
    for pre_k in range(1, ln):  # 输入句子的长度为10
        ps = 0
        pe = pre_k - 1
        txt = en_ws[ps:pe + 1]
        txt = (txt + [1] * (10 - len(txt)))[-10:]
        label = en_ws[pre_k]
        texts.append(txt)
        labels.append(label)
    return texts, labels


def enAllTxts(all_sen_words, vocab_w2i):
    texts, labels = [], []
    for i, words in enumerate(all_sen_words):
        en_words = [vocab_w2i.get(word, 0) for word in words]
        txts, lbs = enOneTxt(en_words)
        texts = texts + txts
        labels = labels + lbs
    texts, labels = torch.LongTensor(texts), torch.LongTensor(labels)
    return texts, labels


path = r'../data/data'
name = r'金庸小说节选.txt'
fn = path + '//' + name
sentence_words, vocab_word2index = get_texts_vocab(fn)  # 构建数据集和编码字典
texts, labels = enAllTxts(sentence_words, vocab_word2index)  # 生成训练数据
vocab_index2word = {index: word for word, index in vocab_word2index.items()}  # 用于解码


# print(vocab_word2index)
# print(sentence_words)
# print(texts)
# print(labels)


class Novel_Model(nn.Module):
    def __init__(self, vocab_size):
        super(Novel_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 256)  # embedding_dim 代表一个索引值对应的行向量
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, batch_first=True, num_layers=1, bidirectional=False)
        self.feature1 = nn.Linear(256, 512)
        self.feature2 = nn.Linear(512, vocab_size) # 多分类问题，一共20个分类

    def forward(self, x):
        out = x # x.shape = [8, 10]
        out = self.embedding(x) # out.shape = [8, 10, 256]
        out, (hidden, cn) = self.lstm(out) # out.shape = [8, 10, 256]
        out = torch.sum(out, dim=1) # out.shape = [8, 256]
        out = self.feature1(out) # out.shape = [8, 512]
        out = nn.ReLU(inplace=True)(out) # out.shape = [8, 512]
        out = self.feature2(out) # out.shape = [8, 20]
        return out


class_dict = dict()
for label in labels:
    lb = label.item()
    class_dict[lb] = class_dict.get(lb, 0) + 1  # 统计各类别词汇出现的频次
weights = []  # 跟dataloader.dataset中的数据行要一一对应
for label in labels:
    lb = label.item()
    weights.append(class_dict[lb])
weights = 1. / torch.FloatTensor(weights)
sampler = WeightedRandomSampler(weights=weights, replacement=True, num_samples=len(labels) * 1000)  # 解决类不平衡问题

dataset = TensorDataset(texts, labels) # texts.shape = [34,10], labels.shape: [34]
dataloader = DataLoader(dataset=dataset, batch_size=128, sampler=sampler, shuffle=False)

novel_model = Novel_Model(vocab_size=len(vocab_word2index)).to(device) # vocab_word2index.size = 20
optimizer = torch.optim.Adam(novel_model.parameters(), lr=0.01)

for ep in range(5):
    for i, (batch_texts, batch_labels) in enumerate(dataloader):
        batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
        batch_out = novel_model(batch_texts)  # torch.Size([128, 10]) ---> torch.Size([128, 1524])

        # print(batch_texts.shape,batch_out.shape)
        # exit(0)
        # torch.Size([128, 10]) torch.Size([128])
        loss = nn.CrossEntropyLoss()(batch_out, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print(ep, round(loss.item(), 8))

torch.save(novel_model, 'novel_model')
torch.save(vocab_word2index, 'vocab_word2index')
torch.save(vocab_index2word, 'vocab_index2word')

novel_model = torch.load('novel_model')
vocab_word2index = torch.load('vocab_word2index')
vocab_index2word = torch.load('vocab_index2word')
novel_model.eval()


def getNextWord(s):  # 给定一个词序列，生成它的下一个词
    words = list(jieba.cut(s))
    words = ['<s>'] + words  # + ['<e>']
    en_words = [vocab_word2index.get(word, 0) for word in words]
    en_words = en_words[len(en_words) - 10:len(en_words)]
    en_words = en_words + [1] * (10 - len(en_words))
    batch_texts = torch.LongTensor(en_words).unsqueeze(0).to(device)
    batch_out = novel_model(batch_texts)
    batch_out = torch.softmax(batch_out, dim=1)
    pre_index = torch.argmax(batch_out, dim=1)
    word = vocab_index2word[pre_index.item()]
    return word


# '杨过' #'郭靖'  #s = '郭靖和黄蓉'   忽必烈
seq = '黄蓉'
while True:  # 生成小说文本
    w = getNextWord(seq)
    if w == '<e>':
        break
    seq = seq + w
print('生成的小说文本：', seq)
