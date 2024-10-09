import jieba
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch.nn as nn

from MHTutorial.unit2.p49 import optimizer

sentences = ['明天去看展览', '今天加班，天气不好', '明天有图书展览', '明天去']

sent_words = []

vocab_dist = dict()

max_len = 0

for sentence in sentences:
    words = list(jieba.cut(sentence))
    # print(words)
    if max_len < len(words):
        max_len = len(words)
    sent_words.append(words)
    # print(sent_words)
    # print('-----------------')

    for word in words:
        vocab_dist[word] = vocab_dist.get(word, 0) + 1

# print(vocab_dist.items())
sort_vocab_dict = sorted(vocab_dist.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
print(sort_vocab_dict)

sort_vocab_dict = sort_vocab_dict[:-2]

vocab_word2index = {'<unk>': 0, '<pad>': 1}
for word, _ in sort_vocab_dict:
    if not word in vocab_word2index:
        vocab_word2index[word] = len(vocab_word2index)
print(vocab_word2index)
print(sent_words) # 语料库


max_len = int(max_len * 0.9) #设置序列的长度
print('--------------------------------')
en_sentences = []

for words in sent_words:
    words = words[:max_len]
    # print(words)
    ent_words = [vocab_word2index.get(word, 0) for word in words]
    ent_words = ent_words + [1] * (max_len - len(ent_words))
    # print(ent_words)
    en_sentences.append(ent_words)


sentences_tensor = torch.LongTensor(en_sentences)

print(len(vocab_word2index))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(vocab_word2index), embedding_dim=20)
        self.lstm = nn.LSTM(input_size=20, hidden_size=28, batch_first=False, bidirectional=False, num_layers=1, bias=True)
        self.fc = nn.Linear(in_features=28, out_features=2)
    def forward(self, x):
        out = self.embedding(x)
        out, (h_n, c_n) = self.lstm(out)
        out = torch.sum(out, dim=1) #按列求和
        out = self.fc(out)

        return out

lstm_model = Model()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
for epoch in range(100):
    for i, (batch_texts, batch_labels) in enumerate(dataloader):
        outputs = lstm_model(batch_texts)
        loss = CrossEntropyLoss()(outputs, batch_labels)
        print(round(loss.item(), 4))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




