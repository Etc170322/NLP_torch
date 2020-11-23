import os
import json
import torch
from tqdm import tqdm
from collections import defaultdict
import pickle as pkl
import time
from datetime import timedelta
import jieba

def build_vocab(file_path, use_words, max_size, min_freq):
    vocab_dic = defaultdict(int)
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            if use_words:
                tokens = list(jieba.cut(content))
            else:
                tokenizer = lambda x: [y for y in x]
                tokens = tokenizer(content)
            for word in tokens:
                vocab_dic[word] = vocab_dic[word] + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({'UNK': len(vocab_dic), 'PAD': len(vocab_dic) + 1})
    return vocab_dic

def read_config(path):
    with open(path,'r') as f:
        config = json.load(f)
    return config

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def build_dataset(config, use_words):
    if os.path.exists(config.get('data','vocab_path')):
        vocab = pkl.load(open(config.get('data','vocab_path'), 'rb'))
    else:
        vocab = build_vocab(config.get('data','train_path'), use_words=use_words, max_size=int(config.get('model','max_vocabulary')), min_freq=1)
        pkl.dump(vocab, open(config.get('data','vocab_path'), 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                if use_words:
                    tokens = list(jieba.cut(content))
                else:
                    tokenizer = lambda x: [y for y in x]
                    tokens = tokenizer(content)
                seq_len = len(tokens)
                if pad_size:
                    if len(tokens) < pad_size:
                        tokens.extend(['PAD'] * (pad_size - len(tokens)))
                    else:
                        tokens = tokens[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in tokens:
                    words_line.append(vocab.get(word, vocab.get('UNK')))
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.get('data','train_path'), int(config.get('model','max_length')))
    dev = load_dataset(config.get('data','dev_path'), int(config.get('model','max_length')))
    test = load_dataset(config.get('data','test_path'), int(config.get('model','max_length')))
    return vocab, train, dev, test

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = torch.device(device)

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, int(config.get('model','batch_size')), config.get('model','device'))
    return iter