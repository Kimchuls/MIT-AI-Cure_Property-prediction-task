from random import shuffle
import torch
from sklearn import metrics

class Item():
    def __init__(self, obj, label):
        self.obj, self.label = obj, label

class RandomSampler():
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.pos = 0

    def get_batch(self):
        for _ in range(self.batch_size):
            item = self.data[self.pos]
            self.pos += 1
            if self.pos >= len(self.data):
                self.pos = 0
                shuffle(self.data)
            yield item

class Sampler():
    def __init__(self, data, batch_sizes):
        mp = {}
        for item in data:
            idx = item.label
            if idx not in mp:
                mp[idx] = []
            mp[idx].append(item)
        self.iters = {}
        for idx, seq in mp.items():
            self.iters[idx] = RandomSampler(seq, batch_sizes[idx])

    def get_batch(self):
        for it in self.iters.values():
            yield from it.get_batch()



def stat_output(std, pred_label):
    c = metrics.confusion_matrix(std, pred_label)
    return f'tn={c[0][0]},fp={c[0][1]}/tp={c[1][1]},fn={c[1][0]}'

def separate_items(items):
    objs, labels = [], []
    for x in items:
        objs.append(x.obj)
        labels.append(x.label)
    return objs, labels
