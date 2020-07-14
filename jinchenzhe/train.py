import time
from math import ceil
import torch
from evaluate import *
from sklearn import metrics

opt = open("./output.txt", "w")
def printf(texts):
    opt.write(texts+"\n")

class MaxCount:
    def __init__(self):
        self.maximal = None
        self.count = 0

    def record(self, value):
        if self.maximal is None or self.maximal < value:
            self.maximal = value
            self.count += 1

def train_model(
        model, sampler, train_data_size, vali_batch, vali_label, batch_size,
        score_expression = '(prc_auc, roc_auc, loss)',
        maximal_count = 15, min_iteration = 6, max_iteration = 50,
):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    maxcount = MaxCount()
    min_loss = 1e100
    batch_per_epoch = ceil(train_data_size / batch_size)
    printf(f'batch_per_epoch={batch_per_epoch}')#
    for i in range(max_iteration):  # epochs
        printf(f"train_fold_cycle: {i}")#
        print(f"train_fold_cycle: {i}")
        sum_loss = 0.0
        for _ in range(batch_per_epoch):
            batch, label = separate_items(sampler.get_batch()) #evaluate.py
            label = torch.tensor(label)
            sum_loss += train_one_step(model, optimizer, batch, label)

        loss = sum_loss / batch_per_epoch
        roc_auc, prc_auc, pred_label = evaluate_model(model, vali_batch, vali_label)
        maxcount.record(eval(score_expression, None, {'prc_auc': prc_auc, 'roc_auc': roc_auc, 'loss': loss}))
        printf(f'[cycle {i}] train:    loss={loss},min={min_loss}')#
        printf(f'[cycle {i}] validate: {stat_output(vali_label, pred_label)}. roc={roc_auc},prc={prc_auc}')
        if i >= min_iteration and maxcount.count >= maximal_count:
            break
        min_loss = min(min_loss, loss)

def train_one_step(model, optimizer, batch, label):
    criterion = torch.nn.CrossEntropyLoss()
    pred = model.forward(batch)
    loss = criterion(pred, label)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss.item()

def evaluate_auc(std, pred):
    roc_auc = metrics.roc_auc_score(std, pred)
    prec, recall, _ = metrics.precision_recall_curve(std, pred)
    prc_auc = metrics.auc(recall, prec)
    return roc_auc, prc_auc

def evaluate_model(model, batch, label):
    with torch.no_grad():
        pred = model.predict(batch)
        pred_label = pred.argmax(dim=1)
    roc, prc = evaluate_auc(label, pred[:, 1])
    return roc, prc, pred_label.tolist()