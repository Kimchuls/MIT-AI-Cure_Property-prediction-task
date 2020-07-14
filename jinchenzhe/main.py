import numpy as np
import torch
from model import *
from models import *
from evaluate import *
from train import *

def load_data(model, path, files):
    raw = []
    for name in files:
        datafile = open(path + name, "r")
        str = datafile.read().splitlines()
        str = str[1: ]
        mp = {}
        for line in str:
            smiles, label = line.split(',')
            mp[smiles] = int(label)
        raw.append(mp)
        datafile.close()
    data = []
    for csv in raw:
        buf = []
        for smiles, label in csv.items():
            obj = model.process(smiles) #from model.py
            buf.append(Item(obj, label)) #item defined in evaluate.py
        data.append(buf)
    return data

def process_fold(batch_size = 64, positive_percentage = 0.5, model_type = "gcn"):
    roc_record = []
    prc_record = []
    time_record = []
    cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    foldrange = 10
    printf(f"the model type is {model_type}\n")
    for fold in range(foldrange):
        # data[0]: training set, data[1]: validate set, data[2]: test set
        printf(f"fold_{fold} is running")
        start_time = time.time()
        if model_type in ["dense_gcn", "dense_sage", "dense_cheb"]:
            model = Models(device = cuda0, model_type = model_type).to(cuda0)
        else:
            model = Model(device=cuda0, model_type=model_type).to(cuda0)
        data = load_data(model, "./train_cv/fold_" + str(fold) + "/", ['train.csv', 'dev.csv', 'test.csv']) #in main.py
        print(f"fold {fold} load is done.")
        vali_batch, vali_label = separate_items(data[1])  #from evaluate.py
        test_batch, test_label = separate_items(data[2]) #from evaluate.py
        train_data = data[0] + data[1]
        num_positive = round(batch_size * positive_percentage)
        num_negative = batch_size - num_positive
        sampler = Sampler(train_data, {0: num_negative, 1: num_positive}) #model from evaluate.py
        #print(f"fold {fold} prepare is done.")
        train_model(model, sampler, len(train_data), vali_batch, vali_label, batch_size=batch_size)
        #print(f"fold {fold} train is done.")
        roc_auc, prc_auc, pred_label = evaluate_model(model, test_batch, test_label)
        printf(f'[fold {fold}] test: {stat_output(test_label, pred_label)}\n')#
        printf(f'[fold {fold}] ROC-AUC: {roc_auc}')#
        printf(f'[fold {fold}] PRC-AUC: {prc_auc}')#
        roc_record.append(roc_auc)
        prc_record.append(prc_auc)
        time_record.append(time.time() - start_time)

    printf(f'fold_id\tPOC-AUC\t\t\tPRC-AUC')
    for i in range(foldrange):
        printf(f'{i}\t{roc_record[i]}\t{prc_record[i]}')
    x, y = sum(roc_record) / foldrange, sum(prc_record) / foldrange
    a = (sum([(x - t) ** 2 for t in roc_record]) / foldrange) ** 0.5
    b = (sum([(y - t) ** 2 for t in prc_record]) / foldrange) ** 0.5
    printf(f'ROC-AUC = \t{x} ± {a}\nPRC-AUC = \t{y} ± {b}')

    for i in range(foldrange):
        printf(f'fold_{i} cost \t{time_record[i]} \tseconds.')
    printf(f'the total cost is \t{sum(time_record)}')

if __name__ == '__main__':
    process_fold(model_type = "sage")