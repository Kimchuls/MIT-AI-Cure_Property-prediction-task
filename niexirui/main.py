import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import keras
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D, BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors

from mol2vec import features
from mol2vec import helpers

import warnings
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from gensim.models import word2vec

from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec


def load_raw_data(path, files):
    res = {}
    for file in files:
        data = pd.read_csv(path + file)
        # print(data.head())
        data = smile2mol(data)
        if not np.isnan(data['activity'][0]):
            draw_count_activity(data, path, file)
        res[file[:-4]] = data

    return res

def draw_count_activity(data, path, file):
    sns.countplot(data=data, x="activity", orient="v")
    plt.ylabel("Count")
    plt.xlabel("Activity")
    plt.title("Count-Activity")
    plt.savefig(path + file[:-4])
    plt.close()


def smile2mol(data):
    data['mol'] = data['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    # 计算分子描述符
    data['tpsa'] = data['mol'].apply(lambda x: Descriptors.TPSA(x))
    data['mol_w'] = data['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
    data['num_valence_electrons'] = data['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
    data['num_heteroatoms'] = data['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))
    return data


def mol2vec(data):
    x = data.drop(columns=['smiles', 'activity', 'mol'])
    model = word2vec.Word2Vec.load('model_300dim.pkl')
    data['sentence'] = data.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
    # Extracting embeddings to a numpy.array
    # Note that we always should mark unseen='UNK' in sentence2vec() so that model is taught how to handle unknown substructures
    data['mol2vec'] = [DfVec(x) for x in sentences2vec(data['sentence'], model, unseen='UNK')]
    x_mol = np.array([x.vec for x in data['mol2vec']])
    x_mol = pd.DataFrame(x_mol)
    # Concatenating matrices of features
    new_data = pd.concat((x, x_mol), axis=1)
    return new_data


def get_x_and_y(data):
    y = data.activity.values
    x = mol2vec(data)
    return x, y

def ROC_Curve(model, title, x_test, y_test):
    preds = model.predict_proba(x_test)[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'g', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    print('ROC AUC score:', round(roc_auc, 4))
    plt.savefig(title + " ROC-AUC")
    plt.close()


def PRC_Curve(model, title, x_test, y_test):
    preds = model.predict_proba(x_test)[:, 1]
    prec, recall, threshold = precision_recall_curve(y_test, preds)
    prc_auc = auc(recall, prec)
    plt.title('PRC Curve')
    plt.plot(recall, prec, 'g', label='PRC = %0.3f' % prc_auc)
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    print('PRC AUC score:', round(prc_auc, 4))
    plt.savefig(title + " PRC-AUC")
    plt.close()


def train(model_choice, title, x_train, x_test, y_train, y_test):
    x_train = StandardScaler().fit_transform(x_train)
    x_test = StandardScaler().fit_transform(x_test)
    if model_choice == "rf":
        model = RandomForestClassifier()
    else:
        model = LogisticRegression(max_iter=150, intercept_scaling=1)
    model.fit(x_train, y_train)
    ROC_Curve(model, model_choice + "-" + title, x_test, y_test)
    PRC_Curve(model, model_choice + "-" + title, x_test, y_test)
    # label("../data/", "test.csv", model, model_choice + "-" + title)


def train_fold(path):
    base_ = "fold_"
    for i in range(10):
        data = load_raw_data(path + base_ + str(i) + "/", files=["test.csv", "train.csv"])
        train_data = data["train"]
        test_data = data["test"]
        x_train, y_train = get_x_and_y(train_data)
        x_test, y_test = get_x_and_y(test_data)
        # print(y_test)
        print("logistic: fold:", i)
        train("lr", base_ + str(i), x_train, x_test, y_train, y_test)
    for i in range(10):
        data = load_raw_data(path + base_ + str(i) + "/", files=["test.csv", "train.csv"])
        train_data = data["train"]
        test_data = data["test"]
        x_train, y_train = get_x_and_y(train_data)
        x_test, y_test = get_x_and_y(test_data)
        print("random_forest: fold:", i)
        train("rf", base_ + str(i), x_train, x_test, y_train, y_test)


def train_complete(path, file):
    data = load_raw_data(path, [file])
    x, y = get_x_and_y(data["train"])
    # print("x:\n", x)
    # print("y:\n", y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
    print("logistic complete data:")
    train("lr", "complete-train-set", x_train, x_test, y_train, y_test)
    print("random forest complete data:")
    train("rf", "complete-train-set", x_train, x_test, y_train, y_test)


def label(path, label_file, model, title):
    data = load_raw_data(path, [label_file])["test"]
    x = data.drop(columns=["smiles", "activity", 'mol'])
    process_model = word2vec.Word2Vec.load('model_300dim.pkl')
    data['sentence'] = data.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
    # Extracting embeddings to a numpy.array
    # Note that we always should mark unseen='UNK' in sentence2vec() so that model is taught how to handle unknown substructures
    data['mol2vec'] = [DfVec(x) for x in sentences2vec(data['sentence'], process_model, unseen='UNK')]
    x_mol = np.array([x.vec for x in data['mol2vec']])
    x_mol = pd.DataFrame(x_mol)
    # Concatenating matrices of features
    x_test = pd.concat((x, x_mol), axis=1)
    x_test = StandardScaler().fit_transform(x_test)
    preds = model.predict_proba(x_test)[:, 1]
    write_data = data.drop(columns=["smiles"])
    # print(type(write_data))
    # print(write_data)
    write_data['activity'] = preds
    # to be finished
    # print(write_data)


def main():
    fold_path = "../data/train_cv/"
    main_path = "../data/"
    train_fold(fold_path)
    train_complete(main_path, "train.csv")
    # train_and_label(main_path, ["train.csv", "test.csv"])


if __name__ == '__main__':
    main()
