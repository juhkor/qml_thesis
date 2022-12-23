import random
import os
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.random import set_seed
import torch

seed = 1337 #0.9236871004104614 / 0.9225929975509644

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
set_seed(seed)

labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

def full_window():
    x_valid = np.load('/datasets/equal_x_valid.npy')
    q_valid = np.load('/datasets/equal_q_valid.npy')
    y_valid = np.load('/datasets/equal_y_valid.npy')

    return x_valid,q_valid,y_valid


def conf(yarr,predarr):
    conf_matrix = [[0 for i in range(yarr.shape[1])] for j in range(predarr.shape[1])]
    for y,pred in zip(yarr,predarr):
        max_y = max(y)
        max_pred = max(pred)
        idx_y = np.where(y == max_y)[0][0]
        idx_pred = np.where(pred == max_pred)[0][0]
        conf_matrix[idx_y][idx_pred] += 1
    
    return conf_matrix    


def get_predictions(model,data):
    predict = model.predict(data,16)
    
    return predict
    
    
def get_df(matrix, labels):
    df = pd.DataFrame(matrix, index = [i for i in labels],columns = [i for i in labels])
    
    return df


def plot_heatmap(df,labels,model):
    plt.figure(figsize = (20,12))
    sn.heatmap(df, annot=True, fmt='g')
    plt.savefig(f"{labels}_{model}_tot_err.png")
    #plt.show()


def reduce_labels(xs,qs,ys,to_drop):
    #X = data[:, [1, 9]]
    new_x=[]
    new_q=[]
    new_y=[]
    drop = []
    for name in labels:
        if name not in to_drop:
            idx = label_to_index(name)
            drop.append(idx)
    for x,q,y in zip(xs,qs,ys):
        idx = np.where(y == 1)
        name = index_to_label(idx[0][0])
        if name not in to_drop:
            new_x.append(x)
            new_q.append(q)
            new_y.append(y)

    new_y = np.array(new_y)
    new_y = new_y[:,drop]
    return np.array(new_x),np.array(new_q),new_y
    
    
def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def main():
    x_valid,q_valid,y_valid = full_window()

    classic_model = load_model('best_classic.hdf5')
    classic_predict = get_predictions(classic_model,x_valid)
    classic_conf_matrix = conf(y_valid,classic_predict)
    df_classic = get_df(classic_conf_matrix,labels)
    plot_heatmap(df_classic)

    qcnn_model = load_model('best_qcnn.hdf5')
    qcnn_predict = get_predictions(qcnn_model,q_valid)
    qcnn_conf_matrix = conf(y_valid,qcnn_predict)
    df_qcnn = get_df(qcnn_conf_matrix,labels)
    plot_heatmap(df_qcnn)

    diagonal = np.diag(classic_conf_matrix)
    column_sums = np.sum(classic_conf_matrix, axis=0)
    column_sums -= diagonal
    min_value = min(column_sums)
    min_idx = np.where(column_sums == min_value)[0][0]
    to_drop = [labels[min_idx]]
    print(labels[min_idx])

    print(y_valid.shape)
    print(x_valid.shape)
    x_valid, q_valid, y_valid = reduce_labels(x_valid, q_valid, y_valid, to_drop)
    print(y_valid.shape)
    print(x_valid.shape)
