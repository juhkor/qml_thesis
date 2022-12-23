import numpy as np

def conf(yarr,predarr):
    conf_matrix = [[0 for i in range(yarr.shape[1])] for j in range(predarr.shape[1])]
    for y,pred in zip(yarr,predarr):
        max_y = max(y)
        max_pred = max(pred)
        idx_y = np.where(y == max_y)[0][0]
        idx_pred = np.where(pred == max_pred)[0][0]
        conf_matrix[idx_y][idx_pred] += 1
    
    return conf_matrix

y = [[0,1,0],[0,1,0],[0,1,0]]
pred = [[0,1,0],[0,1,0],[0,0,1]]
y = np.array(y)
pred = np.array(pred)

conf = conf(y,pred)
print(conf)

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df_classic = pd.DataFrame(conf, index = [i for i in 'abc'],
                  columns = [i for i in 'abc'])


plt.figure(figsize = (10,7))
sn.heatmap(df_classic, annot=True)
plt.show()