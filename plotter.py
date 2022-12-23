import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Load saved training histories
classic_histFULL = np.load('training_histories/classic_histFULL.npy',allow_pickle='TRUE').item()
qcnn_histFULL_ALTKERN = np.load('training_histories/qcnn_histFULL_ALTKERN.npy',allow_pickle='TRUE').item()
cconv_histFULL = np.load('training_histories/cconv_histFULL.npy',allow_pickle='TRUE').item()
qcnn_histFULL = np.load('training_histories/qcnn_histFULL.npy',allow_pickle='TRUE').item()
classic_hist = np.load('training_histories/classic_hist.npy',allow_pickle='TRUE').item()
qcnn_hist = np.load('training_histories/qcnn_hist.npy',allow_pickle='TRUE').item()
cconv_hist = np.load('training_histories/cconv_hist.npy',allow_pickle='TRUE').item()
classic_histTENCLASS = np.load('training_histories/classic_histTENCLASS.npy',allow_pickle='TRUE').item()
qcnn_histTENCLASS = np.load('training_histories/qcnn_histTENCLASS.npy',allow_pickle='TRUE').item()

classic_histDATA_REDUCE_FULL = []
classic_histDATA_REDUCE_FULL.append(np.load('training_histories/classic_hist_134_DATA_REDUCE_FULL.npy',allow_pickle='TRUE').item())
classic_histDATA_REDUCE_FULL.append(np.load('training_histories/classic_hist_106_DATA_REDUCE_FULL.npy',allow_pickle='TRUE').item())
classic_histDATA_REDUCE_FULL.append(np.load('training_histories/classic_hist_78_DATA_REDUCE_FULL.npy',allow_pickle='TRUE').item())
classic_histDATA_REDUCE_FULL.append(np.load('training_histories/classic_hist_50_DATA_REDUCE_FULL.npy',allow_pickle='TRUE').item())
classic_histDATA_REDUCE_FULL.append(np.load('training_histories/classic_hist_22_DATA_REDUCE_FULL.npy',allow_pickle='TRUE').item())

qcnn_histDATA_REDUCE_FULL = []
qcnn_histDATA_REDUCE_FULL.append(np.load('training_histories/qcnn_hist_134_DATA_REDUCE_FULL.npy',allow_pickle='TRUE').item())
qcnn_histDATA_REDUCE_FULL.append(np.load('training_histories/qcnn_hist_106_DATA_REDUCE_FULL.npy',allow_pickle='TRUE').item())
qcnn_histDATA_REDUCE_FULL.append(np.load('training_histories/qcnn_hist_78_DATA_REDUCE_FULL.npy',allow_pickle='TRUE').item())
qcnn_histDATA_REDUCE_FULL.append(np.load('training_histories/qcnn_hist_50_DATA_REDUCE_FULL.npy',allow_pickle='TRUE').item())
qcnn_histDATA_REDUCE_FULL.append(np.load('training_histories/qcnn_hist_22_DATA_REDUCE_FULL.npy',allow_pickle='TRUE').item())

data_pointsDATA_REDUCE_FULL = [134,106,78,50,22]

classic_histPARAM_REDUCE_FULL = []
classic_histPARAM_REDUCE_FULL.append(np.load('training_histories/classic_hist_44544_PARAM_REDUCE_FULL.npy',allow_pickle='TRUE').item())
classic_histPARAM_REDUCE_FULL.append(np.load('training_histories/classic_hist_13796_PARAM_REDUCE_FULL.npy',allow_pickle='TRUE').item())
classic_histPARAM_REDUCE_FULL.append(np.load('training_histories/classic_hist_4614_PARAM_REDUCE_FULL.npy',allow_pickle='TRUE').item())
classic_histPARAM_REDUCE_FULL.append(np.load('training_histories/classic_hist_2350_PARAM_REDUCE_FULL.npy',allow_pickle='TRUE').item())
classic_histPARAM_REDUCE_FULL.append(np.load('training_histories/classic_hist_1571_PARAM_REDUCE_FULL.npy',allow_pickle='TRUE').item())

qcnn_histPARAM_REDUCE_FULL = []
qcnn_histPARAM_REDUCE_FULL.append(np.load('training_histories/qcnn_hist_49950_PARAM_REDUCE_FULL.npy',allow_pickle='TRUE').item())
qcnn_histPARAM_REDUCE_FULL.append(np.load('training_histories/qcnn_hist_14822_PARAM_REDUCE_FULL.npy',allow_pickle='TRUE').item())
qcnn_histPARAM_REDUCE_FULL.append(np.load('training_histories/qcnn_hist_4890_PARAM_REDUCE_FULL.npy',allow_pickle='TRUE').item())
qcnn_histPARAM_REDUCE_FULL.append(np.load('training_histories/qcnn_hist_2435_PARAM_REDUCE_FULL.npy',allow_pickle='TRUE').item())
qcnn_histPARAM_REDUCE_FULL.append(np.load('training_histories/qcnn_hist_1832_PARAM_REDUCE_FULL.npy',allow_pickle='TRUE').item())

classic_data_pointsPARAM_REDUCE_FULL = [44544, 13796, 4614, 2350, 1571]
qcnn_data_pointsPARAM_REDUCE_FULL = [49950, 14822, 4890, 2435, 1832]

linestyle_arr = ['solid', 'dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5))]
color = ['b','g','r','c','m','y','k']
line = 2

def plot_acc_loss(points, x_history, q_history, x_label, q_label, fig_title):
    #plt.figure()
    plt.style.use("seaborn-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

    ax1.plot(q_history["val_accuracy"], linestyle=linestyle_arr[0], linewidth=line, color=color[0], label=q_label)
    ax1.plot(x_history["val_accuracy"], linestyle=linestyle_arr[1], linewidth=line, color=color[1], label=x_label)
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(q_history["val_loss"], linestyle=linestyle_arr[0], linewidth=line, color=color[0], label=q_label)
    ax2.plot(x_history["val_loss"], linestyle=linestyle_arr[1], linewidth=line, color=color[1], label=x_label)
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    fig.suptitle(fig_title)
    plt.title('Number of datapoints: '+str(points))
    plt.tight_layout()
    plt.show()

def plot_acc_loss_triple(points, cnn_history, q1_history, q2_history, cnn_label, q1_label, q2_label, fig_title):
    #plt.figure()
    plt.style.use("seaborn-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

    ax1.plot(q1_history["val_accuracy"], linestyle=linestyle_arr[0], linewidth=line, color=color[0], label=q1_label)
    ax1.plot(cnn_history["val_accuracy"], linestyle=linestyle_arr[1], linewidth=line, color=color[1], label=cnn_label)
    ax1.plot(q2_history["val_accuracy"], linestyle=linestyle_arr[2], linewidth=line, color=color[2], label=q2_label)
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(q1_history["val_loss"], linestyle=linestyle_arr[0], linewidth=line, color=color[0], label=q1_label)
    ax2.plot(cnn_history["val_loss"], linestyle=linestyle_arr[1], linewidth=line, color=color[1], label=cnn_label)
    ax2.plot(q2_history["val_loss"], linestyle=linestyle_arr[2], linewidth=line, color=color[2], label=q2_label)
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    fig.suptitle(fig_title)
    plt.title('Number of datapoints: '+str(points))
    plt.tight_layout()
    plt.show()

def plot_acc_loss_arr_dr(points, x_history_arr, q_history_arr, x_label, q_label, fig_title):
    plt.style.use("seaborn-whitegrid")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

    for x_history, q_history, data, linestyle,col in zip(x_history_arr, q_history_arr, points, linestyle_arr,color):
        ax1.plot(x_history["val_accuracy"], linestyle=linestyle, linewidth=1.5, color=col, label='Number of datapoints '+str(data*35))
        ax1.set_title(x_label)
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim([0, 1])
        ax1.set_xlabel("Epoch")
        ax1.legend()
        
        ax2.plot(q_history["val_accuracy"], linestyle=linestyle, linewidth=1.5, color=col, label='Number of datapoints '+str(data*35))
        ax2.set_title(q_label)
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim([0, 1])
        ax2.set_xlabel("Epoch")
        ax2.legend()

        ax3.plot(x_history["val_loss"], linestyle=linestyle, linewidth=1.5, color=col, label='Number of datapoints '+str(data*35))
        ax3.set_ylabel("Loss")
        ax3.set_ylim([0, 8])
        ax3.set_xlabel("Epoch")
        ax3.legend()
        
        ax4.plot(q_history["val_loss"], linestyle=linestyle, linewidth=1.5, color=col, label='Number of datapoints '+str(data*35))
        ax4.set_ylabel("Loss")
        ax4.set_ylim([0, 8])
        ax4.set_xlabel("Epoch")
        ax4.legend()

    fig.suptitle(fig_title)
    plt.title('Training models and reducing number of datapoints')
    plt.tight_layout()
    plt.show()

def plot_acc_loss_arr_pr(c_params_arr, q_params_arr, x_history_arr, q_history_arr, x_label, q_label, fig_title):
    plt.style.use("seaborn-whitegrid")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

    for x_history, q_history, c_param, q_param, linestyle, col in zip(x_history_arr, q_history_arr, c_params_arr, q_params_arr, linestyle_arr, color):
        ax1.plot(x_history["val_accuracy"], linestyle=linestyle, linewidth=1.5, color=col, label='Number of parameters '+str(c_param))
        ax1.set_title(x_label)
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim([0, 1])
        ax1.set_xlabel("Epoch")
        ax1.legend()
        
        ax2.plot(q_history["val_accuracy"], linestyle=linestyle, linewidth=1.5, color=col, label='Number of parameters '+str(q_param))
        ax2.set_title(q_label)
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim([0, 1])
        ax2.set_xlabel("Epoch")
        ax2.legend()

        ax3.plot(x_history["val_loss"], linestyle=linestyle, linewidth=1.5, color=col, label='Number of parameters '+str(c_param))
        ax3.set_ylabel("Loss")
        ax3.set_ylim([0, 8])
        ax3.set_xlabel("Epoch")
        ax3.legend()
        
        ax4.plot(q_history["val_loss"], linestyle=linestyle, linewidth=1.5, color=col, label='Number of parameters '+str(q_param))
        ax4.set_ylabel("Loss")
        ax4.set_ylim([0, 8])
        ax4.set_xlabel("Epoch")
        ax4.legend()

    fig.suptitle(fig_title)
    plt.title('Training models and reducing number of parameters')
    plt.tight_layout()
    plt.show()

def print_stats(classic, qcnn):
    print('Max acc classic:',round(np.max(classic["val_accuracy"])*100,2))
    print('Avg acc classic:',round(np.average(classic["val_accuracy"])*100,2))
    print('Standard error of the mean:',round(stats.sem(classic["val_accuracy"]),4))
    print('Max acc quantum:',round(np.max(qcnn["val_accuracy"])*100,2))
    print('Avg acc quantum:',round(np.average(qcnn["val_accuracy"])*100,2))
    print('Standard error of the mean:',round(stats.sem(qcnn["val_accuracy"]),4))

def print_stats_triple(classic, qcnn1, qcnn2):
    print('Max acc classic:',round(np.max(classic["val_accuracy"])*100,2))
    print('Avg acc classic:',round(np.average(classic["val_accuracy"])*100,2))
    print('Standard error of the mean:',round(stats.sem(classic["val_accuracy"]),4))
    print('Max acc quantum 1:',round(np.max(qcnn1["val_accuracy"])*100,2))
    print('Avg acc quantum 1:',round(np.average(qcnn1["val_accuracy"])*100,2))
    print('Standard error of the mean:',round(stats.sem(qcnn1["val_accuracy"]),4))
    print('Max acc quantum 2:',round(np.max(qcnn2["val_accuracy"])*100,2))
    print('Avg acc quantum 2:',round(np.average(qcnn2["val_accuracy"])*100,2))
    print('Standard error of the mean:',round(stats.sem(qcnn2["val_accuracy"]),4))

def print_stats_arr_dr(points,classic_hist_arr,qcnn_hist_arr):
    print('Parameters & Max & Avg & SEM & Parameters & Max & Avg & SEM \\\\')
    for param,classic,quantum in zip(points,classic_hist_arr,qcnn_hist_arr):
        print(f"{param} & {round(np.max(classic['val_accuracy'])*100,2)} & {round(np.average(classic['val_accuracy'])*100,2)} & {round(stats.sem(classic['val_accuracy']),4)} & {round(np.max(quantum['val_accuracy'])*100,2)} & {round(np.average(quantum['val_accuracy'])*100,2)} & {round(stats.sem(quantum['val_accuracy']),4)}")
        print('\hline')

def print_stats_arr_pr(c_param_arr,q_param_arr,classic_hist_arr,qcnn_hist_arr):
    print('Parameters & Max & Avg & SEM & Parameters & Max & Avg & SEM \\\\')
    print('\hline')
    for c_param,q_param,classic,quantum in zip(c_param_arr,q_param_arr,classic_hist_arr,qcnn_hist_arr):
        print(f"{c_param} & {round(np.max(classic['val_accuracy'])*100,2)} & {round(np.average(classic['val_accuracy'])*100,2)} & {round(stats.sem(classic['val_accuracy']),4)} & {q_param} & {round(np.max(quantum['val_accuracy'])*100,2)} & {round(np.average(quantum['val_accuracy'])*100,2)} & {round(stats.sem(quantum['val_accuracy']),4)}")
        print('\hline')

def print_stats_arr(classic, qcnn):
    print('Min acc classic:',round(np.min(classic)*100,2))
    print('Avg acc classic:',round(np.average(classic)*100,2))
    print('Max acc classic:',round(np.max(classic)*100,2))
    print('Standard error of the mean:',round(stats.sem(classic),4))
    print('Min acc quantum:',round(np.min(qcnn)*100,2))
    print('Avg acc quantum:',round(np.average(qcnn)*100,2))
    print('Max acc quantum:',round(np.max(qcnn)*100,2))
    print('Standard error of the mean:',round(stats.sem(qcnn),4))
    print(f'{round(np.min(classic)*100,2)} & {round(np.average(classic)*100,2)} & {round(np.max(classic)*100,2)} & {round(stats.sem(classic),4)} & {round(np.min(qcnn)*100,2)} & {round(np.average(qcnn)*100,2)} & {round(np.max(qcnn)*100,2)} & {round(stats.sem(qcnn),4)}')

def plot_times():
    plt.style.use("seaborn-whitegrid")
    encoding_times = [0, 1802, 1636, 35]
    training_times = [126, 53, 95, 95]
    prediction_times = [3, 2, 2, 2]
    times = [0,126,3,1802,53,2,1636,95,2,35,95,2]
    x = [1,2,3,4,5,6,7,8,9,10,11,12]
    bars = plt.bar(x,times,tick_label = ['Encode', 'Train', 'Predict', 'Encode', 'Train', 'Predict','Encode', 'Train', 'Predict','Encode', 'Train', 'Predict'],labelsize=14)
    plt.bar_label(bars,times)
    plt.ylabel('Seconds',size='x-large')
    names = ['Without quanvolution','Quanvolution kernel 1','Quanvolution kernel 2','Classic convolution']
    plt.title('Execution times',size='x-large')
    idx = [1,4,7,10]
    for index,data in zip(idx,names):
        plt.text(x=index, y =-200, s=f"{data}", fontdict=dict(fontsize=14))
    plt.show()


def plot_label_reduce_v2():
    plt.style.use("seaborn-whitegrid")
    num_of_classes = [35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2]
    classic_model_acc = [0.90820,0.887551495361328,0.8485036492347717,0.8908973336219788,0.8777590990066528,0.8616946935653687,0.8798194527626038,0.8944948315620422,0.8641975522041321,0.8768075108528137,0.8717353343963623,0.8851209878921509,0.8551778793334961,0.842060923576355,0.8564742803573608,0.8488307595252991,0.8746767640113831,0.8748117685317993,0.8905550837516785,0.8364779949188232,0.8790425062179565,0.8632141947746277,0.8261956572532654,0.8472679257392883,0.8210896253585815,0.7955645322799683,0.8364241719245911,0.8628013730049133,0.870477557182312,0.8150975108146667,0.8237663507461548,0.8729016780853271,0.6940639019012451,0.8410596251487732]
    qcnn_model_acc = [0.88980,0.8782998919487,0.8699389100074768,0.875295877456665,0.8763025403022766,0.9060457348823547,0.8882381916046143,0.8836377859115601,0.8839237689971924,0.8566770553588867,0.8511317372322083,0.8559904098510742,0.8743082880973816,0.8829769492149353,0.864101231098175,0.8858405351638794,0.8635368943214417,0.8535168766975403,0.845792293548584,0.8467272520065308,0.8576521277427673,0.8910779356956482,0.862144410610199,0.8505595922470093,0.8731107115745544,0.8600806593894958,0.883024275302887,0.8541905879974365,0.8748190999031067,0.8346055746078491,0.8902316093444824,0.8788968920707703,0.8630136847496033,0.8443708419799805]
    dropped_labels = ['happy','backward','six','yes','eight','right','seven','nine','zero','visual','sheila','one','stop','house','left','five','three','marvin','tree','two','down','no','learn','wow','on','off','up','go','dog','bird','bed','cat','four','forward']
    fig, ax = plt.subplots(1, 1, figsize=(8, 9))

    ax.plot(num_of_classes,classic_model_acc, 'o', linestyle=linestyle_arr[0], linewidth=line, color=color[0], label='Classic')
    ax.plot(num_of_classes,qcnn_model_acc, 'x', linestyle=linestyle_arr[1], linewidth=line, color=color[1], label='QCNN')
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1])
    ax.set_xticks(num_of_classes)
    ax.set_xlabel("Number of commands")
    ax.legend()
    plt.title('Evolution of validation accuracies as the number of commands decreases',size='x-large')
    plt.show()
    print_stats_arr(classic_model_acc,qcnn_model_acc)

#plot_acc_loss(20000,classic_hist, qcnn_hist, "Attn-BiLSTM without embedding", "Attn-BiLSTM with Quanvolution kernel 1", "Results with and without quanvolution for Mel Spectrogram of size $60 x 54$")
#print_stats(classic_hist,qcnn_hist)

#plot_acc_loss(20000,cconv_hist, qcnn_hist, "Attn-BiLSTM with classical kernel", "Attn-BiLSTM with Quanvolution kernel 1", "Results with convolution and quanvolution for Mel Spectrogram of size $60 x 54$")
#print_stats(cconv_hist,qcnn_hist)

#plot_acc_loss(20000,classic_histFULL, qcnn_histFULL, "Attn-BiLSTM without embedding", "Attn-BiLSTM with Quanvolution kernel 1", "Results with and without quanvolution for Mel Spectrogram of size $60 x 126$")
#print_stats(classic_histFULL,qcnn_histFULL)

#plot_acc_loss_triple(20000,cconv_histFULL, qcnn_histFULL, qcnn_histFULL_ALTKERN, "Attn-BiLSTM with classical convolution", "Attn-BiLSTM with Quanvolution kernel 1", "Attn-BiLSTM with Quanvolution kernel 2", "Comparison of convolution and quanvolution with Mel Spectrogram of size $60 x 126$")
#print_stats_triple(cconv_histFULL,qcnn_histFULL,qcnn_histFULL_ALTKERN)

#plot_acc_loss(20000,cconv_histFULL, qcnn_histFULL, "Attn-BiLSTM with classical convolution", "Attn-BiLSTM with Quanvolution kernel 1", "Results with classical convolution and quanvolution for Mel Spectrogram of size $60 x 126$")
#print_stats(cconv_histFULL,qcnn_histFULL)

#plot_acc_loss(20000,classic_histTENCLASS, qcnn_histTENCLASS, "Attn-BiLSTM without embedding", "Attn-BiLSTM with Quanvolution kernel 1", "Results with and without quanvolution for Mel Spectrogram of size $60 x 126$")
#print_stats(classic_histTENCLASS,qcnn_histTENCLASS)

#plot_acc_loss_arr_dr(data_pointsDATA_REDUCE_FULL, classic_histDATA_REDUCE_FULL, qcnn_histDATA_REDUCE_FULL, "Attn-BiLSTM without Quanv Layer", "Attn-BiLSTM with Quanv Layer", "Testing the effect of reducing datapoints")
#print_stats_arr_dr(data_pointsDATA_REDUCE_FULL,classic_histDATA_REDUCE_FULL,qcnn_histDATA_REDUCE_FULL)

#plot_acc_loss_arr_pr(classic_data_pointsPARAM_REDUCE_FULL, qcnn_data_pointsPARAM_REDUCE_FULL, classic_histPARAM_REDUCE_FULL, qcnn_histPARAM_REDUCE_FULL, "Attn-BiLSTM without Quanv Layer", "Attn-BiLSTM with Quanv Layer", "Testing the effect of reducing parameters")
#print_stats_arr_pr(classic_data_pointsPARAM_REDUCE_FULL,qcnn_data_pointsPARAM_REDUCE_FULL,classic_histPARAM_REDUCE_FULL,qcnn_histPARAM_REDUCE_FULL)

#plot_times()

plot_label_reduce_v2()