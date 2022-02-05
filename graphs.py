import numpy as np
import matplotlib.pyplot as plt

gnn_file="/home/ruizhe/Code/random_circuits_remove_empty/11-28-2021_170231rollback_True_relax_2_cost.npy"
chong_file="/home/ruizhe/Code/random_circuits_remove_empty/cost_relax2_chong_relax2.npy"
title='QFT'
ylabel='rOEE Computation Time (s)'
xlabel='Number of Qbits'


gnn=np.load(gnn_file,allow_pickle=True)
chong=np.load(chong_file,allow_pickle=True)

gnn_nlc=[np.sum(i) for i in gnn[3]]
chong_nlc=[np.sum(i) for i in chong[2]]
qbits=range(50,101)
plt.subplots(1,1)
plt.subplot(111)
plt.scatter(qbits,gnn_nlc)
plt.scatter(qbits,chong_nlc,marker='s')
plt.plot(qbits,gnn_nlc)
plt.plot(qbits,chong_nlc)
plt.legend(['GNN','Baker'])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.show()

print(np.mean(gnn_nlc))
print(np.mean(chong_nlc))