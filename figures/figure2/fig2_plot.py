import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data1=np.load("fig2data_1.npy",allow_pickle=True)
data2=np.load("fig2data_2.npy",allow_pickle=True)

popce_fdp=[row[0] for row in data1[0]]
popce_tpp=[row[1] for row in data1[0]]

poedce_fdp=[row[0] for row in data1[1]]
poedce_tpp=[row[1] for row in data1[1]]

epoedce_fdp=[row[0] for row in data2]
epoedce_tpp=[row[1] for row in data2]

optimals=data1[2]

alphas=[0.01+i/100 for i in range(60)]
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams['text.usetex'] = True

fig, axs = plt.subplots(1, 3, figsize=(14,4))
fig.tight_layout(pad=4.0)
axs[0].tick_params(axis='both', labelsize=11)
axs[1].tick_params(axis='both', labelsize=11)
axs[2].tick_params(axis='both', labelsize=11)


axs[0].grid(color='grey', linestyle='dotted', linewidth=0.1)
for i in range(9):
    axs[0].scatter(alphas,popce_fdp[i],color='blue',s=2,marker='o',alpha=0.5)
    axs[0].scatter(alphas,popce_tpp[i],color='red',s=20,marker='2',alpha=0.5)
axs[0].scatter(alphas,popce_fdp[9],color='blue',s=2,marker='o',label='FDP',alpha=0.5)
axs[0].scatter(alphas,popce_tpp[9],color='red',s=20,marker='2',label='TPP',alpha=0.5)

axs[0].plot(alphas,[0.9*i for i in alphas],color='black',linestyle='dashed',linewidth=1.5,label='Theoretical FDR')
axs[0].plot(optimals[1],optimals[0],color='black',linewidth=1.5,label='Theoretical TPR')

axs[0].set_title('PoPCe',fontsize=13)
axs[0].set_xticks(ticks=[0.0, 0.2, 0.4,0.6])
axs[0].set_yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8,1])
#axs[0].legend(fontsize=9, loc='lower right', framealpha=0.8, frameon=True)
axs[0].tick_params(axis='both', length=0, pad=5)


axs[1].grid(color='grey', linestyle='dotted', linewidth=0.1)
for i in range(9):
    axs[1].scatter(alphas,poedce_fdp[i],color='blue',s=2,marker='o')
    axs[1].scatter(alphas,poedce_tpp[i],color='red',s=20,marker='2')
axs[1].scatter(alphas,poedce_fdp[9],color='blue',s=2,marker='o',label='FDP',alpha=0.5)
axs[1].scatter(alphas,poedce_tpp[9],color='red',s=20,marker='2',label='TPP',alpha=0.5)

axs[1].plot(alphas,[0.9*i for i in alphas],color='black',linestyle='dashed',linewidth=1.5,label='Theoretical FDR')
axs[1].plot(optimals[1],optimals[0],color='black',linewidth=1.5,label='Theoretical TPR')

axs[1].set_title('PoEdCe',fontsize=13)
axs[1].set_xticks(ticks=[0.0, 0.2, 0.4,0.6])
axs[1].set_yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8,1])
#axs[1].legend(fontsize=9, loc='lower right', framealpha=0.8, frameon=True)
axs[1].tick_params(axis='both', length=0, pad=5)



axs[2].grid(color='grey', linestyle='dotted', linewidth=0.1)
for i in range(9):
    axs[2].scatter(alphas[1:60],epoedce_fdp[i][1:60],color='blue',s=2,marker='o')
    axs[2].scatter(alphas[1:60],epoedce_tpp[i][1:60],color='red',s=20,marker='2')
axs[2].scatter(alphas[1:60],epoedce_fdp[9][1:60],color='blue',s=2,marker='o',label='FDP',alpha=0.5)
axs[2].scatter(alphas[1:60],epoedce_tpp[9][1:60],color='red',s=20,marker='2',label='TPP',alpha=0.5)

axs[2].plot(alphas,[0.9*i for i in alphas],color='black',linestyle='dashed',linewidth=1.5,label='Pred. FDR')
axs[2].plot(optimals[1],optimals[0],color='black',linewidth=1.5,label='Pred. TPR')

axs[2].set_title('EPoEdCe',fontsize=13)
axs[2].set_xticks(ticks=[0.0, 0.2, 0.4,0.6])
axs[2].set_yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8,1])
axs[2].legend(fontsize=11, loc='lower right', framealpha=0.8, frameon=True)
axs[2].tick_params(axis='both', length=0, pad=5)



axs[0].set_xlabel("",fontsize=13)
axs[0].set_ylabel("",fontsize=13)
axs[1].set_xlabel("",fontsize=13)
axs[1].set_ylabel("",fontsize=13)
axs[2].set_xlabel("",fontsize=13)
axs[2].set_ylabel("",fontsize=13)
for ax in axs.flat:
    ax.set(xlabel=r'$\alpha$', ylabel='FDP/TPP')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
    
    
plt.savefig('fig2.pdf')