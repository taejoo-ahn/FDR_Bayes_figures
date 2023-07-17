import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=np.load("fig3data.npy",allow_pickle=True)
d1=data[0]
d2=data[1]

alphas=[0.01+i/100 for i in range(40)]
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams['text.usetex'] = True

fig, axs = plt.subplots(1, 2, figsize=(9,4))
fig.tight_layout(pad=4.0)
axs[0].tick_params(axis='both', labelsize=9)
axs[1].tick_params(axis='both', labelsize=9)


axs[0].grid(color='grey', linestyle='dotted', linewidth=0.1)
for i in range(4):
    axs[0].scatter(alphas,d2[0][i],color='blue',s=2,marker='o',alpha=0.5)
    axs[0].scatter(alphas,d2[2][i],color='red',s=20,marker='2',alpha=0.5)
    axs[0].scatter(alphas,d2[4][i],color='orange',s=20,marker='+',alpha=0.5)
axs[0].scatter(alphas,d2[0][4],color='blue',s=2,marker='o',alpha=0.5,label='Well-specified model')
axs[0].scatter(alphas,d2[2][4],color='red',s=20,marker='2',alpha=0.5,label='Misspecified noise level')
axs[0].scatter(alphas,d2[4][4],color='orange',s=20,marker='+',alpha=0.5,label='Misspecified prior')
axs[0].plot(alphas,[0.9*i for i in alphas],color='black',linewidth=1.5)

axs[0].set_title('$\delta=0.8$, $n=400$, $d=500$',fontsize=12)
axs[0].set_xticks(ticks=[0.0, 0.1, 0.2, 0.3, 0.4])
axs[0].set_yticks(ticks=[0.0, 0.1, 0.2, 0.3, 0.4])
#axs[0].legend(fontsize=11, loc='upper left', framealpha=0.6, frameon=True)
axs[0].tick_params(axis='both', length=0, pad=5)

axs[1].grid(color='grey', linestyle='dotted', linewidth=0.1)
for i in range(4):
    axs[1].scatter(alphas,d1[0][i],color='blue',s=2,marker='o',alpha=0.5)
    axs[1].scatter(alphas,d1[2][i],color='red',s=20,marker='2',alpha=0.5)
    axs[1].scatter(alphas,d1[4][i],color='orange',s=20,marker='+',alpha=0.5)
axs[1].scatter(alphas,d1[0][4],color='blue',s=2,marker='o',alpha=0.5,label='Well-specified model')
axs[1].scatter(alphas,d1[2][4],color='red',s=20,marker='2',alpha=0.5,label='Misspecified noise level')
axs[1].scatter(alphas,d1[4][4],color='orange',s=20,marker='+',alpha=0.5,label='Misspecified prior')
axs[1].plot(alphas,[0.9*i for i in alphas],color='black',linewidth=1.5)

axs[1].set_title('$\delta=1.25$, $n=500$, $d=400$',fontsize=12)
axs[1].set_xticks(ticks=[0.0, 0.1, 0.2, 0.3, 0.4])
axs[1].set_yticks(ticks=[0.0, 0.1, 0.2, 0.3, 0.4])
axs[1].legend(fontsize=11, loc='upper left', framealpha=0.6, frameon=True)
axs[1].tick_params(axis='both', length=0, pad=5)


axs[0].set_xlabel("",fontsize=11)
axs[0].set_ylabel("",fontsize=11)
axs[1].set_xlabel("",fontsize=11)
axs[1].set_ylabel("",fontsize=11)
for ax in axs.flat:
    ax.set(xlabel=r'$\alpha$', ylabel='FDP')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
    
    
plt.savefig('fig3_1.pdf')


alphas=[0.01+i/100 for i in range(40)]
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams['text.usetex'] = True

fig, axs = plt.subplots(1, 2, figsize=(9,4))
fig.tight_layout(pad=4.0)
axs[0].tick_params(axis='both', labelsize=9)
axs[1].tick_params(axis='both', labelsize=9)


axs[0].grid(color='grey', linestyle='dotted', linewidth=0.1)
for i in range(4):
    axs[0].scatter(alphas,d2[1][i],color='blue',s=2,marker='o',alpha=0.5)
    axs[0].scatter(alphas,d2[3][i],color='red',s=20,marker='2',alpha=0.5)
    axs[0].scatter(alphas,d2[5][i],color='orange',s=20,marker='+',alpha=0.5)
axs[0].scatter(alphas,d2[1][4],color='blue',s=2,marker='o',alpha=0.5,label='Well-specified model')
axs[0].scatter(alphas,d2[3][4],color='red',s=20,marker='2',alpha=0.5,label='Misspecified noise level')
axs[0].scatter(alphas,d2[5][4],color='orange',s=20,marker='+',alpha=0.5,label='Misspecified prior')
axs[0].plot(d2[6][1],d2[6][0],color='black',linewidth=1.5,label='Theoretical TPR')

axs[0].set_title('$\delta=0.8$, $n=400$, $d=500$',fontsize=12)
axs[0].set_xticks(ticks=[0.0, 0.1, 0.2, 0.3, 0.4])
axs[0].set_yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8,1])
#axs[0].legend(fontsize=11, loc='lower right', framealpha=0.6, frameon=True)
axs[0].tick_params(axis='both', length=0, pad=5)

axs[1].grid(color='grey', linestyle='dotted', linewidth=0.1)
for i in range(4):
    axs[1].scatter(alphas,d1[1][i],color='blue',s=2,marker='o',alpha=0.5)
    axs[1].scatter(alphas,d1[3][i],color='red',s=20,marker='2',alpha=0.5)
    axs[1].scatter(alphas,d1[5][i],color='orange',s=20,marker='+',alpha=0.5)
axs[1].scatter(alphas,d1[1][4],color='blue',s=2,marker='o',alpha=0.5,label='Well-specified model')
axs[1].scatter(alphas,d1[3][4],color='red',s=20,marker='2',alpha=0.5,label='Misspecified noise level')
axs[1].scatter(alphas,d1[5][4],color='orange',s=20,marker='+',alpha=0.5,label='Misspecified prior')
axs[1].plot(d1[6][1],d1[6][0],color='black',linewidth=1.5,label='Theoretical TPR')

axs[1].set_title('$\delta=1.25$, $n=500$, $d=400$',fontsize=12)
axs[1].set_xticks(ticks=[0.0, 0.1, 0.2, 0.3, 0.4])
axs[1].set_yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8,1])
axs[1].legend(fontsize=11, loc='lower right', framealpha=0.6, frameon=True)
axs[1].tick_params(axis='both', length=0, pad=5)


axs[0].set_xlabel("",fontsize=11)
axs[0].set_ylabel("",fontsize=11)
axs[1].set_xlabel("",fontsize=11)
axs[1].set_ylabel("",fontsize=11)
for ax in axs.flat:
    ax.set(xlabel=r'$\alpha$', ylabel='TPP')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
    
    
plt.savefig('fig3_2.pdf')