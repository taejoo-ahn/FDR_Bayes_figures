import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=np.load("fig1data.npy",allow_pickle=True)


bayes_theo_1=data[0]
lasso_theo_1=data[1]
bayes_data_1=data[2]
lasso_data_1=data[3]
bayes_theo_2=data[4]
lasso_theo_2=data[5]
bayes_data_2=data[6]
lasso_data_2=data[7]


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams['text.usetex'] = True

fig, axs = plt.subplots(1, 2, figsize=(9,4))
fig.tight_layout(pad=4.0)
axs[0].tick_params(axis='both', labelsize=9)
axs[1].tick_params(axis='both', labelsize=9)


axs[0].grid(color='grey', linestyle='dotted', linewidth=0.1)
for k in range(10):
    axs[0].plot(bayes_data_2[0][k],bayes_data_2[1][k],color='blue',linewidth=0.1 ,alpha=0.3)
#    axs[0].plot(d3[2][k],d3[3][k],'darkturquoise',linewidth=0.1)
    axs[0].plot(lasso_data_2[0][k*300:k*300+300],lasso_data_2[1][k*300:k*300+300],color='red',linewidth=0.1, alpha=0.3)
#axs[0].plot(d3[0][9],d3[1][9],color='blue',linewidth=0.1,label="TPoP realization")
#axs[0].plot(d4[0][2700:3000],d4[1][2700:3000],color='red',linewidth=0.1,label="Lasso realization")
axs[0].plot(bayes_theo_2[1],bayes_theo_2[0],linestyle='-',color='blue',linewidth=1,label="TPoP/CPoP")
axs[0].plot(lasso_theo_2[1],lasso_theo_2[0],color='red',linewidth=1,label="LASSO")

axs[0].set_title('$\delta=0.8$, $n=2000$, $d=2500$',fontsize=12)
axs[0].set_xticks(ticks=[0.0, 0.2, 0.4, 0.6])
axs[0].set_yticks(ticks=[0.0, 0.2, 0.4, 0.6,0.8,1])
axs[0].legend(fontsize=11, loc='lower right', framealpha=0.8, frameon=True)
axs[0].tick_params(axis='both', length=0, pad=5)

axs[1].grid(color='grey', linestyle='dotted', linewidth=0.1)
for k in range(10):
    axs[1].plot(bayes_data_1[0][k],bayes_data_1[1][k],color='blue',linewidth=0.1, alpha=0.3)
#    axs[1].plot(d1[2][k],d1[3][k],color='darkturquoise',linewidth=0.1)
    axs[1].plot(lasso_data_1[0][k*300:k*300+300],lasso_data_1[1][k*300:k*300+300],color='red',linewidth=0.1, alpha=0.3)
#axs[1].plot(d1[0][9],d1[1][9],color='blue',linewidth=0.1,label="TPoP realization")
#axs[1].plot(d2[0][2700:3000],d2[1][2700:3000],color='red',linewidth=0.1,label="Lasso realization")
axs[1].plot(bayes_theo_1[1],bayes_theo_1[0],linestyle='-',color='blue',linewidth=1,label="TPoP/CPoP")
axs[1].plot(lasso_theo_1[1],lasso_theo_1[0],color='red',linewidth=1,label="LASSO")

axs[1].set_title('$\delta=1.25$, $n=2500$, $d=2000$',fontsize=12)
axs[1].set_xticks(ticks=[0.0, 0.2, 0.4, 0.6])
axs[1].set_yticks(ticks=[0.0, 0.2, 0.4, 0.6,0.8,1])
axs[1].legend(fontsize=11, loc='lower right', framealpha=0.8, frameon=True)
axs[1].tick_params(axis='both', length=0, pad=5)


axs[0].set_xlabel("",fontsize=11)
axs[0].set_ylabel("",fontsize=11)
axs[1].set_xlabel("",fontsize=11)
axs[1].set_ylabel("",fontsize=11)
for ax in axs.flat:
    ax.set(xlabel='FDP', ylabel='TPP')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
    
    
plt.savefig('fig1.pdf')