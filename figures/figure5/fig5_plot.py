import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=np.load("fig5data.npy",allow_pickle=True)
formalism2_data=data[0]
formalism3_data=data[1]

formalism2_data_1=formalism2_data[0]
formalism2_data_2=formalism2_data[1]
formalism2_data_3=formalism2_data[2]

formalism3_data_1=formalism3_data[0]
formalism3_data_2=formalism3_data[1]
formalism3_data_3=formalism3_data[2]


formalism2_ns=[25,50,100,200]
formalism2_logns=np.log(np.array(formalism2_ns))

means=[np.mean(formalism2_data_1[i]) for i in range(4)]
stds=[np.std(formalism2_data_1[i]) for i in range(4)]
formalism2_logmeans1=np.log(np.array(means))
formalism2_logstds1=np.divide(np.array(stds),np.array(means))

means=[np.mean(formalism2_data_2[i]) for i in range(4)]
stds=[np.std(formalism2_data_2[i]) for i in range(4)]
formalism2_logmeans2=np.log(np.array(means))
formalism2_logstds2=np.divide(np.array(stds),np.array(means))

means=[np.mean(formalism2_data_3[i]) for i in range(4)]
stds=[np.std(formalism2_data_3[i]) for i in range(4)]
formalism2_logmeans3=np.log(np.array(means))
formalism2_logstds3=np.divide(np.array(stds),np.array(means))

formalism3_ns=[200,500,1000,2000]
formalism3_logns=np.log(np.array(formalism3_ns))

means=[np.mean(formalism3_data_1[i]) for i in range(4)]
stds=[np.std(formalism3_data_1[i]) for i in range(4)]
formalism3_logmeans1=np.log(np.array(means))
formalism3_logstds1=np.divide(np.array(stds),np.array(means))

means=[np.mean(formalism3_data_2[i]) for i in range(4)]
stds=[np.std(formalism3_data_2[i]) for i in range(4)]
formalism3_logmeans2=np.log(np.array(means))
formalism3_logstds2=np.divide(np.array(stds),np.array(means))

means=[np.mean(formalism3_data_3[i]) for i in range(4)]
stds=[np.std(formalism3_data_3[i]) for i in range(4)]
formalism3_logmeans3=np.log(np.array(means))
formalism3_logstds3=np.divide(np.array(stds),np.array(means))

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams['text.usetex'] = True

fig, axs = plt.subplots(1, 2, figsize=(9,4))
fig.tight_layout(pad=5.0)
axs[0].tick_params(axis='both', labelsize=14)
axs[1].tick_params(axis='both', labelsize=14)


axs[0].grid(color='grey', linestyle='dotted', linewidth=0.1)

axs[0].errorbar(x=formalism2_logns,y=formalism2_logmeans1,yerr=formalism2_logstds1,color='darkturquoise',capsize=5,label='$\delta=0.8$')
axs[0].errorbar(x=formalism2_logns,y=formalism2_logmeans2,yerr=formalism2_logstds2,color='blue',capsize=5,label='$\delta=1.25$')
axs[0].errorbar(x=formalism2_logns,y=formalism2_logmeans3,yerr=formalism2_logstds3,color='red',capsize=5,label='$\delta=2$')


axs[0].set_title('p-values from PoPCe',fontsize=15)
axs[0].set_xticks(ticks=[3,4,5])
#axs[0].set_yticks(ticks=[-5,-4.5,-4,-3.5,-3])
#axs[0].legend(fontsize=9, loc='upper right', framealpha=0.8, frameon=True)
axs[0].tick_params(axis='both', length=0, pad=5)

for spine in ['top', 'right','left','bottom']:
    axs[0].spines[spine].set_linewidth(1.5)


axs[1].grid(color='grey', linestyle='dotted', linewidth=0.1)

axs[1].errorbar(x=formalism3_logns,y=formalism3_logmeans1,yerr=formalism3_logstds1,color='darkturquoise',capsize=5,label='$\delta=0.8$')
axs[1].errorbar(x=formalism3_logns,y=formalism3_logmeans2,yerr=formalism3_logstds2,color='blue',capsize=5,label='$\delta=1.25$')
axs[1].errorbar(x=formalism3_logns,y=formalism3_logmeans3,yerr=formalism3_logstds3,color='red',capsize=5,label='$\delta=2$')

axs[1].set_title('p-values from PoEdCe',fontsize=15)
axs[1].set_xticks(ticks=[5,6,7,8])
#axs[1].set_yticks(ticks=[-5,-4.5,-4,-3.5,-3])
axs[1].legend(fontsize=12, loc='upper right', framealpha=0.8, frameon=True)
axs[1].tick_params(axis='both', length=0, pad=5)

for spine in ['top', 'right','left','bottom']:
    axs[1].spines[spine].set_linewidth(1.5)


axs[0].set_xlabel(r'$\log n$',fontsize=15)
axs[0].set_ylabel(r'$\log$ Wasserstein distance',fontsize=15)
axs[1].set_xlabel(r'$\log n$',fontsize=15)
axs[1].set_ylabel("",fontsize=9)

#for ax in axs.flat:
#    ax.set(xlabel=r'$\log n$', ylabel=r'$\log$ Wasserstein distance')

# Hide x labels and tick labels for top plots and y ticks for right plots.
#for ax in axs.flat:
#    ax.label_outer()
    
plt.savefig('fig5.pdf')