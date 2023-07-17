import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=np.load("fig4data.npy",allow_pickle=True)
d1=data[0]
d2=data[1]
d3=data[2]

ns=[250,500,1000,2000]
logns=np.log(np.array(ns))

means=[np.mean(d1[i][1]) for i in range(4)]
stds=[np.std(d1[i][1]) for i in range(4)]
plogmeans1=np.log(np.array(means))
plogstds1=np.divide(np.array(stds),np.array(means))

means=[np.mean(d2[i][1]) for i in range(4)]
stds=[np.std(d2[i][1]) for i in range(4)]
plogmeans2=np.log(np.array(means))
plogstds2=np.divide(np.array(stds),np.array(means))

means=[np.mean(d3[i][1]) for i in range(4)]
stds=[np.std(d3[i][1]) for i in range(4)]
plogmeans3=np.log(np.array(means))
plogstds3=np.divide(np.array(stds),np.array(means))


means=[np.mean(d1[i][0]) for i in range(4)]
stds=[np.std(d1[i][0]) for i in range(4)]
elogmeans1=np.log(np.array(means))
elogstds1=np.divide(np.array(stds),np.array(means))


means=[np.mean(d2[i][0]) for i in range(4)]
stds=[np.std(d2[i][0]) for i in range(4)]
elogmeans2=np.log(np.array(means))
elogstds2=np.divide(np.array(stds),np.array(means))

means=[np.mean(d3[i][0]) for i in range(4)]
stds=[np.std(d3[i][0]) for i in range(4)]
elogmeans3=np.log(np.array(means))
elogstds3=np.divide(np.array(stds),np.array(means))

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams['text.usetex'] = True

fig, axs = plt.subplots(1, 2, figsize=(9,4))
fig.tight_layout(pad=5.0)
axs[0].tick_params(axis='both', labelsize=14)
axs[1].tick_params(axis='both', labelsize=14)


axs[0].grid(color='grey', linestyle='dotted', linewidth=0.1)

axs[0].errorbar(x=logns,y=plogmeans1,yerr=plogstds1,color='darkturquoise',capsize=5,label='$\delta=0.8$')
axs[0].errorbar(x=logns,y=plogmeans2,yerr=plogstds2,color='blue',capsize=5,label='$\delta=1.25$')
axs[0].errorbar(x=logns,y=plogmeans3,yerr=plogstds3,color='red',capsize=5,label='$\delta=2$')


axs[0].set_title('Posterior probability',fontsize=15)
axs[0].set_xticks(ticks=[5,6,7,8])
axs[0].set_yticks(ticks=[-5,-4,-3,-2])
axs[0].set_ylim([-5.5,-2])
#axs[0].legend(fontsize=14, loc='upper right', framealpha=0.8, frameon=True)
axs[0].tick_params(axis='both', length=0, pad=5)
for spine in ['top', 'right','left','bottom']:
    axs[0].spines[spine].set_linewidth(1.5)


axs[1].grid(color='grey', linestyle='dotted', linewidth=0.1)

axs[1].errorbar(x=logns,y=elogmeans1,yerr=elogstds1,color='darkturquoise',capsize=5,label='$\delta=0.8$')
axs[1].errorbar(x=logns,y=elogmeans2,yerr=elogstds2,color='blue',capsize=5,label='$\delta=1.25$')
axs[1].errorbar(x=logns,y=elogmeans3,yerr=elogstds3,color='red',capsize=5,label='$\delta=2$')

axs[1].set_title('Posterior expectation',fontsize=15)
axs[1].set_xticks(ticks=[5,6,7,8])
axs[1].set_yticks(ticks=[-5,-4,-3,-2])
axs[1].set_ylim([-5,-2])
axs[1].legend(fontsize=12, loc='upper right', framealpha=0.8, frameon=True)
axs[1].tick_params(axis='both', length=0, pad=5)
for spine in ['top', 'right','left','bottom']:
    axs[1].spines[spine].set_linewidth(1.5)
    

axs[0].set_xlabel(r'$\log n$',fontsize=15)
axs[0].set_ylabel(r'$\log$ Wasserstein distance',fontsize=15)
axs[1].set_xlabel(r'$\log n$',fontsize=15)
axs[1].set_ylabel("",fontsize=9)
    
    
plt.savefig('fig4.pdf')