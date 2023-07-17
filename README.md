# FDR_Bayes_figures
 
This repository provides the codes used to generate figures for the paper:

**Near-optimal multiple testing in Bayesian linear models with finite-sample FDR control**.  
Taejoo Ahn, Licong Lin, Song Mei.  
Paper: https://arxiv.org/abs/2211.02778.

![](fig1.png)
![](fig2.png)


## Testing procedures

Our primary goal of the paper is to suggest a multiple testing procedure that controls FDR from finite samples and achieves near-optimal power under
well-specified Bayesian linear models. We introduce 5 procedures in this paper, which are **TPoP, CPoP, PoPCe, PoEdCe, EPoEdCe**\\

**TPoP** is thresholding posterior probability (or local FDR), and most powerful under mFDR control.\\
**CPoP** is thresholding local fdr where the threshold is data dependent, and most powerful under BFDR control.\\

**PoPCe** and **PoEdCe** are procedures suggested to control finite sample FDR at the same time even under misspecified models, and achieve near-optimal power under well-specified Bayesian linear models. They are using local FDR as a base statistics to generate p-values using CRT and dCRT respectively, then apply eBH procedure to guarantee finite sample FDR control. **PoEdCe** is computationally lighter than **PoPCe** by using distilled CRT instead of CRT.

**EPoEdCe** is a procedure mixing empricial Bayes and **PoEdCe** to suggest a procedure we can use even when prior is not given or unknown. 

### Bayesian linear model
