# FDR_Bayes_figures
 
This repository provides the codes used to generate figures for the paper:

**Near-optimal multiple testing in Bayesian linear models with finite-sample FDR control**.  
Taejoo Ahn, Licong Lin, Song Mei.  
Paper: https://arxiv.org/abs/2211.02778.

![](fig1.png)
![](fig2.png)


## Testing procedures

Our primary goal of the paper is to suggest a multiple testing procedure that controls FDR from finite samples and achieves near-optimal power under
well-specified Bayesian linear models. We introduce 5 procedures in this paper, which are **TPoP, CPoP, PoPCe, PoEdCe, EPoEdCe**

**TPoP** is thresholding posterior probability (or local FDR), and most powerful under mFDR control.
**CPoP** is thresholding local fdr where the threshold is data dependent, and most powerful under BFDR control.

**PoPCe** and **PoEdCe** are procedures suggested to control finite sample FDR at the same time even under misspecified models, and achieve near-optimal power under well-specified Bayesian linear models. They are using local FDR as a base statistics to generate p-values using CRT and dCRT respectively, then apply eBH procedure to guarantee finite sample FDR control. **PoEdCe** is computationally lighter than **PoPCe** by using distilled CRT instead of CRT.

**EPoEdCe** is a procedure mixing empricial Bayes and **PoEdCe** to suggest a procedure we can use even when prior is not given or unknown. 

### Bayesian linear model

**PoPCe, PoEdCe, EPoEdCe** reaches optimal power only under well-specified Bayesian linear model. Under Bayesian linear model, we assume ${\bf Y}={\bf X\beta_0}+{\bf \epsilon}$ where ${\bf Y},{\bf \epsilon}\in\mathbb{R}^n$, ${\bf X}\in\mathbb{R}^{n\times d}$ and ${\bf \beta_0}\in\mathbb{R}^d$ where $\epsilon_i\sim N(0,\sigma^2)$, $X_{ij}\sim N(0,1/n)$ and $\beta_{0i}\sim\Pi$ independently, with some prior $\Pi$, either known or unknown.

## Examples (figure1, 2)

For example, figure1 compares **TPoP** and **CPoP** with thresholded LASSO procedure, with its theoretical asymptotic limit and realization of 10 instances of generated data. 

Figure 2 performs **PoPCe, PoEdCE** and **EPoEdCe** on 10 instances of generated data under Bayesian linear model, and plot there TPP and FDP against target FDR control $\alpha$. 

## How to run the code

On each folder, for example for figure1, run `fig1_data.py` to generate data file `fig1data.npy` first. Then run `fig1_plot.py` to generate the plot `fig1.pdf`.

## Miscellanous
  
More information about the procedures can be found in the original paper. 
If you use this code in your research, please cite our paper
```
@misc{ahn2023nearoptimal,
      title={Near-optimal multiple testing in Bayesian linear models with finite-sample FDR control}, 
      author={Taejoo Ahn and Licong Lin and Song Mei},
      year={2023},
      eprint={2211.02778},
      archivePrefix={arXiv},
      primaryClass={math.ST}
}
```

