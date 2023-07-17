import numpy as np
import scipy
from scipy.stats import norm
from scipy.integrate import quadrature
import datetime
import random
from sklearn import linear_model
from scipy.optimize import fsolve
import cvxpy as cp


def self_consistent_tau_star(delta, sigma, points, weights, error=0.01, it=20):
    '''
    Compute tau_star, solution of the self consistent equation for bayes linear model Y=beta_0 + tau*Z
    where beta_0 is scalar with three delta prior and Z is standard gaussian. 
    tau solves tau^2=sigma^2 + MSE(beta_tau,beta_0)/delta, where beta_tau is posterior mean of beta_0 given Y.  
    
    Outputs : btau
        solution of self consistent equation. 
    Variables :
        delta : signal to noise ratio n/d in original model
        sigma : noise level
        points: prior distribution location
        weights: prior distribution weight
        error : iteration stops when new update of tau differ less than error from the previous tau in iteration.
        it : number of iterations
    '''
    x1 = points[0]
    x2 = points[1]
    x3 = points[2]
    p1 = weights[0]
    p2 = weights[1]
    p3 = weights[2]

    def mmse(z, tau):
        b1 = (x1 * p1 * norm.pdf(z + (x1 - x1) / tau) + x2 * p2 * norm.pdf(z + (x1 - x2) / tau) + x3 * p3 * norm.pdf(
            z + (x1 - x3) / tau)) / (
                         p1 * norm.pdf(z + (x1 - x1) / tau) + p2 * norm.pdf(z + (x1 - x2) / tau) + p3 * norm.pdf(
                     z + (x1 - x3) / tau))
        b2 = (x1 * p1 * norm.pdf(z + (x2 - x1) / tau) + x2 * p2 * norm.pdf(z + (x2 - x2) / tau) + x3 * p3 * norm.pdf(
            z + (x2 - x3) / tau)) / (
                         p1 * norm.pdf(z + (x2 - x1) / tau) + p2 * norm.pdf(z + (x2 - x2) / tau) + p3 * norm.pdf(
                     z + (x2 - x3) / tau))
        b3 = (x1 * p1 * norm.pdf(z + (x3 - x1) / tau) + x2 * p2 * norm.pdf(z + (x3 - x2) / tau) + x3 * p3 * norm.pdf(
            z + (x3 - x3) / tau)) / (
                         p1 * norm.pdf(z + (x3 - x1) / tau) + p2 * norm.pdf(z + (x3 - x2) / tau) + p3 * norm.pdf(
                     z + (x3 - x3) / tau))
        return (p1 * ((b1 - x1) ** 2) + p2 * ((b2 - x2) ** 2) + p3 * ((b3 - x3) ** 2)) * norm.pdf(z)

    #btau = abs(points[0]) + abs(points[1]) + abs(points[2])
    btau=5
    for i in range(it):
        btauold = btau
        btau = (quadrature(mmse, -3, 3, args=(btauold))[0] / delta + sigma ** 2) ** 0.5

    if abs(btau - btauold) / btau > error:
        print('tau* not converged')
    if delta * (btau ** 2 - sigma ** 2) < error:
        print('tau* not solved')
    return btau
    

def univariate_posteriors(h, tau, points, weights):
    '''
    compute one dimensional posterior mean and posterior zero probability under univariate model y=beta_0 + tau*Z, 
    beta_0 is a scalar with three delta prior, Z is standard gaussian. 

    Output : [postmean, postzero]
        postmean = E(beta|beta+tau*Z=h), 
        postzero = P(beta=0|beta+tau*Z=h)
    Variables :
        h: observation of y
        tau : model parameter tau (y=beta_0+tau*Z)
        points: prior distribution location
        weights: prior distribution weight
    '''
    newweights = np.multiply(norm.pdf((h * np.ones(len(points)) - points) / tau), weights)
    pm = np.sum(np.multiply(newweights, points)) / np.sum(newweights)
    pz = newweights[0] / np.sum(newweights)
    return np.array([pm, pz])


def bayes_optimal_tpp_fdp(btau, points, weights):
    '''
    compute bayes optimal tpp,fdp curve, under model y=Ax + eps, x are iid some prior,
    and eps Gaussian noise with variance sigma^2.
    
    Output : [tpps,fdps]
        tpps : theoretical/asymptotic tpps for thresholding posterior zero probability with threshold t, t =0.01*i from i=1 to 100. 
        fdps : theoretical/asymptotic fdps for thresholding posterior zero probability with threshold t, t =0.01*i from i=1 to 100.
    
    Variables :
        btau : solution of self consistent equation in bayes linear model
        points: prior distribution location
        weights: prior distribution weight

    '''
    x1 = points[0]
    x2 = points[1]
    x3 = points[2]
    p1 = weights[0]
    p2 = weights[1]
    p3 = weights[2]

    def pp(t, x):
        if univariate_posteriors(0, btau, points, weights)[1] < t:
            return 1
        else:
            left = -5
            right = 5
            mid = -x / btau
            mid1 = mid
            mid2 = mid
            for i in range(10):
                ln = (mid1 + left) / 2
                rn = (mid2 + right) / 2
                if univariate_posteriors(ln * btau + x, btau, points, weights)[1] > t:
                    mid1 = ln
                else:
                    left = ln
                if univariate_posteriors(rn * btau + x, btau, points, weights)[1] > t:
                    mid2 = rn
                else:
                    right = rn
            return 1 + norm.cdf(ln) - norm.cdf(rn)

    fdps = []
    tpps = []
    p1s = []
    p2s = []
    p3s = []
    for i in range(100):
        t = (i + 1) / 100
        p1s += [pp(t, x1)]
        p2s += [pp(t, x2)]
        p3s += [pp(t, x3)]
    p1s = [0] + p1s
    p2s = [0] + p2s
    p3s = [0] + p3s
    for i in range(101):
        if p1s[i] == 0:
            fdps += [0]
        else:
            fdps += [p1 * p1s[i] / (p1 * p1s[i] + p2 * p2s[i] + p3 * p3s[i])]
        tpps += [(p2 * p2s[i] + p3 * p3s[i]) / (p2 + p3)]
    return np.array([tpps, fdps])


def threshold_tpp_fdp(x, test, lower, upper, sgn, d):
    '''
    return fdp-tpp given true x in R^d and corresponding test statistic to threshold in R^d
    
    Output : [tpps,fdps]
        tpps : tpps when we threshold test statistic 
        fdps : fdps when we threshold test statistic 
    
    Variables :
        x : true values x in R^d
        test : test statistics in R^d, that we threshold in multiple testing
        lower : lower bound of the thresholds 
        upper : upper bound of the thresholds
        sgn : decides the direction of threshold, sgn=1 means we reject when test >t, while sgn=-1 means we reject when test<t
    '''
    fdps = []
    tpps = []
    for j in range(300):
        t = (j + 1) * (upper - lower) / 300 + lower
        dis = 0
        pos = 0
        td = 0
        for i in range(d):
            if sgn * test[i] > sgn * t:
                dis += 1
                if x[i] != 0:
                    td += 1
            if x[i] != 0:
                pos += 1
        fd = dis - td
        if dis == 0:
            fdp = 0
        else:
            fdp = fd / dis
        tpp = td / pos
        fdps += [fdp]
        tpps += [tpp]
    if sgn == 1:
        fdps += [0]
        tpps += [0]
    else:
        fdps = [0] + fdps
        tpps = [0] + tpps
    return np.array([tpps, fdps])


def MCMC(sigma, points, weights, A, y, it=5000):
    '''
    Calculate the posterior mean and posterior zero probability of x given
    (y, A) under the statistical model y = Ax + eps, x are iid with three delta prior,
    and eps Gaussian noise with variance sigma^2.
    
    Outputs: [postmean, postzero]
        postmean: posterior mean of x given (y, A)
        postzero: posterior probability of x = 0 given (y, A)
    Variables:
        sigma: noise level
        points: prior distribution location
        weights: prior distribution weight
        A: matrix of size n times d
        y: respons of size d times 1
        it: number of MCMC iterates
    '''
    n = len(A)
    d = len(A[0])
    x1 = points[0]
    x2 = points[1]
    x3 = points[2]
    p1 = weights[0]
    p2 = weights[1]
    p3 = weights[2]

    update = np.zeros(d)
    X = [np.zeros(d)]
    us = np.random.uniform(0, 1, it * d)
    norms = []
    for i in range(d):
        norms += [np.linalg.norm(A[:, i]) ** 2]
    z = y - A @ update + update[0] * A[:, 0]
    newx = 0
    for k in range(it * d):
        i = k % d
        z = z + update[i] * A[:, i] - newx * A[:, (i - 1) % d]
        dotprod = 2 * np.dot(A[:, i], z)
        w1 = p1 * np.exp(-(norms[i] * x1 ** 2 - dotprod * x1) / (2 * sigma ** 2))
        w2 = p2 * np.exp(-(norms[i] * x2 ** 2 - dotprod * x2) / (2 * sigma ** 2))
        w3 = p3 * np.exp(-(norms[i] * x3 ** 2 - dotprod * x3) / (2 * sigma ** 2))
        u = us[k]
        if u < w1 / (w1 + w2 + w3):
            newx = x1
        elif w1 / (w1 + w2 + w3) < u < (w1 + w2) / (w1 + w2 + w3):
            newx = x2
        else:
            newx = x3
        update[i] = newx
        newxx = np.array(update)
        if i == 0:
            X += [newxx]
    X = np.array(X)
    postmean = [np.mean(X[:, i]) for i in range(d)]
    postzero = [np.count_nonzero(X[:, i] == 0) / it for i in range(d)]
    return (np.array([postmean, postzero]))



def posterior_mean_amp(sigma,btau,points,weights,A,y,it=100,eps=0.0001):
    '''
    Calculate the posterior mean of x via AMP algorithm 
    given (y, A) under the statistical model y = Ax + eps, x are iid with three delta prior,
    and eps Gaussian noise with variance sigma^2.
    
    Output: posterior mean of x given (y, A)
    Variables:
        sigma: noise level
        btau: initialization of zeta in amp algorithm
        points: prior distribution location
        weights: prior distribution weight
        A: matrix of size n times d
        y: respons of size d times 1
        it: number of AMP iterates
        eps: we finish AMP algorithm when distance between update and old posterior mean is smaller than eps
    '''
    n=len(A)
    d=len(A[0])
    x1=points[0]
    x2=points[1]
    x3=points[2]
    p1=weights[0]
    p2=weights[1]
    p3=weights[2]

    '''
    following are functions frequently used in AMP algorithms 
    '''
    def pr(x,theta,zeta):
        return np.exp(-(theta-x)**2/(2*zeta**2))
    def dpr(x,theta,zeta):
        return (x-theta)*np.exp(-(theta-x)**2/(2*zeta**2))/(zeta**2)
    def etab(theta,zeta):
        a1=pr(x1,theta,zeta)*p1
        a2=pr(x2,theta,zeta)*p2
        a3=pr(x3,theta,zeta)*p3
        return (a1*x1+a2*x2+a3*x3)/(a1+a2+a3)
    def etapb(theta,zeta):
        a1=pr(x1,theta,zeta)*p1
        a2=pr(x2,theta,zeta)*p2
        a3=pr(x3,theta,zeta)*p3
        aa1=dpr(x1,theta,zeta)*p1
        aa2=dpr(x2,theta,zeta)*p2
        aa3=dpr(x3,theta,zeta)*p3
        return ((aa1*x1+aa2*x2+aa3*x3)*(a1+a2+a3)-(a1*x1+a2*x2+a3*x3)*(aa1+aa2+aa3))/((a1+a2+a3)**2)
    def varr(theta,zeta):
        a1=pr(x1,theta,zeta)*p1
        a2=pr(x2,theta,zeta)*p2
        a3=pr(x3,theta,zeta)*p3
        return (a1*x1**2+a2*x2**2+a3*x3**2)/(a1+a2+a3)-((a1*x1+a2*x2+a3*x3)/(a1+a2+a3))**2
    m0=np.random.uniform(-0.5,0.5,d)
    z0=np.random.uniform(-0.5,0.5,n)
    zeta0=btau
    ms=[m0]
    zs=[z0]
    zetas=[zeta0]
    index=0
    for k in range(it):
        mold=ms[k]
        zold=zs[k]
        zetaold=zetas[k]
        theta=mold+A.T@zold
        mnew=etab(theta,zetaold)
        znew=y-A@mnew+zold*(np.sum(etapb(theta,zetaold))/n)
        zetanew=sigma**2+np.sum(varr(theta,zetaold))/n
        ms+=[mnew]
        zs+=[znew]
        zetas+=[btau]
        index+=1
        if np.linalg.norm(mnew-mold)**2/d<eps:
            break
    return ms[index]

def CRT(points,weights,sigma,btau,A,y,it=100):
    '''
    Calculate the pvalues using conditional randomization test
    given (y, A) under the statistical model y = Ax + eps, x are iid with three delta prior,
    and eps Gaussian noise with variance sigma^2.
    
    Output: 
        pvals: pvalues of each coordinate, vector of size d times 1 
    Variables:
        points: prior distribution location
        weights: prior distribution weight
        sigma: noise level
        btau: self consistent tau star in bayesian linear model
        A: matrix of size n times d
        y: respons of size n times 1
        it: number of conditional randomizations to calculate pvalues
    '''
    n=len(A)
    d=len(A[0])
    pvals=[]
    #now=datetime.datetime.now()
    for j in range(d):
        p0=abs(posterior_mean_amp(sigma,btau,points,weights,A,y,20)[j])
        X=np.random.normal(0,(1/n)**0.5,size=(it,n))
        pval=0
        Am=np.delete(A,j,axis=1)
        for k in range(it):
            Anew=np.append(Am,np.array([X[k]]).T,axis=1)
            if abs(posterior_mean_amp(sigma,btau,points,weights,Anew,y,20)[d-1])>p0:
                pval+=1/(1+it)
        pvals+=[pval]
    #print(datetime.datetime.now()-now)
    return pvals


def dCRT(points,weights,sigma,btau,A,y):
    '''
    Calculate the pvalues using distilled conditional randomization test
    given (y, A) under the statistical model y = Ax + eps, x are iid with three delta prior,
    and eps Gaussian noise with variance sigma^2.
    
    Output: 
        pvals: pvalues of each coordinate, vector of size d times 1 
    Variables:
        points: prior distribution location
        weights: prior distribution weight
        sigma: noise level
        btau: self consistent tau star in bayesian linear model
        A: matrix of size n times d
        y: respons of size n times 1
    '''
    n=len(A)
    d=len(A[0])
    pvals=[]
    for i in range(d):
        a0=A[:,i]
        Am=np.delete(A,i,axis=1)
        xm=posterior_mean_amp(sigma,btau,points,weights,Am,y,it=100,eps=0.0001)
        z=y-Am@xm
        S0=abs(btau**2*np.dot(z,a0)/sigma**2)
        pval=2*norm.cdf(-((n**0.5)*S0*sigma**2)/(np.linalg.norm(z)*(btau**2)))
        pvals+=[pval]
    return pvals



def psi(p,btau,points,weights):
    '''
    Calculate the cumulant distribution function of P(tau*Z) where tau is self consistent tau star in bayesian linear model, Z is standard gaussian random variable, 
    and P is one dimensional posterior probability P(y):=Pr(b=0|b+tau*Z=y)
    
    Output: 
        psi(p): cumulant distribution function of P(tau*Z), i.e., Pr(P(tau*Z) <= p)
    Variables:
        p: input variable for psi(p)
        btau: self consistent tau star in bayesian linear model
        points: prior distribution location
        weights: prior distribution weight
    '''
    x=-4
    while (univariate_posteriors(x,btau,points,weights)[1]-p)*(univariate_posteriors(x+0.01,btau,points,weights)[1]-p)>0:
        x+=0.005    
    left=x
    x+=0.01
    while (univariate_posteriors(x,btau,points,weights)[1]-p)*(univariate_posteriors(x+0.01,btau,points,weights)[1]-p)>0:
        x+=0.005
    right=x
    return 1-(norm.cdf(right/btau)-norm.cdf(left/btau))


def p_to_e_calibration_threshold(alpha,epsilon,btau,points,weights,optimals):
    '''
    Calculate p-to-e calibration threshold, it calculates threshold t, where TPoP with threshold t has limit FDP to be alpha-epsilon

    Output: 
        t: scalar value t, FDP(TPoP(.;t,Pi)=alpha-epsilon
    Variables:
        alpha: target FDP level of eBH
        epsilon: small scalar to keep FDR converges to slightly less than alpha
        btau: self consistent tau star in bayesian linear model
        points: prior distribution location
        weights: prior distribution weight
        optimals: bayes optimal tpp and fdps in bayesian linear model
    '''
    l=0
    r=99
    for k in range(6):
        m=(l+r)//2
        if optimals[1][m]<alpha-epsilon:
            l=m
        else:
            r=m
    s=(m-1)/100
    return psi(s,btau,points,weights)



def calculate_tpp_fdp(x,rejection):
    '''
    Calculate tpp and fdp of a given x and result of a test.

    Output: [tpp,fdp]
        tpp: number of true positive divided by number of nonzero x_i
        fdp: number of false positive divided by number of rejection
    Variables:
        x: vector of size d times 1, 0's are nulls and nonzeros are alternatives
        rejection: vector of size d times 1, result of a test, 0's are not rejecting, 1's are rejections.
    '''
    d=len(x)
    dis=0
    pos=0
    td=0
    for i in range(d):
        if rejection[i]==1:
            dis+=1
            if x[i]!=0:
                td+=1
        if x[i]!=0:
            pos+=1
    fd=dis-td
    if dis==0:
        fdp=0
    else:
        fdp=fd/dis
    tpp=td/pos
    return np.array([tpp,fdp])

def eBH(pvals,t,p0,alpha):
    '''
    Perform eBH with p-to-e calibrator 1{p<=t}/t and null ratio p0. After p-to-e calibration of given pvalues, 
    we reject k largest e-values where k is the largest number that p0*d/k*e_(k) <= alpha, where d is number of pvalues, and e_(k) is kth largest evalue. 
    
    Output: rejection vector of 0 and 1s where 1 means rejection.
    Variables: 
        pvals : size d times 1, input vector of p values
        t : threshold for p-to-e calibrator
        p0 : given null proportion for eBH
        alpha : desired FDR control level for eBH
    '''
    d=len(pvals)
    rej=1*(pvals<t*np.ones(d))
    if np.sum(rej)<p0*d*t/alpha:
        return np.zeros(d)
    else:
        return rej
    
def empirical_bayes(A,y,points,sigma,delta,lm):
    """
    Calculate estimate of a prior using empirical Bayes. We assume three delta prior of x on 0,1,-1 for Bayesian linear model y=Ax+eps.
    Given observation A and y, we use ridge estimate with ridge parameter lm, and solve prior to maximize the likelihood of ridge estimator. 


    Output : prior
        prior : vector of size 3 times 1, prior of x on 0,1 and -1. 
    Variables : 
        A: matrix of size n times d
        y: respons of size n times 1
        points: prior distribution location [0,1,-1] usually.
        sigma: noise level
        delta: signal to noise ratio n/d
        lm : ridge parameter, where beta_ridge = argmin 1/2||y-Ax||_2^2 + lm *||x||_2^2
    """
    n=len(A)
    d=len(A[0])
    delta=n/d
    a=2*delta
    b=delta*(1-2*lm)-1
    c=-delta*lm
    at=(-b+(b**2-4*a*c)**0.5)/(2*a)
    tau2=(delta*sigma**2+0.4*4*delta**2*(at-lm)**2)/(delta-(1/(1+2*at))**2)
    tau=tau2**0.5
    reg = linear_model.Ridge(alpha=2*lm)
    reg.fit(A,y)
    xhat=reg.coef_
    z=xhat*(1+2*at)
    p=cp.Variable(3)
    target=0
    for i in range(d):
        a=norm.pdf((z[i]*np.ones(3)-points)/tau)
        target+=cp.log(a.T@p)
    prob = cp.Problem(cp.Maximize(target),[cp.sum(p)==1,p>=np.zeros(3)])
    prob.solve()
    prior=p.value
    for i in range(3):
        if prior[i]<0:
            prior[i]=0
    return prior   


def PoPCe(A,y,x,sigma,points,weights,btau,optimals):
    '''
    Calculate FDP and TPP of PoPCe procedure for bayes linear model y=Ax+w, given prior of x. 
    We run PoPCe for given A,y for 60 alphas (from 0.01 to 0.6), and epsilon=0.1*alpha, and for each 60 tests, it returns
    FDP and TPP of the test comparing with true x. 
    
    Output: [tpps,fdps]
        tpps: size 60 times 1 vector of TPPs for each PoPCe with different alphas. 
        fdps: size 60 times 1 vector of FDPs for each PoPCe with different alphas
    Variables:
        A: matrix of size n times d
        y: response of size n times 1
        x: vector of size d times 1
        sigma: noise level of w
        points: prior distribution location
        weights: prior distribution weight
        btau: self consistent tau star in bayesian linear model
        optimals: bayes optimal tpp and fdps in bayesian linear model
    '''
    pvalues=CRT(points,weights,sigma,btau,A,y,it=100)
    alphas=[]
    rejections=[]
    fdps=[]
    tpps=[]
    for i in range(60):
        alpha=0.01+i/100
        alphas+=[alpha]
        t=p_to_e_calibration_threshold(alpha,alpha*0.1,btau,points,weights,optimals)
        rejection=eBH(pvalues,t,weights[0],alpha)
        rejections+=[rejection]
        tpps+=[calculate_tpp_fdp(x,rejection)[0]]
        fdps+=[calculate_tpp_fdp(x,rejection)[1]]
    return [tpps,fdps]

def PoEdCe(A,y,x,sigma,points,weights,btau,optimals,numalpha):
    '''
    Calculate FDP and TPP of PoEdCe procedure for bayes linear model y=Ax+w, given prior of x. 
    We run PoEdCe for given A,y for alphas (from 0.01 to 0.01*numalpha), and epsilon=0.1*alpha, and for each tests, it returns
    FDP and TPP of the test comparing with true x. 
    
    Output: [tpps,fdps]
        tpps: size numalpha times 1 vector of TPPs for each PoEdCe with different alphas. 
        fdps: size numalpha times 1 vector of FDPs for each PoEdCe with different alphas
    Variables:
        A: matrix of size n times d
        y: response of size n times 1
        x: vector of size d times 1
        sigma: noise level of w
        points: prior distribution location
        weights: prior distribution weight
        btau: self consistent tau star in bayesian linear model
        optimals: bayes optimal tpp and fdps in bayesian linear model
    '''
    pvalues=dCRT(points,weights,sigma,btau,A,y)
    alphas=[]
    rejections=[]
    fdps=[]
    tpps=[]
    for i in range(numalpha):
        alpha=0.01+i/100
        alphas+=[alpha]
        t=p_to_e_calibration_threshold(alpha,alpha*0.1,btau,points,weights,optimals)
        rejection=eBH(pvalues,t,weights[0],alpha)
        rejections+=[rejection]
        tpps+=[calculate_tpp_fdp(x,rejection)[0]]
        fdps+=[calculate_tpp_fdp(x,rejection)[1]]
    return [tpps,fdps]
    
def EPoEdCe(A,y,x,sigma,points,weights,btau,optimals,M,lm=1):
    '''
    Calculate FDP and TPP of PoPCe procedure for bayes linear model y=Ax+w, given prior of x. 
    We run PoPCe for given A,y for 60 alphas (from 0.01 to 0.6), and epsilon=0.1*alpha, and for each 60 tests, it returns
    FDP and TPP of the test comparing with true x. 
    
    Output: [tpps,fdps]
        tpps: size 60 times 1 vector of TPPs for each PoPCe with different alphas. 
        fdps: size 60 times 1 vector of FDPs for each PoPCe with different alphas
    Variables:
        A: matrix of size n times d
        y: response of size n times 1
        x: vector of size d times 1
        sigma: noise level of w
        points: prior distribution location
        weights: prior distribution weight
        btau: self consistent tau star in bayesian linear model
        optimals: bayes optimal tpp and fdps in bayesian linear model
        M: number of batches for empirical bayes
        lm: ridge regression parameter for empirical bayes
    '''
    n=len(A)
    d=len(A[0])
    delta=n/d
    delta2=n*M/(d*(M-1))
    priors=[]
    btaus=[]
    tss=[]
    for m in range(M):
        Am=np.delete(A, np.s_[m*(d//M):(m+1)*(d//M)], axis=1) 
        prior=empirical_bayes(Am,y,(M/(M-1))**0.5*points,sigma,delta2,lm)
        btau=self_consistent_tau_star(delta2,sigma,points,prior,error=0.01)
        ts=[p_to_e_calibration_threshold(i/100,i/1000,btau,points,prior,optimals) for i in range(60)]
        priors+=[prior]
        btaus+=[btau]
        tss+=[ts]
    pvals=[]
    rej=0
    rejection=[]
    for j in range(d):
        m=j//(d//M)
        prior=priors[m]
        btau=btaus[m]
        a0=A[:,j]
        Am=np.delete(A,j,axis=1)
        xm=posterior_mean_amp(sigma,btau,points,prior,Am,y,it=100)
        z=y-Am@xm
        S0=univariate_posteriors(btau**2*np.dot(z,a0)/sigma**2,btau,points,prior)[1]
        pval=0
        a=np.random.normal(0,np.linalg.norm(z)*btau**2*(1/n**0.5)/sigma**2,size=1000)
        for k in range(1000):
            if univariate_posteriors(a[k],btau,points,prior)[1]<S0:
                pval+=1/1001
        pvals+=[pval]
        rej+=np.array([1*(pval<tss[m][ind]) for ind in range(60)])
        rejection+=[[1*(pval<tss[m][ind]) for ind in range(60)]]
    rejection=np.array(rejection)    
    fdps=[]
    tpps=[]

    for ind in range(60):
        if rej[ind]==d:
            l=np.argsort(np.array(pvals))[d-1]//(d//M)
        else:
            l=np.argsort(np.array(pvals))[rej[ind]]//(d//M)
        t=tss[l][ind]
        if rej[ind]<weights[0]*d*t/((ind+1)*0.01):
            test=np.zeros(d)
        else:
            test=rejection[:,ind]
        tpps+=[calculate_tpp_fdp(x,test)[0]]
        fdps+=[calculate_tpp_fdp(x,test)[1]]
    return [tpps,fdps]


def posterior_mean_to_prob(m,btau,points,weights):
    '''
    Calculate univariate posterior probability P(b=0|b+btau*Z=h) where h satisfies m=E(b|b+tau*Z=h) 
    
    Output: posterior probability P(b=0|b+btau*Z=h)
    
    Variables:
        m: posterior mean E(b|b+tau*Z=h)
        btau: self consistent tau star in bayesian linear model
        points: prior distribution location
        weights: prior distribution weight
    '''
    x=-2
    while (univariate_posteriors(x,btau,points,weights)[0]-m)*(univariate_posteriors(x+0.01,btau,points,weights)[0]-m)>0:
        x+=0.005
    return univariate_posteriors(x,btau,points,weights)[1]

def formalism1(n,d,sigma,points,weights):
    '''
    Calculate 10 wasserstein distance between multivariate posterior probability/mean P(xi=0|Ax+w=y)/E(xi|Ax+w=y) and 
    univariate posterior probability/mean P(b=0|b+btau*Z=h_i)/E(b|b+tau*Z=h_i) 
    where y=Ax+w is from usual Bayesian linear model, and h_i=x_i+tau*Z_i where Z_i are iid standard normal random variables
    
    Output: [wass_mean,wass_prob]
        wass_mean: wasserstein distances between univariate/multivariate posterior means 10 generated data of A,y,h, size 10 times 1
        wass_prob: wasserstein distances between univariate/multivariate posterior probabilitiess 10 generated data of A,y,h, size 10 times 1
    
    Variables:
        n: length of response variable y
        d: length of features x
        sigma: noise level of w
        points: prior distribution location
        weights: prior distribution weight
    '''
    delta=n/d
    btau=self_consistent_tau_star(delta,sigma,points,weights,0.01)
    optimals=bayes_optimal_tpp_fdp(btau,points,weights)
    wass_mean=[]
    wass_prob=[]
    
    for i in range(10):
        now=datetime.datetime.now()
        A=np.random.normal(0,(1/n)**0.5,size=(n,d))
        x=np.random.choice(points,d,p=weights)
        w=np.random.normal(0,sigma,size=n)
        y=A@x+w
        multi_posterior_mean=posterior_mean_amp(sigma,btau,points,weights,A,y,it=100,eps=0.0001)
        multi_posterior_prob=np.array([posterior_mean_to_prob(j,btau,points,weights) for j in multi_posterior_mean])
        z=np.random.normal(0,1,size=d)
        b=np.random.choice(points,d,p=weights)
        h=b+btau*z
        univ_posteriors=np.array([univariate_posteriors(j,btau,points,weights) for j in h])
        wass_mean+=[scipy.stats.wasserstein_distance(multi_posterior_mean,univ_posteriors[:,0])]
        wass_prob+=[scipy.stats.wasserstein_distance(multi_posterior_prob,univ_posteriors[:,1])]
        print(datetime.datetime.now()-now)
        print('formalism1 simulation '+str(i)+' out of 10')
    return np.array([wass_mean,wass_prob])



def formalism2(n,d,sigma,points,weights,btau,psis):
    '''
    Calculate 10 wasserstein distance between pvalues from CRT and 
    iid samples from Psi(P(b0+tau*Z)). Z is standard normal, b0 is from same prior as our regression coefficients.
    For detailed explanation of function Psi and P, see appendix F of the paper. 
    
    
    Output: distances
        distances: wasserstein distances between pvalues and Psi(P(b0+tau*Z))s from 10 generated data of A,y,b0,Z, size 10 times 1
    
    Variables:
        n: length of response variable y
        d: length of features x
        sigma: noise level of w
        points: prior distribution location
        weights: prior distribution weight
        btau: self consistent tau star in bayesian linear model
        psis: pre-calculated values of psi(P(h))s to avoid repetitive calculation. 
    '''
    
    distances=[]
    for i in range(10):
        now=datetime.datetime.now()
        A=np.random.normal(0,(1/n)**0.5,size=(n,d))
        x=np.random.choice(points,d,p=weights)
        w=np.random.normal(0,sigma,size=n)
        y=A@x+w
        pvalues=CRT(points,weights,sigma,btau,A,y,it=100)
        print(datetime.datetime.now()-now,i+1)
        z0=np.random.normal(0,1,size=d)
        b0=np.random.choice(points,d,p=weights)
        c=b0+btau*z0
        index=[int((j+4)*50) for j in c]
        rhs=[psis[j] for j in index0]
        distances+=[scipy.stats.wasserstein_distance(pvalues,rhs)]
    return distances


def formalism3(n,d,sigma,points,weights,btau,psis):
    '''
    Calculate 10 wasserstein distance between pvalues from CRT and 
    iid samples from Psi(P(b0+tau*Z)). Z is standard normal, b0 is from same prior as our regression coefficients.
    For detailed explanation of function Psi and P, see appendix F of the paper. 
    
    
    Output: distances
        distances: wasserstein distances between pvalues and Psi(P(b0+tau*Z))s from 10 generated data of A,y,b0,Z, size 10 times 1
    
    Variables:
        n: length of response variable y
        d: length of features x
        sigma: noise level of w
        points: prior distribution location
        weights: prior distribution weight
        btau: self consistent tau star in bayesian linear model
        psis: pre-calculated values of psi(P(h))s to avoid repetitive calculation. 
    '''

    distances=[]
    for i in range(50):
        now=datetime.datetime.now()
        A=np.random.normal(0,(1/n)**0.5,size=(n,d))
        x=np.random.choice(points,d,p=weights)
        w=np.random.normal(0,sigma,size=n)
        y=A@x+w
        pvalues=dCRT(points,weights,sigma,btau,A,y)
        print(datetime.datetime.now()-now,i+1)
        z0=np.random.normal(0,1,size=d)
        b0=np.random.choice(points,d,p=weights)
        c=b0+btau*z0
        index=[int((j+4)*50) for j in c]
        rhs=[psis[j] for j in index0]
        distances+=[scipy.stats.wasserstein_distance(pvalues,rhs)]
    return distances
