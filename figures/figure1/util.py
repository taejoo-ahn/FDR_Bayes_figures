import numpy as np
from scipy.stats import norm
from scipy.integrate import quadrature
import datetime
import random
from sklearn import linear_model
from scipy.optimize import fsolve


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

    btau = abs(points[0]) + abs(points[1]) + abs(points[2])
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


def tpop_cpop_simulation(n, d, delta, points, weights, sigma=0.25):
    '''
    perform 10 realizations of tpop and cpop procedures:
        generate 10 data of y=Ax+eps, x are d*1 iid with three delta prior,
        and eps are n*1 Gaussian noise with variance sigma^2. A are n*d with iid gaussian with variance 1/n. 
        then run cpop and tpop on observed A and y, and compare with original x to return tpp and fdp. 

    Output : [tpopfdp, tpoptpp, cpopfdp, cpoptpp]
        tpopfdp : 10 lists of fdps of tpop procedure on each realization
        tpoptpp : 10 lists of tpps of tpop procedure on each realization
        cpopfdp : 10 lists of fdps of cpop procedure on each realization
        cpoptpp : 10 lists of tpps of cpop procedure on each realization

    Variable : 
        n: dimension of data y
        d: dimension of features x
        delta: signal to noise ratio n/d
        points: prior distribution location
        weights: prior distribution weight
        sigma: noise level

    '''

    tpopfdp = []
    tpoptpp = []
    print("start tpop simulation")
    for k in range(10):
        now=datetime.datetime.now()
        A = np.random.normal(0, (1 / n) ** 0.5, size=(n, d))
        x = np.random.choice(points, d, p=weights)
        w = np.random.normal(0, sigma, size=n)
        y = A @ x + w
        test = MCMC(sigma, points, weights, A, y, it=5000)[1]
        z = threshold_tpp_fdp(x, test, 0, 1, -1, d)
        tpopfdp += [z[1]]
        tpoptpp += [z[0]]
        print("out of 10 simulations, " + str(k+1) + " are done",datetime.datetime.now()-now)


    cpopfdp = []
    cpoptpp = []
    print("start cpop simulation")
    for k in range(10):
        now=datetime.datetime.now()
        A = np.random.normal(0, (1 / n) ** 0.5, size=(n, d))
        x = np.random.choice(points, d, p=weights)
        w = np.random.normal(0, sigma, size=n)
        y = A @ x + w
        test = MCMC(sigma, points, weights, A, y, it=5000)[1]
        z = threshold_tpp_fdp(x, test, 0, 1, -1, d)
        cpopfdp += [z[1]]
        cpoptpp += [z[0]]
        print("out of 10 simulations, " + str(k+1) + " are done",datetime.datetime.now()-now)

    return np.array([tpopfdp, tpoptpp, cpopfdp, cpoptpp])


def lasso_theoretical_tpp_fdp(delta,sigma, points, weights):
    '''
    compute theoretical tpp,fdp curve of thresholded lasso, under model y=Ax + eps, x are iid some prior,
    and eps Gaussian noise with variance sigma^2.
    
    Output : [tpps,fdps]
        tpps : theoretical tpps for thresholding lasso estimate with threshold t, t =0.01*i from i=1 to 100. 
        fdps : theoretical fdps for thresholding lasso estimate with threshold t, t =0.01*i from i=1 to 100.
    
    Variables :
        delta: signal to noise ratio n/d
        sigma: noise level
        points: prior distribution location
        weights: prior distribution weight

    '''

    # solve theoretically optimal lambda for Lasso
    p=weights[1]
    def eq(tau, alpha):
        sum1 = 2 * ((1 + alpha ** 2) * norm.cdf(-alpha) - alpha * norm.pdf(alpha))
        sum2 = (1 + alpha ** 2) * (1 - norm.cdf(alpha - 1 / tau) + norm.cdf(-alpha - 1 / tau)) - (
                    alpha + 1 / tau) * norm.pdf(alpha - 1 / tau) - (alpha - 1 / tau) * norm.pdf(-alpha - 1 / tau) + (
                           norm.cdf(alpha - 1 / tau) - norm.cdf(-alpha - 1 / tau)) / (tau ** 2)
        return ((1 - 2 * p) * sum1 + 2 * p * sum2) * (tau ** 2) / delta + sigma ** 2 - tau ** 2

    def lam(tau, alpha):
        return alpha * tau * (1 - (2 * (1 - 2 * p) * norm.cdf(-alpha) + 2 * p * (
                    1 - norm.cdf(alpha - 1 / tau) + norm.cdf(-alpha - 1 / tau))) / delta)

    def amineq(alpha):
        return (1 + alpha ** 2) * (norm.cdf(-alpha)) - alpha * norm.pdf(alpha) - delta / 2

    amin = fsolve(amineq, 0.5)[0]
    alphas = []
    taus = []
    lambdas = []
    risks = []
    for i in range(100):
        alpha = amin + (i + 1) * 0.05

        def taueq(x):
            return (eq(x, alpha))

        tau = fsolve(taueq, 0.5)[0]
        lamb = lam(tau, alpha)
        risk = delta * (tau ** 2 - sigma ** 2)
        alphas += [alpha]
        taus += [tau]
        lambdas += [lamb]
        risks += [risk]

    taus = np.delete(taus, 0)
    lambdas = np.delete(lambdas, 0)
    alphas = np.delete(alphas, 0)
    lambdastar = lambdas[np.argmin(taus)]
    taustar = taus[np.argmin(taus)]
    alphastar = alphas[np.argmin(taus)]

    # theoretical curve for Lasso
    def tpplc(t):
        return 1 - norm.cdf(alphastar + (t - 1) / taustar) + norm.cdf(-alphastar - (t + 1) / taustar)

    def fdplc(t):
        fd = 2 * (1 - 2 * p) * norm.cdf(-alphastar - t / taustar)
        return fd / (fd + 2 * p * tpplc(t))

    tpps = []
    fdps = []
    for i in range(100):
        t = i / 100
        tpps += [tpplc(t)]
        fdps += [fdplc(t)]
    tpps[99] = 0
    fdps[99] = 0
    tpps = [1] + tpps
    fdps = [1-2*p] + fdps

    return np.array([tpps,fdps,lambdastar])


def lasso_simulation(n, d, delta, points, weights, sigma, lambdastar):
    '''
    perform 10 realizations of thresholded lasso procedures:
        generate 10 data of y=Ax+eps, x are d*1 iid with three delta prior,
        and eps are n*1 Gaussian noise with variance sigma^2. A are n*d with iid gaussian with variance 1/n. 
        then run thresholded lasso on observed A and y, and compare with original x to return tpp and fdp. 

    Output : [rlfdp, rltpp]
        rlfdp : 10 lists of fdps of tpop procedure on each realization
        rltpp : 10 lists of tpps of tpop procedure on each realization

    Variable : 
        n: dimension of data y
        d: dimension of features x
        delta: signal to noise ratio n/d
        points: prior distribution location
        weights: prior distribution weight
        sigma: noise level
    '''
    

    # realization for Lasso
    rlfdp = []
    rltpp = []
    print("start LASSO simulation")
    for k in range(10):
        now=datetime.datetime.now()
        A = np.random.normal(0, (1 / n) ** 0.5, size=(n, d))
        x = np.random.choice(points, d, p=weights)
        w = np.random.normal(0, sigma, size=n)
        y = A @ x + w
        reg = linear_model.Lasso(alpha=lambdastar / n)
        reg.fit(A, y)
        for j in range(300):
            t = (j + 1) * 0.01
            dis = 0
            pos = 0
            td = 0
            for i in range(d):
                if abs(reg.coef_[i]) > t:
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
            rlfdp += [fdp]
            rltpp += [tpp]

        print("out of 10 simulations, " + str(k+1) + " are done",datetime.datetime.now()-now)
    return np.array([rlfdp, rltpp])