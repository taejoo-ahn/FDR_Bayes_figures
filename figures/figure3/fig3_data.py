import numpy as np
import util
import datetime


'''
In this simulation we perform 3 different PoEdCe test on  20 instances of generated data, with one as usual, one with misspecified prior and one with misspecified noise level
then we calculate FDP and TPP of each tests corresponding to target FDP control level alpha.

We use delta=1.25 n=500, d=400, and delta=0.8, n=400, d=500 for 10 times each. 
'''


#set prior
points=np.array([0,1,-1])
weights=np.array([0.4,0.5,0.1])
mis_prior=np.array([0.6,0.1,0.3])
sigma=0.25


#delta=1.25
delta=1.25

#solve self consistent equation and derive theoretical FDP-TPP tradeoff curve for each misspecifications
btau=util.self_consistent_tau_star(delta,sigma,points,weights,0.01)
optimals=util.bayes_optimal_tpp_fdp(btau,points,weights)

mis_prior_btau=util.self_consistent_tau_star(delta,sigma,points,mis_prior,0.01)
mis_prior_optimals=util.bayes_optimal_tpp_fdp(mis_prior_btau,points,mis_prior)

mis_noise_btau=util.self_consistent_tau_star(delta,sigma*2,points,weights,0.01)
mis_noise_optimals=util.bayes_optimal_tpp_fdp(mis_noise_btau,points,weights)


#start simulation for delta=1.25
correct_fdp_1=[]
correct_tpp_1=[]
mis_prior_fdp_1=[]
mis_prior_tpp_1=[]
mis_noise_fdp_1=[]
mis_noise_tpp_1=[]

print('Start simulation of PoEdCe for figure 3, delta=1.25')
for j in range(10):
    n=500
    d=400
    A=np.random.normal(0,(1/n)**0.5,size=(n,d))
    x=np.random.choice(points,d,p=weights)
    w=np.random.normal(0,sigma,size=n)
    y=A@x+w

    now=datetime.datetime.now()
    correct_test=util.PoEdCe(A,y,x,sigma,points,weights,btau,optimals,numalpha=40)
    mis_prior_test=util.PoEdCe(A,y,x,sigma,points,mis_prior,mis_prior_btau,mis_prior_optimals,numalpha=40)
    mis_noise_test=util.PoEdCe(A,y,x,sigma*2,points,weights,mis_noise_btau,mis_noise_optimals,numalpha=40)
    print('PoEdCe with correct and misspecified models, delta=1.25, out of 10 simulations '+str(j+1)+' are done',datetime.datetime.now()-now)
    
    correct_fdp_1+=[correct_test[1]]
    correct_tpp_1+=[correct_test[0]]
    
    mis_prior_fdp_1+=[mis_prior_test[1]]
    mis_prior_tpp_1+=[mis_prior_test[0]]
    
    mis_noise_fdp_1+=[mis_noise_test[1]]
    mis_noise_tpp_1+=[mis_noise_test[0]]


result_1=np.array([correct_fdp_1, correct_tpp_1, mis_prior_fdp_1, mis_prior_tpp_1, mis_noise_fdp_1, mis_noise_tpp_1,optimals])

#delta=0.8
delta=0.8

#solve self consistent equation and derive theoretical FDP-TPP tradeoff curve for each misspecifications
btau=util.self_consistent_tau_star(delta,sigma,points,weights,0.01)
optimals=util.bayes_optimal_tpp_fdp(btau,points,weights)

mis_prior_btau=util.self_consistent_tau_star(delta,sigma,points,mis_prior,0.01)
mis_prior_optimals=util.bayes_optimal_tpp_fdp(mis_prior_btau,points,mis_prior)

mis_noise_btau=util.self_consistent_tau_star(delta,sigma*2,points,weights,0.01)
mis_noise_optimals=util.bayes_optimal_tpp_fdp(mis_noise_btau,points,weights)


correct_fdp_2=[]
correct_tpp_2=[]
mis_prior_fdp_2=[]
mis_prior_tpp_2=[]
mis_noise_fdp_2=[]
mis_noise_tpp_2=[]

#start simulation for delta=0.8

print('Start simulation of PoEdCe for figure 3, delta=0.8')
for j in range(10):
    n=400
    d=500
    A=np.random.normal(0,(1/n)**0.5,size=(n,d))
    x=np.random.choice(points,d,p=weights)
    w=np.random.normal(0,sigma,size=n)
    y=A@x+w
    
    
    now=datetime.datetime.now()
    correct_test=util.PoEdCe(A,y,x,sigma,points,weights,btau,optimals,numalpha=40)
    mis_prior_test=util.PoEdCe(A,y,x,sigma,points,mis_prior,mis_prior_btau,mis_prior_optimals,numalpha=40)
    mis_noise_test=util.PoEdCe(A,y,x,sigma*2,points,weights,mis_noise_btau,mis_noise_optimals,numalpha=40)
    
    print('PoEdCe with correct and misspecified models, delta=0.8, out of 10 simulations '+str(j+1)+' are done',datetime.datetime.now()-now)
    correct_fdp_2+=[correct_test[1]]
    correct_tpp_2+=[correct_test[0]]
    
    mis_prior_fdp_2+=[mis_prior_test[1]]
    mis_prior_tpp_2+=[mis_prior_test[0]]
    
    mis_noise_fdp_2+=[mis_noise_test[1]]
    mis_noise_tpp_2+=[mis_noise_test[0]]

result_2=np.array([correct_fdp_2, correct_tpp_2, mis_prior_fdp_2, mis_prior_tpp_2, mis_noise_fdp_2, mis_noise_tpp_2,optimals])


data=np.array([result_1,result_2])
np.save("fig3data",data)