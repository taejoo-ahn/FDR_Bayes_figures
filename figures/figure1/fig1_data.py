
import numpy as np
import util

'''
In this simulation we compare TPoP, CPoP and thresholded LASSO. We generate 10 instances of simulated data and apply three different tests and plot FDP and TPP, 
and calculate theoretical FDP-TPP tradeoff of each tests. 

We use delta=1.25 n=2000, d=1600 and delta=0.8, n=1600, d=2000 for each simulation. 
'''


# set prior
p=0.2
points=np.array([0,1,-1])
weights=np.array([1-2*p,p,p])
# set noise level
sigma=0.25


#for delta=1.25
delta=1.25

print('Start tpop,cpop and lasso simulation for delta=1.25 for figure1')

#solve self consistent equation
btau1=util.self_consistent_tau_star(delta,sigma,points,weights,0.01)

#simulations
bayes_theo_1=util.bayes_optimal_tpp_fdp(btau1,points,weights)
lasso_theo_1=util.lasso_theoretical_tpp_fdp(delta,sigma,points,weights)
bayes_data_1=util.tpop_cpop_simulation(2000,1600,delta,points,weights,sigma)
lasso_data_1=util.lasso_simulation(2000,1600,delta,points,weights,sigma,lasso_theo_1[2])

#for delta=0.8
delta=0.8

print('Start tpop,cpop and lasso simulation for delta=0.8 for figure1')

#solve self consistent equation
btau2=util.self_consistent_tau_star(delta,sigma,points,weights,0.01)

#simulations
bayes_theo_2=util.bayes_optimal_tpp_fdp(btau2,points,weights)
lasso_theo_2=util.lasso_theoretical_tpp_fdp(delta,sigma,points,weights)
bayes_data_2=util.tpop_cpop_simulation(1600,2000,delta,points,weights,sigma)
lasso_data_2=util.lasso_simulation(1600,2000,delta,points,weights,sigma,lasso_theo_2[2])


data=np.array([bayes_theo_1,lasso_theo_1,bayes_data_1,lasso_data_1,bayes_theo_2,lasso_theo_2,bayes_data_2,lasso_data_2])
np.save("fig1data",data)