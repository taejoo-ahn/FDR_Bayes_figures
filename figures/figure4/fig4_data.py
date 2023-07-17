import numpy as np
import util



'''
In this simulation we simulate univariate/multivariate posterior mean/zero probabilities and compare their wasserstein distances to verify formalism 1. 
We choose 3 different deltas 0.8,1.25 and 2. and for each delta we choose n=250,500,1000,2000. For each n, we generate 10 instances of data and calculate wasserstein distances. 
'''


#set prior
p=0.2
points=np.array([0,1,-1])
weights=np.array([1-2*p,p,p])
sigma=0.25


#simulation for delta=0.8
print('delta=0.8, n=240, calculating wasserstein distances for formailsm 1')
delta1_1=util.formalism1(240,300,0.25,points,weights)
print('delta=0.8, n=500, calculating wasserstein distances for formailsm 1')
delta1_2=util.formalism1(500,625,0.25,points,weights)
print('delta=0.8, n=1000, calculating wasserstein distances for formailsm 1')
delta1_3=util.formalism1(1000,1250,0.25,points,weights)
print('delta=0.8, n=2000, calculating wasserstein distances for formailsm 1')
delta1_4=util.formalism1(2000,2500,0.25,points,weights)

data_delta_1=np.array([delta1_1,delta1_2,delta1_3,delta1_4])

#simulation for delta=1.25
print('delta=1.25, n=250, calculating wasserstein distances for formailsm 1')
delta2_1=util.formalism1(250,200,0.25,points,weights)
print('delta=1.25, n=500, calculating wasserstein distances for formailsm 1')
delta2_2=util.formalism1(500,400,0.25,points,weights)
print('delta=1.25, n=1000, calculating wasserstein distances for formailsm 1')
delta2_3=util.formalism1(1000,800,0.25,points,weights)
print('delta=1.25, n=2000, calculating wasserstein distances for formailsm 1')
delta2_4=util.formalism1(2000,1600,0.25,points,weights)

data_delta_2=np.array([delta2_1,delta2_2,delta2_3,delta2_4])

#simulation for delta=2
print('delta=2, n=250, calculating wasserstein distances for formailsm 1')
delta3_1=util.formalism1(250,125,0.25,points,weights)
print('delta=2, n=500, calculating wasserstein distances for formailsm 1')
delta3_2=util.formalism1(500,250,0.25,points,weights)
print('delta=2, n=1000, calculating wasserstein distances for formailsm 1')
delta3_3=util.formalism1(1000,500,0.25,points,weights)
print('delta=2, n=2000, calculating wasserstein distances for formailsm 1')
delta3_4=util.formalism1(2000,1000,0.25,points,weights)

data_delta_3=np.array([delta3_1,delta3_2,delta3_3,delta3_4])


data=np.array([data_delta_1,data_delta_2,data_delta_3])
np.save("fig4data",data)