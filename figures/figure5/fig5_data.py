import numpy as np
import util
import datetime


'''
In this simulation we simulate pvalues from CRT, dCRT and simulated sample from their theoretically calculated distributions, and compare their wasserstein distances to verify formalism 2,3. 
We choose 3 different deltas 0.8,1.25 and 2. and for each delta we choose n=50,100,200,500 for CRT and formalism2, n=200,500,1000,2000 for dCRT and formalism3.
For each n, we generate 10 instances of data and calculate wasserstein distances. 
'''


#set prior
p=0.2
points=np.array([0,1,-1])
weights=np.array([1-2*p,p,p])
sigma=0.25
domain=[i/50-4 for i in range(400)]


#delta =0.8
delta=0.8
now=datetime.datetime.now()
btau=util.self_consistent_tau_star(delta,sigma,points,weights,0.01)
print('calculating psis for delta=0.8')
ps=[util.univariate_posteriors(c,btau,points,weights)[1] for c in domain]
psis1=[util.psi(c,btau,points,weights) for c in ps]
print('done calculating psis',datetime.datetime.now()-now)

#formalism 2 for delta=0.8
print('delta=0.8, n=50, calculating wasserstein distances for formailsm 2')
formlaism2_d11=util.formalism2(40,50,0.25,points,weights,btau,psis1)
print("delta=0.8, n=100, calculating wasserstein distances for formailsm 2")
formlaism2_d12=util.formalism2(100,125,0.25,points,weights,btau,psis1)
print("delta=0.8, n=200, calculating wasserstein distances for formailsm 2")
formlaism2_d13=util.formalism2(200,250,0.25,points,weights,btau,psis1)
print("delta=0.8, n=500, calculating wasserstein distances for formailsm 2")
formlaism2_d14=util.formalism2(500,625,0.25,points,weights,btau,psis1)

formlaism2_data1=np.array([formlaism2_d11,formlaism2_d12,formlaism2_d13,formlaism2_d14])

#formalism 3 for delta=0.8


print("delta=0.8, n=200, calculating wasserstein distances for formailsm 3")
formlaism3_d11=util.formalism3(200,250,0.25,points,weights,btau,psis1)
print("delta=0.8, n=500, calculating wasserstein distances for formailsm 3")
formlaism3_d12=util.formalism3(500,625,0.25,points,weights,btau,psis1)
print("delta=0.8, n=1000, calculating wasserstein distances for formailsm 3")
formlaism3_d13=util.formalism3(1000,1250,0.25,points,weights,btau,psis1)
print("delta=0.8, n=2000, calculating wasserstein distances for formailsm 3")
formlaism3_d14=util.formalism3(2000,2500,0.25,points,weights,btau,psis1)

formlaism3_data1=np.array([formlaism3_d11,formlaism3_d12,formlaism3_d13,formlaism3_d14])


#delta=1.25
delta=1.25
now=datetime.datetime.now()
btau=util.self_consistent_tau_star(delta,sigma,points,weights,0.01)
print('calculating psis for delta=1.25')
ps=[util.univariate_posteriors(c,btau,points,weights)[1] for c in domain]
psis2=[util.psi(c,btau,points,weights) for c in ps]
print('done calculating psis',datetime.datetime.now()-now)

#formalism 2 for delta=1.25

print("delta=1.25, n=50, calculating wasserstein distances for formailsm 2")
formlaism2_d21=util.formalism2(50,40,0.25,points,weights,btau,psis2)
print("delta=1.25, n=100, calculating wasserstein distances for formailsm 2")
formlaism2_d22=util.formalism2(100,80,0.25,points,weights,btau,psis2)
print("delta=1.25, n=200, calculating wasserstein distances for formailsm 2")
formlaism2_d23=util.formalism2(200,160,0.25,points,weights,btau,psis2)
print("delta=1.25, n=500, calculating wasserstein distances for formailsm 2")
formlaism2_d24=util.formalism2(500,400,0.25,points,weights,btau,psis2)

formlaism2_data2=np.array([formlaism2_d21,formlaism2_d22,formlaism2_d23,formlaism2_d24])

#formalism 3 for delta=0.8

print("delta=1.25, n=200, calculating wasserstein distances for formailsm 3")
formlaism3_d21=util.formalism3(200,160,0.25,points,weights,btau,psis2)
print("delta=1.25, n=500, calculating wasserstein distances for formailsm 3")
formlaism3_d22=util.formalism3(500,400,0.25,points,weights,btau,psis2)
print("delta=1.25, n=1000, calculating wasserstein distances for formailsm 3")
formlaism3_d23=util.formalism3(1000,800,0.25,points,weights,btau,psis2)
print("delta=1.25, n=2000, calculating wasserstein distances for formailsm 3")
formlaism3_d24=util.formalism3(2000,1600,0.25,points,weights,btau,psis2)

formlaism3_data2=np.array([formlaism3_d21,formlaism3_d22,formlaism3_d23,formlaism3_d24])

#delta=2

delta=2
now=datetime.datetime.now()
btau=util.self_consistent_tau_star(delta,sigma,points,weights,0.01)
print('calculating psis for delta=2')
ps=[util.univariate_posteriors(c,btau,points,weights)[1] for c in domain]
psis3=[util.psi(c,btau,points,weights) for c in ps]
print('done calculating psis',datetime.datetime.now()-now)

#formalism 2 for delta=2
print("delta=2, n=50, calculating wasserstein distances for formailsm 2")
formlaism2_d31=util.formalism2(50,25,0.25,points,weights,btau,psis3)
print("delta=2, n=100, calculating wasserstein distances for formailsm 2")
formlaism2_d32=util.formalism2(100,50,0.25,points,weights,btau,psis3)
print("delta=2, n=200, calculating wasserstein distances for formailsm 2")
formlaism2_d33=util.formalism2(200,100,0.25,points,weights,btau,psis3)
print("delta=2, n=500, calculating wasserstein distances for formailsm 2")
formlaism2_d34=util.formalism2(500,250,0.25,points,weights,btau,psis3)

formlaism2_data3=np.array([formlaism2_d31,formlaism2_d32,formlaism2_d33,formlaism2_d34])


#formalism 3 for delta=2

print("delta=2, n=200, calculating wasserstein distances for formailsm 3")
formlaism3_d31=util.formalism3(200,100,0.25,points,weights,btau,psis3)
print("delta=2, n=500, calculating wasserstein distances for formailsm 3")
formlaism3_d32=util.formalism3(500,250,0.25,points,weights,btau,psis3)
print("delta=2, n=1000, calculating wasserstein distances for formailsm 3")
formlaism3_d33=util.formalism3(1000,500,0.25,points,weights,btau,psis3)
print("delta=2, n=2000, calculating wasserstein distances for formailsm 3")
formlaism3_d34=util.formalism3(2000,1000,0.25,points,weights,btau,psis3)

formlaism3_data3=np.array([formlaism3_d31,formlaism3_d32,formlaism3_d33,formlaism3_d34])





formlaism2_data=np.array([formlaism2_data1,formlaism2_data2,formlaism2_data3])

formlaism3_data=np.array([formlaism3_data1,formlaism3_data2,formlaism3_data3])

data=[formlaism2_data,formlaism3_data]


np.save("fig5data",data)

