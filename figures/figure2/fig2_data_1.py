import numpy as np
import util

'''
In this simulation we perform PoPCe and PoEdCe on  10 instances of generated data, and calculate FDP and TPP of each tests corresponding to target FDP control level alpha. 
We use delta=1.25 n=500, d=400 for each simulation. 
'''


delta=1.25

# set prior
sigma=0.25
weights=np.array([0.6,0.2,0.2])
points=np.array([0,1,-1])


print('Start simulation of 10 PoPCe and PoEdCe for figure 2')

#solve self consistent equation and derive theoretical FDP-TPP tradeoff curve
btau=util.self_consistent_tau_star(1.25,sigma,points,weights,error=0.01)
optimals=util.bayes_optimal_tpp_fdp(btau, points, weights)


#simulation of PoPCe and PoEdCe
result_popce=[]
result_poedce=[]

for k in range(10):
    n=500
    d=400
    A=np.random.normal(0,(1/n)**0.5,size=(n,d))
    x=np.random.choice(points,d,p=weights)
    w=np.random.normal(0,sigma,size=n)
    y=A@x+w
    now=datetime.datetime.now()
    result_popce+=[util.PoPCe(A,y,x,sigma,points,weights,btau,optimals)]
    print('out of 10 PoPCe simulations,'+str(k+1)+' are done',datetime.datetime.now()-now)
    now=datetime.datetime.now()
    result_poedce+=[util.PoEdCe(A,y,x,sigma,points,weights,btau,optimals,numalpha=60)]
    print('out of 10 PoEdCe simulations,'+str(k+1)+' are done',datetime.datetime.now()-now)


data=np.array([result_popce,result_poedce,optimals])
np.save("fig2data_1",data)