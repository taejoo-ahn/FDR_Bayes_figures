import numpy as np
import util
import datetime

'''
In this simulation we perform EPoEdCe on  10 instances of generated data, and calculate FDP and TPP of each tests corresponding to target FDP control level alpha. 
We use delta=1.25 n=500, d=400 for each simulation. 
'''


delta=1.25

# set prior
sigma=0.25
weights=np.array([0.6,0.2,0.2])
points=np.array([0,1,-1])

print('Start simulation of 10 EPoEdCe for figure 2')

#solve self consistent equation and derive theoretical FDP-TPP tradeoff curve
btau=util.self_consistent_tau_star(1.25,sigma,points,weights,error=0.01)
optimals=util.bayes_optimal_tpp_fdp(btau, points, weights)

#simulation of EPoEdCe
result_epoedce=[]

for k in range(10):
	now=datetime.datetime.now()
	n=500
	d=400
	A=np.random.normal(0,(1/n)**0.5,size=(n,d))
	x=np.random.choice(points,d,p=weights)
	w=np.random.normal(0,sigma,size=n)
	y=A@x+w
	result_epoedce+=[util.EPoEdCe(A,y,x,sigma,points,weights,btau,optimals,M=50,lm=1)]
	print("out of 10 EPoEdCe simulations, " + str(k+1) + " are done", datetime.datetime.now()-now)


data=np.array(result_epoedce)
np.save("fig2data_2",data)