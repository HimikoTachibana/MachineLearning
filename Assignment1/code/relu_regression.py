#!/usr/bin/env python
import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries,features,values)=a1.load_unicef_data()

targets=values[:,1]
x=values[:,7:]

N_TRAIN=100
#Example of selecting a single feature for training 
x_train=x[0:N_TRAIN,3]
t_train=targets[0:N_TRAIN]


#Selecting a feature for both test inputs and test targets [example]
x_test=x[N_TRAIN:,3]
t_test=targets[N_TRAIN:]

#print("x_train",x_train)
(weights,training_error)=a1.linear_regression(x_train,t_train,"ReLU",0,0)
tup=(weights,training_error)

(estimate,test_err)=a1.evaluate_regression(x_test,t_test,tup[0],"ReLU",0)
tupl=(estimate,test_err)


print(tup[0])
print("training error: ",tup[1])

print("test error is: ",tupl[1])

min=np.amin(x_train)
max=np.amax(x_train)

x_ev=np.linspace(min,max,num=500)
x_ev_col=x_ev.reshape((500,1))
#(y_ev,te_err)=a1.

(y_ev,te_err)=a1.evaluate_regression(x_ev_col,None,tup[0],"ReLU",0)
tupp=(y_ev,te_err)
print(tupp[0])

plt.title("Plot of points under ReLU regression")

plt.plot(x_ev_col,y_ev,'r.-')
plt.plot(x_train,t_train,'bo')
plt.plot(x_test,t_test,'go',mfc='none')
plt.xlabel("inputs")
plt.ylabel("targets")
plt.show()
