#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# list to hold onto our list of weights, training error for each degree
lst=[]
# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py

degree_cap=6
for i in range(1,degree_cap+1):
    (w,tr_err)=a1.linear_regression(x_train,t_train,"polynomial",0,i)
    tup=(w,tr_err)
    lst.append(tup)



lengthlst=len(lst)

for i in range(lengthlst):
    print("---",lst[i][1])


lstZ=[]

print("length of list ",lengthlst)

for i in range(1,lengthlst+1):
    #print("i",i)
    #print("i-1",i-1)
    #print("i-1",i-1)
    #print(lst[i-1][0].shape[0])
    (t_est,te_err)=a1.evaluate_regression(x_test,t_test,lst[i-1][0],"polynomial",i)
    tup2=(t_est,te_err)
    lstZ.append(tup2)
#for i in range(lengthlst):
    #(t_est,te_err)=a1.evaluate_regression(x_test,t_test,lst[i][0],"polynomial",i+1)
    #tup2=(t_est,te_err)
    #lstZ.append(tup2)

#print(len(lstZ))

#for i in range(len(lstZ)):
    #print(lstZ[i][1])

#(w,tr_err) = a1.linear_regression(x_train,t_train,"polynomial",0,1)
#tup=(w,tr_err)
#lst.append(tup)
#print("weights, training error tuple")
#(w,tr_err)=a1.linear_regression(x_train,t_train,"polynomial",0,2)
#tup=(w,tr_err)
#lst.append(tup)

#print(lst[0][1])
#print(tup[0].shape[0])

#print("Size of weights vector: ",w.shape[0])

#print([0].shape[0]))

## We map polynomial degree to a specific error value
train_err={}
test_err={}

lArg=len(lstZ)

# lst contains our training error
# lstZ contains our test error vector
for i in range(1,lArg+1):
    train_err.update({i:lst[i-1][1]})
    test_err.update({i:lstZ[i-1][1]})

# (w, tr_err) = a1.linear_regression()
# (t_est, te_err) = a1.evaluate_regression()






# Produce a plot of results.


plt.plot(train_err.keys(), train_err.values())
plt.plot(test_err.keys(), test_err.values())


plt.ylabel('RMS')
plt.legend(['Training Error','Test error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')

#plt.xlim((0,6))
plt.show()
