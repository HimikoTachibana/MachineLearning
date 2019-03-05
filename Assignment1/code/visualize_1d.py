#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]


#x = a1.normalize_data(x)

N_TRAIN = 100;
# Select a single feature for training both training inputs and training targets
x_train = x[0:N_TRAIN,10]
t_train = targets[0:N_TRAIN]

#Selecting a feature for test both test inputs and test targets

x_test=x[N_TRAIN:,10]
t_test=targets[N_TRAIN:]

## --- For debugging diagnostics, feel free to comment out
#print("Value of x: ",x)
#print("x_train vector ",x_train)
#print("t_train vector ",t_train)
#x_train2=x[0:N_TRAIN,0]
#print("value of values[0:N_TRAIN,0]",x_train2)

#### Begin code to display bar plot

number_of_input_features=8

## List to store our input features for training
feature_list_input_train=[]
for i in range(number_of_input_features):
    #print("Value of i: ",i)
    #print("Value at x[0:N_TRAIN,",i," ] ",x[0:N_TRAIN,i])
    feature_list_input_train.append(x[0:N_TRAIN,i])

## List to store our test inputs for test validation
feature_list_target_test=[]
for i in range(number_of_input_features):
    feature_list_target_test.append(x[N_TRAIN:,i])

print("---Printing element list from feature list---")

print("Target dimensions: ",t_train.shape[0],t_train.shape[1])
lenArg=len(feature_list_input_train)

# List to store the following as elements (weights,trainingerror)
werr=[]
for i in range(lenArg):
    (w,tr_err)=a1.linear_regression(feature_list_input_train[i],t_train,"polynomial",0,3)
    tup=(w,tr_err)
    werr.append(tup)
    #print(feature_list[i].shape[0])
    #print(feature_list[i].shape[1])

#for j in range(len(werr)):
    #print(werr[j][0])


### List to store the following as elements (estimates, te_err)
lstZ=[]
for i in range(lenArg):
    (t_est,te_err)=a1.evaluate_regression(feature_list_target_test[i],t_test,werr[i][0],"polynomial",3)
    tup2=(t_est,te_err)
    lstZ.append(tup2)

## Note: technically we do not need these list variables but we put them there for code readability
training_error_list=[]
test_error_list=[]
for i in range(lenArg):
   #print("training error, test error: ",werr[i][1],lstZ[i][1])
    training_error_list.append(werr[i][1])
    test_error_list.append(lstZ[i][1])

### To do on above: refactor when have time

""" Code for barplot comment out to see barplot"""

n_features=8
fig,ax=plt.subplots()
index=np.arange(n_features)
bar_width=0.35
opacity=0.8

rects1=plt.bar(index,training_error_list,bar_width,alpha=opacity,color='b',label='training error')
rects2=plt.bar(index+bar_width,test_error_list,bar_width,alpha=opacity,color='g',label='test error')

plt.xlabel('Feature')
plt.ylabel('Error')
plt.title('Errors for each feature')
plt.xticks(index+bar_width,('8f1','9f2','10f3','11f4','12f5','13f6','14f7','15f8'),rotation=170)
plt.legend()

plt.tight_layout()
plt.show()




#print("Value of min(x_train): ")
#print(min(x_train))
#print("Value of max(x_train): ")
#print(max(x_train))

####



# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate

### Begin code to plot actual polynomial
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
#x_ev_col=x_ev[:np.newaxis]
x_ev_col=x_ev.reshape((500,1))
y_dummy=np.ones((x.shape[0],1))


#print("x_ev",x_ev)
#for i in range(3):



#weight_list=[]
#for i in range(3,6):
    #(w,tr_err)=a1.linear

## List to store our list of features
f_list=[]
for i in range(3,6):
    f_list.append(x[0:N_TRAIN,i])

### List to store our interval of x coordinates



co_ords=[]
#min=min(x[0:N_TRAIN,3])

#print(min)
for i in range(3,6):
    #print("i",i)
    min=np.amin(x[0:N_TRAIN,i])
    max=np.amax(x[0:N_TRAIN,i])
    #print(min)
    #print(max)
    #print("i: ",i)
    #print(x[0:N_TRAIN,i])
    #print(min)
    #print(max)

    x_ev=np.linspace(np.asscalar(min),np.asscalar(max),num=500)
    #print(x_ev)
    #x_ev_col=x_ev[:np.newaxis]
    x_ev_col=x_ev.reshape((500,1))
    co_ords.append(x_ev_col)



## List to store our test inputs for testing | and later plotting
f_lst_input=[]
for i in range(3,6):
    f_lst_input.append(x[N_TRAIN:,i])

##List to store our test targets for testing | and later plotting


# List to store the following as elements (weights,trainingerror)

weighterror=[]
length=len(f_list)

for i in range(length):
    #print("Inside weight error loop: ")
    #print(f_list[i])
    (weights,trainingerror)=a1.linear_regression(f_list[i],t_train,"polynomial",0,3)
    tup=(weights,trainingerror)
    weighterror.append(tup)

#print("weighterror[0]",weighterror[0])
#r=weighterror[0][0].dot(x_ev)
#for i in range(len(weighterror)):
    #print("Weight set: ")
    #print(weighterror[i][0])
#List to store the following as elements (estimates, te_err)

#estlist=[]
#for i in range(length):
    #(estimates,te_err)=a1.evaluate_regression(feature_list_target_test[i],t_test,weighterror[i][0],"polynomial",3)
    #tup=(estimates,te_err)
    #estlist.append(tup)

train_error=[]
test_error=[]

for i in range(length):
    train_error.append(weighterror[i][1])
    #test_error.append(estlist[i][1])

## Create a y_dummy vectors full of ones to compute y estimates for each feature 

for i in range(length):
    print("Value of i in weighterror loop:",i)
    print(weighterror[i][0])

y_dummy_vec=[]
for i in range(3):
    y_dummy=np.ones((co_ords[i].shape[0],1))
    y_dummy_vec.append(y_dummy)

### Evaluate regression on plotting points using previously computed weights

#y_ev_vec=[]
#for j in range(3):
     #print("Value of j: ",j)
     #print(co_ords[i].shape[0])
     #print(y_dummy_vec[i].shape[0])
     #print(weighterror[i][0])
     #print("Value of j: ",j)
     #print(co_ords[j])

     #for k in range(co_ords[j].shape[0]):
         #print(co_ords[j][k])
     #y_ev,_=a1.evaluate_regression(co_ords[i],y_dummy_vec[i],weighterror[i][0],"polynomial",3)
     #y_ev_vec.append(y_ev)

reversecoeffs=[]
for j in range(3):
    coeffs=np.flip(weighterror[i][0],0)
    reversecoeffs.append(coeffs)

#print(reversecoeffs[0])
y_ev_vec=[]
for k in range(3):
    #y_ev=np.polyval(reversecoeffs[k],co_ords[k])
    print(co_ords[k].shape[0])
   
    y_ev=a1.evaluate_regression(co_ords[k],None,weighterror[k][0],"polynomial",3)
    y_ev_vec.append(y_ev)

#print(type(y_ev_vec[0]))

#print("The contents of x_ev are: ")
#print(x_ev)
# TO DO:: Put your regression estimate here in place of x_ev.
# Evaluate regression on the linspace samples.
# y_ev, _  = a1.evaluate_regression()

#print(len(y_ev_vec[0]))

for i in range(3):
    print(i)


plt.figure(200)
print(co_ords[0].shape[0])
print(co_ords[0].shape[1])
print("y_ev_vec[0][0]",y_ev_vec[0][0])
print("y_ev_vec[0][0].shape[0]",y_ev_vec[0][0].shape[0])
#print(y_ev_vec[0])

plt.title("Plot of learned polynomial against datapoints")
plt.plot(co_ords[0],y_ev_vec[0][0],'r.-')
plt.plot(f_list[0],t_train,'bo')
plt.plot(f_lst_input[0],t_test,'go')
plt.show()

plt.figure(300)
plt.title("Plot of learned polynomial against datapoints")
plt.plot(co_ords[1],y_ev_vec[1][0],'r.-')
plt.plot(f_list[1],t_train,'bo')
plt.plot(f_lst_input[1],t_test,'go')
plt.show()

plt.figure(400)
plt.title("Plot of learned polynomial against datapoints")
#plt.legend("Function","Points")
plt.plot(co_ords[2],y_ev_vec[2][0],'r-')
plt.plot(f_list[2],t_train,'bo')
plt.plot(f_lst_input[2],t_test,'go')
plt.show()


"""

plt.plot(x_ev,y_ev,'r.-')
plt.plot(x_train,t_train,'bo')
plt.title('A visualization of a regression estimate using random outputs')
plt.show()

"""



