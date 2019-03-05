import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features,values)=a1.load_unicef_data()
reg_lambda_list=[0,0.01,.1,1,10,10**2,10**3,10**4]

v=values[:,7:]
targets=values[:,1]

N_TRAIN=100

x=v[0:N_TRAIN,:]
t=targets[0:N_TRAIN,:]

avg_list=[]
for l in reg_lambda_list:
	error_list=[]
	sig=0
	avg=0
	#avg_list=[]
	for i in range(0,10):
		error_list=[]
		#sig=0
		#avg=0
		#avg_list=[]
		#print("i",i)
		xv=x[0:10,:]
		#print("xv",xv)
		xt=x[10:,:]
		#print("xt",xt)
		tv=t[0:10,:]
		#print("tv",tv)
		tt=t[10:,:]
		#print("tt",tt)
		print("lambda: ",l)
		w,_=a1.linear_regression(xt,tt,"polynomial",l,2)
		#print("w at i",i,w)
		x=np.vstack((xt,xv))
		#print("x",x)
		#x=np.vstack((xv,xt))
		## For debug feel free to comment out to see diagonistic information
		#xv=x[0:10,:]
		#tv=t[0:10:,:]
		_,v_err=a1.evaluate_regression(xv,tv,w,"polynomial",2)
		print("validation error",v_err)
		t=np.vstack((tt,tv))
		#print("t",t)
		#t=np.vstack((tv,tt))
		error_list.append(v_err)
		#sig=sum(error_list)
		#avg=sig/len(error_list)
		#avg_list.append(avg)
		print(v_err,"at",i)
		#sig+=v_err
		##print("Weight vector is: ")
		##print(w)
	#print("l",l,error_list)
	#sig=0
	#sig=sum(error_list)
	#print("sig",l,sig)
	#avg=0
	print("sig",sig)
	avg=sum(error_list)/len(error_list)
	print("avg",avg)
	#print("Length of error list: ",len(error_list))
	#print("l: ",l,avg)
	avg_list.append(avg)
	print("avg_list",avg_list)
	#print(sig)
	#print (l)
        #xv=x[0:10,:]
        #xt=x[10:,:]

print(avg_list)
print(len(avg_list))

error_dict={}


for i in range(len(avg_list)):
	error_dict.update({reg_lambda_list[i]:avg_list[i]})

print("For diagnostic purposes: ")
#print(error_dict)
print("Keys of dictionary: ")
print(error_dict.keys())

print("Values of dictionary: ")
print(error_dict.values())

plt.title("Average validation error against lambda")
plt.xlabel("Lambda")
plt.ylabel("Average validation error")
plt.semilogx(error_dict.keys(),error_dict.values(),'b.-')
plt.show()
