import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


nero=[
    [4,5,6],
    [6,-7000,8],
    [8,12,-143000000]
]

nerotrix=np.asmatrix(nero)

a=np.add(nerotrix,500)

r=np.maximum(a,0,a)

result=None

newMatrix=np.add(nerotrix,500)
reLUtrix=np.maximum(newMatrix,0,newMatrix)
if result is None:
    result=reLUtrix
else:
    result=np.hstack((result,reLUtrix))

res_rows=result.shape[0]
ones_col=np.ones((res_rows,1))
phi=np.hstack((ones_col,result))

print(r)
print(phi)



#print(a)

#print("Value of r: ",r)
