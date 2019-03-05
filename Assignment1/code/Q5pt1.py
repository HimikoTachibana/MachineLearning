import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries,features,values)=a1.load_unicef_data()
#print(countries,features)

print(values)

#print(values[:,4])

#Libera
print(np.amax(values[:,4]))

#print(np.amax(values[:,1]))

#Sierra Leone
#print(values[:,5])
print(np.amax(values[:,5]))
