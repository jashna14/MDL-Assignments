import numpy as np

f=open("overfit.txt","r")
txt=f.read()
# print(txt)
arr=np.array(txt)
print(arr)
print(arr[0][1])