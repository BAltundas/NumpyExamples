# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:38:41 2021

@author: Bilgin
https://numpy.org/doc/stable/user/quickstart.html
"""
### Examples of Numpy
import numpy as np
#
x= np.arange(15).reshape(5,3)
rwx,clmx= x.shape
print(x)
print(clmx,rwx)

np.zeros(12).reshape(4,3)
np.zeros((4,3))

np.empty(36).reshape(6,6)

x = np.array([1,2,3,4,5,6,7])
print(type(x))
print(x)

# create a numpy array between 10 and 30 with increments of 5
x=np.arange( 10, 30, 5 )
print(type(x))
print(x)

# creating numpy array using linspace
pi=np.pi
x = np.linspace( 0, 2*pi, 5 )    
print(type(x))
print(x)    

# using functions
fx = np.sin(x)
print(type(fx))
print(fx)    

# 
# operations with numpy arrays
x=np.arange( 10, 30, 5 )
y=np.arange( 15, 35, 5 )
z=x-y
zxz = z*z
z2=z**2  # square entrywise
print(z)
print(zxz)  # dot entrywise product
print(z2)

z = x<y  # entrywise comparison
print(x,'\n',y)
print(z)

# matrix operations

x = np.arange(12).reshape(2,6)
y = np.arange(12).reshape(6,2)

# 2 ways of matrix product
print(x,'\n',y)
z = x@y
z1= np.dot(x,y)

print(z,'\n',z1)

# entrywise product
z = x*x
print(x,'\n',z)

# adding matrices
z=x+x
print('x=',x,'\n','x+x=',z)
# copying mutable objects (dictionary and list)
import copy
dict = [1,2,3,4,5,6,7]
dict2=dict
dict3=copy.copy(dict)
print(dict,'\n',dict2,'\n',dict3)
dict[3]=-10
print(dict,'\n',dict2,'\n',dict3)

# max, min, sum, mean, st
x=np.arange(12).reshape(3,4)

print(x,'\n','sum=',np.sum(x),'\n','min_x= ',np.min(x),'\n','max_x = ',np.max(x))
print('axis=0 rowwise')
print(x,'\n','sum=',np.sum(x, axis=0),'\n','min_x= ',np.min(x,axis=0),'\n','max_x = ',np.max(x,axis=0))
print('axis=1 columnwise')
print(x,'\n','sum=',np.sum(x, axis=1),'\n','min_x= ',np.min(x,axis=1),'\n','max_x = ',np.max(x,axis=1))

print(x)
print(np.mean(x))
print(np.mean(x,axis=0))
print(np.mean(x,axis=1))
print(np.std(x))

# loading and saving to files
fname='npdata.txt'
with open(fname) as f:
    x = f.read()

print(x)
#  numpy.any(a, axis=None, out=None, keepdims=<no value>)[source]
# Test whether any array element along a given axis evaluates to True.
x=np.arange(12)
y=np.linspace(1,5,12)
z=x < y
print(z)
print(x,'\n',y)
print(np.any(z))
print(all(z))

# sorting
x=np.random.randint(19, size=20)
print(x)
ys=np.sort(x)
print('sorted x=',ys)

# sorting indices
ind = np.argsort(x, axis=0)

# sorting x using indices
x=np.take_along_axis(x, ind, axis=0)  # same as np.sort(x, axis=0)
print(x)

# Column stack
x=np.random.randint(5,size=(5,1))
y=np.random.randint(5,size=(5,1))
z=np.column_stack((x,y))
print(z)

######################### use of where ##############################
# numpy.where(condition[, x, y])
# Return elements chosen from x or y depending on condition.
x=np.arange(8).reshape(4,2)
y=np.where(x<2, 0,1)
print(y)

# slicing
a = np.arange(10)**3
a[1:5]
a[0:8:2]
a[ : :-1] # this alos reverse the roder

################################  meshgrid ##########################
x=np.arange(4)
y= x+1
xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

# passing a function to create a matrix
def f(x,y):
    print(x,y)
    return 10*x+y

b = np.fromfunction(f,(4,3),dtype=int)
print('**')
print(b)
b[:,1]
b[3,:]
b[3][:]
# write the whole entries
for element in b.flat:
    print(element)
    
for index, x in np.ndenumerate(b):
    x[index]
    print(index, x)

x = np.arange(6).reshape(2, 3)
# indices retunr the indices of x
row, col = np.indices((2, 3))
x[row, col]
row
col

############################# SORT ##################################
# count, search
# Numpy.argmax(a, axis=None, out=None)[source]
# Returns the indices of the maximum values along an axis.
import numpy as np
x=np.random.randint(19, size=(5))
print(x)
max_in = np.argmax(x, axis=0)
max_in

x=np.random.randint(19, size=(3,5))
print(x)
max_indx = np.argmax(x, axis=1)
min_indx = np.argmin(x, axis=1)
print(min_indx)
print(max_indx)

# SEARCHSORTED
# np.searchsorted([1,2,3,4,5], 3)
# 
x=np.array([ 2, 10,  6, 18, 15])
print(np.sort(x))
indx = np.searchsorted(x, 15)
print(indx)

print(np.searchsorted([1,2,4,3,5], 4))


##############################  ARGWHERE for finding index of entries ######
# numpy.argwhere(a)[source]¶
# Find the indices of array elements that are non-zero, grouped by element.


x=np.array([ 2, 2,2,2,10, 0, 6, 18, 15,2,3,5,6,7,8,3,4,5,6])

x_unique = np.unique(x)
xdict_count = {}
for entry in x_unique:
    xdict_count[str(entry)]=np.sum(x==entry)
print(xdict_count) 

xdict_indx = {}
for entry in x_unique:
    ind = np.argwhere(x==entry)
    xdict_indx[str(entry)]=ind.tolist()
    
print(xdict_indx)   

############################  NUMPY.INSERT
# numpy.insert
# numpy.insert(arr, obj, values, axis=None)[source]
# Insert values along the given axis before the given indices.

#1st approach
x = np.random.randint(12,size=20)
indx=0
indx_val = 65
x_new = np.insert(x, indx, indx_val)
print(x,'\n',x_new)

#2nd approach
x_list = x.tolist()
print(x_list)
x_list.insert(indx,indx_val)
print(x_list)


########################## delete ###################################
# numpy.delete

# numpy.delete(arr, obj, axis=None)[source]
# Return a new array with sub-arrays along an axis deleted. For a one dimensional array, 
# this returns those entries not returned by arr[obj].

# numpy.delete(arr, row_indx (or column_indx), 0 (for a deleting row) or 1 (for deleting a column))

arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
narr = np.delete(arr, 1, 0)
print(arr,'\n','\n',narr)

# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]] 
 
#  [[ 1  2  3  4]
#  [ 9 10 11 12]]

arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
narr = np.delete(arr, 0, 1)
print(arr,'\n','\n',narr)

# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]] 
 
#  [[ 2  3  4]
#  [ 6  7  8]
#  [10 11 12]]

arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
narr = np.delete(arr, [0,1], 1)
print(arr,'\n','\n',narr)

# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]] 
 
#  [[ 3  4]
#  [ 7  8]
#  [11 12]]

arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
narr = np.delete(arr, [0,1], None)  # this is similar to extend  
print(arr,'\n','\n',narr)

 # [ 3  4  5  6  7  8  9 10 11 12]


