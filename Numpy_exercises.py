#!/usr/bin/env python
# coding: utf-8

# #### 1. Import the numpy package under the name `np` (★☆☆) 
# (**hint**: import … as …)

# In[1]:


import numpy as np


# #### 2. Create a null vector of size 10 (★☆☆) 
# (**hint**: np.zeros)

# In[2]:


np.zeros(10)


# #### 3.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆) 
# (**hint**: array\[4\])

# In[8]:


X=np.zeros(10)
X[4]=1
X


# #### 4.  Create a vector with values ranging from 10 to 49 (★☆☆) 
# (**hint**: np.arange)

# In[12]:


np.arange(10,50)


# #### 5.  Reverse a vector  (★☆☆) 
# (**hint**: array\[::-1\])

# In[14]:


x=np.arange(10,50)
x[::-1]


# #### 6.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆) 
# (**hint**: reshape)

# In[25]:


x1=np.arange(0,9)
x1.reshape(3,3)


# #### 7. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆) 
# (**hint**: np.nonzero)

# In[2]:


x2=[1,2,0,0,4,0]
x3=list(filter(lambda x : x>0,x2))
x3


# #### 8. Create a 3x3 identity matrix (★☆☆) 
# (**hint**: np.eye)

# In[49]:


#처음봄
#np.eye(n,k=m,dtype=int)
np.eye(3)


# #### 9. Create a 3x3x3 array with random values (★☆☆) 
# (**hint**: np.random.random)

# In[64]:


x5=np.random.random(27)
x5.reshape(3,3,3)


# #### 10. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆) 
# (**hint**: min, max)

# In[75]:


x6=np.random.random(100)
x6.reshape(10,10)
min(x6),max(x6)


# #### 11. Create a random vector of size 30 and find the mean value (★☆☆) 
# (**hint**: mean)

# In[82]:


z=np.random.random(30)
z.mean()


# #### 12. Create a 2d array with 1 on the border and 0 inside (★☆☆) 
# (**hint**: array\[1:-1, 1:-1\])

# In[115]:


z1=np.ones((5,5))
z1[1:-1,1:-1]=0
z1


# #### 13. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆) 
# (**hint**: np.diag)

# In[96]:


np.diag([0,1,2,3,4])


# #### 14. fill it with a checkerboard pattern (★☆☆) 
# (**hint**: array\[::2\])

# In[119]:


z1=np.zeros((8,8))
z1[::2,::2]=1
z1[1::2,1::2]=1
z1
#답봐도 모르겠음


# #### 15. Normalize a 5x5 random matrix (★☆☆) 
# (**hint**: (x - mean) / std)

# In[144]:


x=np.random.random((5,5))
x1=x.mean()
x2=x.std()
(x-x1)/x2


# #### 16. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆) 
# (**hint**: np.dot | @)

# In[147]:


#몰랏음, 매트릭스 콥하는 함수인가봄
x1=np.random.random((5,3))
x2=np.random.random((3,2))
x1.dot(x2)


# #### 17. Extract the integer part of a random array using 5 different methods (★★☆) 
# (**hint**: %, np.floor, np.ceil, astype, np.trunc)

# In[161]:


Z=np.random.uniform(0,10,10)


# #### 18. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆) 
# (**hint**: np.arange)

# In[4]:


x=np.arange(0,5)
X=np.vstack((x,x,x,x,x))
X
C=np.hstack((x,x)) #옆으로 쌓는 것
C
#vstack은 array를 쌓는 건갑다


# #### 19. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆) 
# (**hint**: np.linspace)

# In[197]:


#linspace 함수 초면임
np.linspace(0,1,10)


# #### 20. Create a random vector of size 10 and sort it (★★☆) 
# (**hint**: sort)

# In[169]:


z=np.random.random(10)
z.sort()
z


# #### 21. Create random vector of size 10 and replace the maximum value by 0 (★★☆) 
# (**hint**: argmax)

# In[195]:


z=np.random.random(10)
z[z.argmax()]=0
z
#최대값의 인덱스 프리팅


# #### 22. How to find the closest value (to a given scalar) in a vector? (★★☆) 
# (**hint**: argmin)

# In[11]:


Z = np.arange(100)
v = np.random.uniform(0,100)
np.argmin(np.abs(Z-v))


# #### 23. How to tell if a given 2D array has null columns? (★★☆) 
# (**hint**: any, ~)

# In[12]:


a=np.array([[1,2,3],[0,3,2]])
np.any(a==0)


# #### 24. Find the nearest value from a given value in an array (★★☆) 
# (**hint**: np.abs, argmin, flat)

# In[15]:


Z=np.random.uniform(0,1,10)
z=0.5
Z1=np.abs(Z-z)
Z[Z1.argmin()]


# #### 25. How to get the diagonal of a dot product? (★★★) 
# (**hint**: np.diag)

# In[18]:


A = np.random.uniform(0,1,(5,5))
B = np.random.uniform(0,1,(5,5))
np.diag(A.dot(B))


# #### 26. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★) 
# (**hint**: array\[::4\])

# In[27]:


a=np.array([1,2,3,4,5])
b=np.zeros(17)
b[::4]=a
b


# #### 27. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★) 
# (**hint**: array\[:, :, None\])

# In[58]:


a=np.arange(75)
a=a.reshape(5,5,3)

answer=[]
for i in range(75):
    answer.append(a.flat[i])
b=np.array(answer)
b.resize(3,5,5)
b


# #### 28. How to swap two rows of an array? (★★★) 
# (**hint**: array\[\[\]\] = array\[\[\]\])

# In[65]:


A = np.arange(25).reshape(5,5)
x=A[1].copy()
A[1]=A[2].copy()
A[2]=x
print(A)

