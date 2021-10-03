#!/usr/bin/env python
# coding: utf-8

# # Theory

# In[3]:


system("jupyter" "notebook" "list")


# In[4]:


from IPython.display import Image
Image(filename=r'D:\My paper\Application\photo_jupyter\algorithm_paper1.png', width=900, height=100)


# # Read Data (CSTR)

# In[1]:


from scipy.io import loadmat 
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import adjusted_rand_score as ars
from sklearn.metrics import adjusted_mutual_info_score as amis



#Read Data

file_name = r"D:\My paper\Application\Dataset\cstr.mat"
#file_name=r"C:\Users\saeid\Desktop\Dataset\cstr.mat"

mydata = loadmat(file_name)
#matlab_dict=pd.read_csv(file_name)
#matlab_dict=numpy.loadtxt(file_name)
mydata = mydata.copy()


X = mydata['fea']    # X is my Sparse Data Matrix
print(X[0])
true_labels = mydata['gnd'].flatten().tolist()  #True labels list [1,1,1,..,2,2,2,..,..]

print(X,true_labels)


# # Models

# In[2]:


from NMTFcoclust.Models import NMTFcoclus_OSNMTF 
from NMTFcoclust.Models import NMTFcoclus_NBVD
from NMTFcoclust.Models import NMTFcoclus_DNMF  
#from NMTFcoclust.Models import NMTFcoclus_ODNMF  
from NMTFcoclust.Models import NMTFcoclus_ONM3F
from NMTFcoclust.Models import NMTFcoclus_ONMTF


# In[3]:


model_1 = NMTFcoclus_OSNMTF.OSNMTF(n_row_clusters = 4, n_col_clusters = 4, mu = 0, landa = 0)


# In[4]:


model_1.fit(X)


# In[17]:



model_3 = NMTFcoclus_DNMF.DNMF(n_row_clusters = 4, n_col_clusters = 4)
model_3.fit(X)


# In[13]:


model_4 = NMTFcoclus_ODNMF.ODNMF(n_row_clusters = 4, n_col_clusters = 4)
model_4.fit(X)


# In[14]:


model_2 = NMTFcoclus_NBVD.NBVD(n_row_clusters = 4, n_col_clusters = 4, max_iter= 100)
model_2._fit_single(X, random_state = None)


# In[5]:


model_2.fit(X)


# In[8]:


model_2.row_labels_


# In[9]:


predicted_labels_2 = model_2.row_labels_
print(nmi(true_labels, predicted_labels_2), acc(true_labels, predicted_labels_2), ars(true_labels, predicted_labels_2), amis(true_labels, predicted_labels_2))


# In[11]:


model_5 = NMTFcoclus_ONM3F.ONM3F(n_row_clusters = 4, n_col_clusters = 4)
model_5.fit(X)


# In[15]:


predicted_labels_5 = model_5.row_labels_
print(nmi(true_labels, predicted_labels_5), acc(true_labels, predicted_labels_5), ars(true_labels, predicted_labels_5), amis(true_labels, predicted_labels_5))


# In[12]:


model_6 = NMTFcoclus_ONMTF.ONMTF(n_row_clusters = 4, n_col_clusters = 4)
model_6.fit(X)


# In[14]:


predicted_labels_6 = model_6.row_labels_
print(nmi(true_labels, predicted_labels_6), acc(true_labels, predicted_labels_6), ars(true_labels, predicted_labels_6), amis(true_labels, predicted_labels_6))


# # Evaluation models

# In[ ]:



predicted_labels_1 = model_1.row_labels_ 
predicted_labels_2 = model_2.row_labels_
predicted_labels_3 = model_3.row_labels_
predicted_labels_4 = model_4.row_labels_
predicted_labels_5 = model_5.row_labels_
predicted_labels_6 = model_6.row_labels_

#print(nmi(true_labels,predicted_labels_1))
#print(nmi(true_labels,predicted_labels_2))
#print(nmi(true_labels,predicted_labels_3))
#print(nmi(true_labels,predicted_labels_4)) # Error labels_pred must be 1D: shape is ()
#print(nmi(true_labels,predicted_labels_5))
#print(nmi(true_labels,predicted_labels_6))

print(nmi(true_labels, predicted_labels_1), acc(true_labels, predicted_labels_1), ars(true_labels, predicted_labels_1), amis(true_labels, predicted_labels_1))
print(nmi(true_labels, predicted_labels_2), acc(true_labels, predicted_labels_2), ars(true_labels, predicted_labels_2), amis(true_labels, predicted_labels_2))
print(nmi(true_labels, predicted_labels_3), acc(true_labels, predicted_labels_3), ars(true_labels, predicted_labels_3), amis(true_labels, predicted_labels_3))
print(nmi(true_labels, predicted_labels_4), acc(true_labels, predicted_labels_4), ars(true_labels, predicted_labels_4), amis(true_labels, predicted_labels_4))
print(nmi(true_labels, predicted_labels_5), acc(true_labels, predicted_labels_5), ars(true_labels, predicted_labels_5), amis(true_labels, predicted_labels_5))
print(nmi(true_labels, predicted_labels_6), acc(true_labels, predicted_labels_6), ars(true_labels, predicted_labels_6), amis(true_labels, predicted_labels_6))


# # Visualization

# In[ ]:





# In[ ]:





# # Classic3 Data Set

# In[37]:


#Read Data  #Classic3

file_name=r"D:\My paper\Application\Dataset\Classic3\classic3.mat"
#file_name=r"C:\Users\saeid\Desktop\Dataset\cstr.mat"

mydata = loadmat(file_name)
#matlab_dict=pd.read_csv(file_name)
#matlab_dict=numpy.loadtxt(file_name)

mydata=mydata.copy()


#X=mydata['fea']    # Data Matrix
#true_labels = mydata['gnd'].flatten().tolist()  #True labels list [1,1,1,..,2,2,2,..,..]

#print(X,true_labels)
print(mydata)

print(mydata['labels'])
print(mydata['ts'].shape)
print(mydata['ms'].shape)


# # Ng20 Data Set

# In[25]:


#Read Data  #Ng20
import pandas as pd
file_name=r"D:\My paper\Application\Dataset\ng20.csv"
#file_name=r"C:\Users\saeid\Desktop\Dataset\cstr.mat"

#mydata = loadmat(file_name)
mydict=pd.read_csv(file_name)
#matlab_dict=numpy.loadtxt(file_name)

#mydata=mydata.copy()


#X=mydata['fea']    # Data Matrix
#true_labels = mydata['gnd'].flatten().tolist()  #True labels list [1,1,1,..,2,2,2,..,..]

#print(X,true_labels)
print(mydict)


