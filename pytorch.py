#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from torch.autograd import Variable
import json


# In[2]:


def json2edgelist(x):
    edges=[]
    for i in x.keys():
        for j in x[i].keys():
            edges.append([i,j,x[i][j]])
    edges2=np.array(edges)
    m=edges2[:,2].astype(int)
    m=m/np.sum(m)
    return(edges2[:,:2],m)
            


# In[3]:


d=json.loads(open("./pr/data/Gy.json","rb").read())
edges,m=json2edgelist(d)


# In[4]:


nodes=np.unique(edges[:,:2].flatten())
node2i={wq:qw for qw,wq in enumerate(nodes)}


# In[5]:


# Creating the graph
lambd_t = Variable(torch.rand(len(nodes)), requires_grad = True).float()
lambd1_t = Variable(torch.tensor(1.0), requires_grad = True).float()
muv_=Variable(torch.tensor(m)).float()

def exp(lambd1_t,lambd_t):
    lambd_t_i= torch.index_select(lambd_t, 0, Variable(torch.tensor([node2i[i[0]] for i in edges])))
    lambd_t_j= torch.index_select(lambd_t, 0, Variable(torch.tensor([node2i[i[1]] for i in edges])))
    puv_t=muv_/(lambd1_t+lambd_t_j-lambd_t_i)
    s=torch.sum(-muv_*torch.log(puv_t)/np.log(2))+(torch.sum(puv_t)-1.0)*lambd1_t+torch.sum((lambd_t_j-lambd_t_i)*puv_t)
    return(s)

def exp2(lambd1_t,lambd_t):
    lambd_t_i= torch.index_select(lambd_t, 0, Variable(torch.tensor([node2i[i[0]] for i in edges])))
    lambd_t_j= torch.index_select(lambd_t, 0, Variable(torch.tensor([node2i[i[1]] for i in edges])))
    puv_t=muv_/(lambd1_t+lambd_t_j-lambd_t_i)
    s=torch.sum(-muv_*torch.log(puv_t)/np.log(2))+(torch.sum(puv_t)-1.0).pow(2)
    return(torch.sum((lambd_t_i-lambd_t_j).pow(2)))


# In[7]:


loss=exp(lambd1_t,lambd_t)
for i in range(900):
    learning_rate=0.1
    
    opt = torch.optim.Adam(params=[lambd_t,lambd1_t])
    
    loss.backward(retain_graph=True)
    opt.step()
    loss2=exp(lambd1_t,lambd_t)
    print(loss2,lambd_t[:5],lambd1_t)
    #print(loss,lambd1_t, lambd_t.grad[:5],lambd1_t.grad,lambd_t[:5])
    '''with torch.no_grad():
        #lambd1_t -= learning_rate * lambd1_t.grad
        lambd_t -= learning_rate * lambd_t.grad
    #lambd1_t.grad.zero_()
    lambd_t.grad.zero_()'''
    opt.zero_grad()


# In[ ]:





# In[ ]:




