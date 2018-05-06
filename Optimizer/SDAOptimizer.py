## Saddle point Dual Averaging Optimizer in Pytorch
## Author: Amol Damare
## SBU ID: 107914028
## contact: adamare@cs.stonybrook.edu
## Reference : Yurii Nesterov. Primal-dual subgradient methods for convex problems. Mathematical Programming, 120(1):221–259, August 2009.


import torch
import torch.nn
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np


import numpy as np
class NestorovSaddlePointOptimizer(optim.Optimizer):
    def __init__(self, params,lr,alpha=0.3):
        defaults = dict(lr=lr)
        
        super(NestorovSaddlePointOptimizer,self).__init__(params,defaults)
        self.grad_d_sum=None
        self.grad_g_sum=None
        self.beta_0=1
        self.beta_1=1
        self.beta=[]
        self.beta.append(self.beta_0)
        self.beta.append(self.beta_1)
        self.alpha=alpha
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        i=1

        beta_cap=np.asarray(self.beta)
        beta_cap=1/beta_cap
        beta_cap=np.sum(beta_cap)
        self.beta.append(beta_cap)
            
        for group in self.param_groups:
            for p in group['params']:
                b=beta_cap/group['lr']
                if(i==1):
                    d=p
                    grad_d=p.grad.data
                    if self.grad_d_sum is not None:
                        grad_d_sum=torch.add(grad_d_sum,grad_d)
                    else:
                        grad_d_sum=grad_d
                    p.data.add_(grad_d_sum/(self.alpha*b))
                    
                    ##Update d
                else:
                    g=d
                    grad_g=p.grad.data
                    if self.grad_g_sum is not None:
                        grad_g_sum=torch.add(grad_g_sum,grad_g)
                    else:
                        grad_g_sum=grad_g
                    
                    p.data.add_(-grad_g_sum/((1-self.alpha)*b))
                    ##Update g
                i+=1
        
        
        return loss
        
        
        
                

