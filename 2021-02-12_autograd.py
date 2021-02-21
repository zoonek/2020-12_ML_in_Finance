##
## To use gradients in the loss function, compute them with
##   df = torch.autograd( outputs = ..., inputs = ..., retain_graph = True, create_graph = True )[0]
##
## retain_graph=True   Otherwise the computation graph is destroyed -- but we still need it to compute the gradient of the loss wrt the parameters
##                     (You get an explicit error message if you forget this one)
##
## create_graph=True   Otherwise this only returns a number, which would be considered as a constant when included in the loss function
##                     (There is no error message in this case)
##
## This example is a (1-dimensional) polynomial fit with a penalty to encourage the model to be increasing.
##
## More details:
## 
## [1] Certified Monotonic Neural Networks, X. Liu et al. (2020)
##     https://arxiv.org/abs/2011.10219
## 

import torch 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

n = 100
k = 5
x = torch.rand(n, requires_grad=True)  # Input
y = torch.sin( 7 * x.detach() ) + x.detach()      # Desired output
a = torch.rand(k, requires_grad=True)  # Parameters

def f(a,x):      # Simple model (polynomial)
    y = torch.zeros_like(x)
    for i in range(k):
        y += a[i] * x ** i
    return y

optimizer = torch.optim.Adam([a])
N = 10_000
losses = np.zeros((N,3)) * np.nan
pbar = tqdm(range(N))
for i in pbar:
    yhat = f(a,x)    # Output
    df = torch.autograd.grad(outputs = yhat.sum(), inputs = x, retain_graph = True, create_graph = True)[0]
    error = y - yhat
    ## The loss has 2 terms: the reconstruction error,
    ## and a penalty computed from the gradient of f wrt the input x.
    term1 = (error**2).mean()
    term2 = (df.clip(max=0)**2).mean()
    loss = term1 + term2
    losses[i,:] = [ loss.item(), term1.item(), term2.item() ]
    pbar.set_description( f"Loss={loss.item():.5f}" )
    
    ## The optimizer now needs the gradient of the loss wrt the parameters.
    ## The first term is fine, but the second term is ignored -- it has no grad_fn.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

fig, ax = plt.subplots()
ax.plot( 1+np.arange(N), losses[:,0], label = 'Loss', linewidth = 5 )
ax.plot( 1+np.arange(N), losses[:,1], label = 'Error' )
ax.plot( 1+np.arange(N), losses[:,2], label = 'Penalty' )
ax.legend()
ax.set_xscale('log')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.show()

fig, ax = plt.subplots()
xs = np.linspace(0,1,100)
ys = f(a.detach(),torch.tensor(xs))
ax.plot( xs, ys, label='Model' )
ax.scatter( x.detach(), y, color='black', label='Data' )
ax.set_xlabel("x")
ax.set_xlabel("y")
ax.legend()
plt.show()
