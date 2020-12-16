import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scipy, scipy.sparse, scipy.sparse.linalg, scipy.optimize
import matplotlib
import matplotlib.pyplot as plt

n = 100   # Stocks
k = 5     # Signals
l = 11    # Dates

x = np.random.normal( size=(n,k,l) )  # Signals
r = np.random.normal( size=(n,l) )    # Trailing log-returns

x = torch.tensor(x, dtype=torch.float32)
r = torch.tensor(r, dtype=torch.float32)

class Test(torch.nn.Module):
    def __init__(self,k):
        super(Test,self).__init__()
        self.linear = torch.nn.Linear(k,1)
    def forward(self,x):
        # x is n×k×l; the linear layer is applied on the last dimension; I want it applied on "k"
        x = x.permute(0, 2, 1)
        y = self.linear(x)   # n×l×1
        y = y[:,:,0]         #  n×l
        p = y.exp()             # Use a softplus instead of an exponential?
        p = p / p.sum(axis=0)   # portolio weights: positive, sum up to 1 for each date
        return p

    
x = np.random.normal( size=(n,k,l) )  # Signals
r = np.random.normal( size=(n,l) )    # Trailing log-returns

x = torch.tensor(x, dtype=torch.float32)
r = torch.tensor(r, dtype=torch.float32)

model = Test(k)
p = model(x)
# Returns of the strategy
ret = ( p[:,:-1] * torch.expm1(r[:,1:]) ).sum(axis=0).log1p()
IR = ret.mean() / ret.std()
log_wealth = F.pad( ret.cumsum(0), (1,0) )
maxDD = ( log_wealth.cummax(0).values - log_wealth ).max()
ret   # BUG: WHY ARE THE RETURNS ALWAYS POSITIVE???

DONE: Compute the returns of the strategy (using the stock returns)
DONE: Compute the information ratio
DONE: Compute the maximum drawdown (it is differentiable)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
for t in range(5000):
    y_pred = model(x)
    loss = criterion(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

beta = list( model.parameters() )[0].detach().numpy().flatten()
beta


Generate sample data, to quickly test the code

Model 1: Linear model score --> weights
Loss 1: information ratio; input = weights + returns
Loss 1bis: information ratio, after transaction costs
Loss 2: maximum drawdown
Loss 3: probability of exceeding x% (how do I make that differentiable?)

Model 2: Normalize the weights so they are positive and sum to 1
Model 3: Add the sign constraints (with a softplus)
Model 4: Add the optimization
Model 5: Add lattice layers


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        ...
    def forward(self,x):
        y = self.score(x)
        w = self.weights(y)
        
        ...
        return y
