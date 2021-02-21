from functions import *
import numpy as np
import pandas as pd

def get_data_3(all=False, flip_signs=False, date='2016-01-01', signs=None, verbose=True):
    """
    Data, as tensors.
    Arguments: all: whether ti return all the data or just the training data
    Returns:   x: id×date×signal
               y: id×date, forward ratio returns
               universe: id×date, 0 or 1
    Global variables used: d, target, predictors, DATE1
    """

    assert signs is not None
    
    if verbose:
        LOG( "  Data (data-frame)" )
    filename = "raw/data_ml.csv"
    if verbose: 
        LOG( f"  Reading {filename} [20 seconds]" )
    d = pd.read_csv(filename)
    d['date'] = pd.to_datetime( d['date'] )

    predictors = list( signs.keys() )
    target = 'R1M_Usd'

    if verbose:
        LOG( "  data_frame_to_list" )
    train = data_frame_to_list(d, id_name = 'stock_id', date_name = 'date')


    if all:
        i = np.array( [ True for u in train[ predictors[0] ].columns ] )
    else:
        i = np.array([ str(u) < date for u in train[ predictors[0] ].columns ])
    train = { k: v.T[i].T for k,v in train.items() }
        #x = train[ predictors ]
        #y = np.log1p(train[ target ])
    y = train[target]
    if flip_signs: 
        x = [ signs[k] * train[k] for k in predictors ]
    else:
        x = [ train[k] for k in predictors ]
    y = y.values

    x = np.stack( [ u.values for u in x ] )

    y = np.log1p(y)
    universe = np.where( np.isfinite(y), 1, 0 )
    # x = x.fillna(.5)
    x = np.nan_to_num(x, nan=.5)
    y = np.nan_to_num(y, nan=0)

    n = y.shape[0]   # Number of stocks
    l = y.shape[1]   # Number of dates
    k = x.shape[0]   # Number of predictors
    assert k == len(predictors)
    assert n == x.shape[1]
    assert l == x.shape[2]

    # x is now: signal×id×date

    x = x.transpose([1,2,0])
    assert x.shape == (n,l,k) # id×date×signal
    assert y.shape == (n,l)   # id×date
    assert universe.shape == (n,l)

    if False: 
        x = x[:5,:4,:3]
        y = y[:5,:4]
        universe = universe = universe[:5,:4]

    assert ( ~ np.isfinite(x) ).sum() == 0
    assert ( ~ np.isfinite(y) ).sum() == 0
    
    return x, y, universe
