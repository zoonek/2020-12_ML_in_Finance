##
## Reference strategies
##

from functions import *
from parameters import *

from sklearn.linear_model import Lasso
from sklearn.linear_model import lasso_path
from tqdm import tqdm

filename = "raw/data_ml.csv"
LOG( f"Reading {filename} [20 seconds]" )
d = pd.read_csv(filename)
d['date'] = pd.to_datetime( d['date'] )

predictors = list( signs.keys() )
target = 'R1M_Usd'

LOG( "Training data" )
i = np.array([ str(u) < DATE1 for u in d['date'] ]) 
train = d[i].copy()
x = train[ predictors ]
y = np.log1p(train[ target ])

LOG( "Clean the data" )
i = np.isfinite(y)
x = x[i]
y = y[i]

x = x.fillna(.5)

LOG( "Lasso" )
alphas, coef, _ = lasso_path( X = x, y = y, max_iter = 10_000 )

LOG( "Load the list of matrices" )
dd = load( "data/data_ml.pickle" )

trailing_log_returns = LAG( np.log1p( dd[ 'R1M_Usd' ] ) )
y = trailing_log_returns.copy()
y.fillna(0, inplace=True)

LOG( "Loop: backtests of the lasso models [10 minutes]" )
res = {}
for j in tqdm(range(len(alphas))):
    signal = np.zeros( shape = dd[ list(dd.keys())[0] ].shape )
    for i,predictor in enumerate(predictors):
        signal += coef[i,j] * dd[predictor].fillna(.5)
    signal = np.where( dd['universe'], 1, np.nan ) * signal
    r = signal_backtest(signal, y, date=DATE1)
    r['performance'  ]['alpha'] = alphas[j]
    r['in-sample'    ]['alpha'] = alphas[j]
    r['out-of-sample']['alpha'] = alphas[j]
    r['performance'  ]['period'] = 'all'
    r['in-sample'    ]['period'] = 'in-sample'
    r['out-of-sample']['period'] = 'out-of-sample'    
    res[ f"{alphas[j]} all" ] = r['performance']
    res[ f"{alphas[j]} in"  ] = r['in-sample']
    res[ f"{alphas[j]} out" ] = r['out-of-sample']

    fig, ax = plt.subplots()
    for i in range(6):
        ax.plot( r['dates'], r['prices'].iloc[i,:], color = quintile_colours[i] )
    ax.set_yscale('log')
    ax.set_title( f"alpha={alphas[j]:5.3}")
    plt.show()

res = pd.concat( res.values() )

i1 = ( res['portfolio'] == 'LS' ) & ( res['period'] == 'in-sample' )
i2 = ( res['portfolio'] == 'LS' ) & ( res['period'] == 'out-of-sample' )
i3 = ( res['portfolio'] == 'LS' ) & ( res['period'] == 'all' )
fig, axs = plt.subplots(1,3, figsize=(15,5))
for j,key in enumerate(['Information Ratio', 'CAGR', 'Annualized Volatility']):
    ax = axs[j]
    ax.plot( res[i1]['alpha'], res[i1][key], label = "in-sample" )
    ax.plot( res[i2]['alpha'], res[i2][key], label = "out-of-sample" )
    #ax.plot( res[i3]['alpha'], res[i3][key], label = "all" )
    ax.set_xlabel('Alpha')
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.set_title(key)
    ax.legend()
plt.show()

number_of_predictors = ( coef != 0 ).sum(axis=0)
fig, ax = plt.subplots()
ax.plot( alphas, number_of_predictors )
ax.set_xlabel('Alpha')
ax.set_xscale('log')
ax.invert_xaxis()
ax.set_title('Number of predictors')
plt.show()

############################################################

LOG( "Constrained regression" )
from scipy.optimize import lsq_linear

LOG( "Training data" )
i = np.array([ str(u) < DATE1 for u in d['date'] ]) 
train = d[i].copy()
x = train[ predictors ]
y = np.log1p(train[ target ])

LOG( "Clean the data" )
i = np.isfinite(y)
x = x[i]
y = y[i]

x = x.fillna(.5)

LOG( "Fit the model" )
r = lsq_linear(
    x, y,
    (
        [ 0      if signs[u] > 0 else -np.inf for u in predictors ],
        [ np.inf if signs[u] > 0 else 0       for u in predictors ],
    )
)
w = np.round( 1e6 * r.x )   # It is not supposed to be that sparse: usually, 2/3 of the coefficients are non-zero...

{ p: w[i] for i,p in enumerate(predictors) if w[i] != 0 }

LOG( "Fit the opposite model, in case I got the signs wrong (I did not)" )
r_rev = lsq_linear(
    x, y,
    (
        [ 0      if signs[u] < 0 else -np.inf for u in predictors ],
        [ np.inf if signs[u] < 0 else 0       for u in predictors ],
    )
)
np.round( 1e6 * r_rev.x )

LOG( "Fit a model with the signs of the OLS estimate" )
r_ols = lsq_linear(
    x, y,
    (
        [ 0      if u > 0 else -np.inf for u in coef[:,-1] ],
        [ np.inf if u > 0 else 0       for u in coef[:,-1] ],
    )
)
np.round( 1e3 * r_ols.x )
np.stack( [ r_ols.x, coef[:,-1] ],axis=1 )
fig, ax = plt.subplots()
ax.scatter( r_ols.x, coef[:,-1] )
plt.show()

LOG( "Data for the backtest" )
dd = load( "data/data_ml.pickle" )
trailing_log_returns = LAG( np.log1p( dd[ 'R1M_Usd' ] ) )
y = trailing_log_returns.copy()
y.fillna(0, inplace=True)

signal = np.zeros( shape = dd[ list(dd.keys())[0] ].shape )
for i,predictor in enumerate(predictors):
    signal += r.x[i] * dd[predictor].fillna(.5)
    #signal += r_rev.x[i] * dd[predictor].fillna(.5)
    #signal += r_ols.x[i] * dd[predictor].fillna(.5)
signal = np.where( dd['universe'], 1, np.nan ) * signal
r = signal_backtest(signal, y, date=DATE1)

r['performance']['Information Ratio']

r['out-of-sample']['Information Ratio']

LOG( "TODO: Save the results to a file" )

dates = dd[ predictors[0] ].columns

LOG( "Look at the correlation of the predictors" )

dates = dd[ predictors[0] ].columns
ids   = dd[ predictors[0] ].index
tmp = np.stack(dd.values())
C = np.zeros( shape=(len(dates),len(predictors),len(predictors)) )
for i in range(len(dates)):
    tmp2 = tmp[:,:,i]
    tmp2 = pd.DataFrame( tmp2, index = dd.keys(), columns = ids )
    tmp2 = tmp2.loc[predictors,:]
    C[i,:,:] = tmp2.T.corr().values
C = np.nanmean(C,axis=0)
C = pd.DataFrame(C, index=predictors, columns=predictors)

corrplot(C)
corrplot(C, order=True)

LOG( "MST of the predictors" )
