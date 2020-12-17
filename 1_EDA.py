##
## Input:   raw/ml_data.csv                      # Dataframe
## Outputs: data/data_ml.pickle                  # List of matrices
##          results/individual_backtests.pickle  # DataFrame (currently only contains the performance numbers, for the whole period, not the track records)
##          (Many plots, currently not saved)
##

from functions import *
from parameters import *

filename = "raw/data_ml.csv"
LOG( f"Reading {filename} [20 seconds]" )
d = pd.read_csv(filename)

LOG( "Convert the dates" )
d['date'] = pd.to_datetime( d['date'] )

LOG( f"Rows:    {d.shape[0]}" )
LOG( f"Columns: {d.shape[1]}" )

LOG( f"Number of stocks: {len( d['stock_id'].unique() )}" )
LOG( f"Number of dates:  {len( d['date'].unique() )}" )

f = np.diff( sorted( d['date'].unique() ) )
f = [ int(u) / 1e9 / 3600 / 24 for u in f ]
f = np.median(f)
LOG( f"Median time between observations: {int(f)} days" )

LOG( "Number of stocks for each date" )
x = d[['stock_id','date']].groupby("date").count()
fig, ax = plt.subplots()
ax.plot( x.index, x )
ax.set_ylabel("Number of stocks")
m = x.values.max()
ax.set_ylim(-.04*m, 1.04*m)
plt.show()

LOG( "Column types" )
numeric_columns = [ u for u in d.columns if type(d[u][0]) == np.float64 ]
other_columns = set( d.columns ) - set( numeric_columns ) - set([ 'stock_id', 'date' ])
assert len(other_columns) == 0, f"I expect all columns, except 'stock_id' and 'date' to be numeric, but those are not: {', '.join(other_columns)}" 

LOG( "Universe" )
if not "universe" in d.columns:
    LOG( "  No 'universe' column..." )
    d['universe'] = True

LOG( "Distribution of each variable [WHY IS IT SO SLOW?]" )
a,b = mfrow( len(numeric_columns) )
fig, axs = plt.subplots( a, b, figsize=(12,12) )
for i,column in enumerate(numeric_columns[:2]):
    ax = axs.flatten()[i]
    ax.hist( d[column] )
    ax.set_title(column)
plt.show()

LOG( "Coverage of the variables" )
for i, column in enumerate( numeric_columns ):
    tmp = d[ ['stock_id', 'date', column] ].copy()
    tmp['missing'] = ~ np.isfinite( tmp[column] )
    tmp['non-zero'] = np.isfinite( tmp[column] ) & ( tmp[column] != 0 )
    tmp['zero'] = np.isfinite( tmp[column] ) & ( tmp[column] == 0 )
    tmp['total'] = 1
    tmp = tmp[[ 'date', 'non-zero', 'zero', 'missing', 'total' ]]
    tmp = tmp.groupby('date').sum()
    tmp = tmp.reset_index()

    fig, ax = plt.subplots()
    ax.stackplot(
        tmp['date'],
        tmp['non-zero'], tmp['zero'], tmp['missing'],
        colors = colours[:2] + [ 'grey' ],
        labels = ['Non-zero', 'Zero', 'Missing']
    )
    ax.set_ylabel("Number of stocks")
    ax.set_title( column )
    if i == 0:
        ax.legend()
    plt.show()

LOG( "Read data/data_ml.pickle" )
dd = load( "data/data_ml.pickle" )

LOG( "Guess if if have returns or log-returns, if they are in percent or not" )
x = dd['R1M_Usd'].values.flatten()
x = x[ np.isfinite(x) ]
x.min(), x.max() 
x[ x > 1 ]
assert x.min() > -1
assert len( x[ x > 1 ] ) > 10
LOG( "  Ratio returns, not in percent" )

predictors = [ u for u in numeric_columns if u not in [ 'R1M_Usd', 'R3M_Usd', 'R6M_Usd', 'R12M_Usd' ] ]

LOG( "Individual backtests" )
trailing_log_returns = LAG( np.log1p( dd[ 'R1M_Usd' ] ) )
y = trailing_log_returns.copy()
y.fillna(0, inplace=True)
res = {}
for i,signal in enumerate(predictors):
    LOG( f"[{i+1}/{len(predictors)}] {signal}" )
    x = dd[signal].copy()
    x.fillna(.5, inplace=True)  ## Replace missing values with the median (0.5, since the predictors are uniform)

    x = np.where( dd['universe'], 1, np.nan ) * x   # Only invest in stocks in the universe
        
    r = signal_backtest(x, y, date = DATE1)
    
    r['performance']['signal'] = signal
    r['in-sample'  ]['signal'] = signal
    r['performance']['period'] = 'all'
    r['in-sample'  ]['period'] = 'train'
    res[signal + ' train'] = r['in-sample']
    res[signal + ' all'  ] = r['performance']
    
    fig, ax = plt.subplots()
    for i in range(6):
        ax.plot( r['dates'], r['prices'].iloc[i,:], color = quintile_colours[i] )
    ax.set_yscale('log')
    ax.set_title(signal)
    plt.show()
   
res = pd.concat( res.values() )

save( res, "results/individual_backtests.pickle" )

tmp = res[ ( res['portfolio'] == 'LS' ) & ( res['period'] == 'train' ) ][ ['signal','Information Ratio'] ].copy()
tmp['signal'] = "'" + tmp['signal'] + "':"
for i in range(tmp.shape[0]):
    print( f"  {tmp['signal'].values[i]:34} { np.sign( tmp['Information Ratio'].values[i] ) : 4},   # {tmp['Information Ratio'].values[i]:6.2}" )
    
LOG( "Done." )

