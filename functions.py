from numba import jit
import sys, time, os, re, math, datetime, pickle, tqdm, numbers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats
from bisect import bisect_left

def LOG(*o):
    print( time.strftime("%Y-%m-%d %H:%M:%S "), *o, file = sys.stderr, sep = "", flush = True )

    
def mfrow(n, aspect = 4/3, width = 29.7, height = 21):
    """
    Shape of a grid of n plots, on an A4 page, so that their aspect ratio be close to 4/3
    """
    best = (1,1)
    best_value = float('inf')
    for nc in range(1,n+1):
        nr = math.ceil(n/nc)
        a = (width/nc) / (height/nr)
        d = abs(a-aspect)
        if d < best_value:
            best_value = d
            best = (nr, nc)
    return best


colours = [
    "#193296", "#0092D0", "#FFA000", "#D70032", "#2D962D", "#0055AA", 
    "#B4D2F0", "#8296AA", "#0018A8", "#8C9FEC", "#82C3FF", "#FFD999", 
    "#FF8DA7", "#9EE29E", "#3399FF", "#113557", "#E6EAEE", "#B6C0C7"
]

quintile_colours = ["#D7191C", "#FDAE61", "#CCCC66", "#ABD9E9", "#2C7BB6", "black"]

LETTERS = np.array( [ u for u in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" ] )


def uniformize(x):
    """
    Transform the input to make it uniformly distributed on [0,1].
    """
    assert type(x) == list or len(x.shape) == 1
    y = x.copy()
    y = scipy.stats.rankdata( y )
    if isinstance( x, pd.Series ):
        i = np.isnan(x.values)
    else:
        i = np.isnan(x)
    y[:] = np.where( i, np.NaN, y )        
    y = (y - .5) / np.nanmax(y, axis=0)
    return y


def unique(xs):
    """
    Remove duplicates, but keep the order.
    TODO: Use np.unique() instead...
    """
    seen = set()
    result = []
    for x in xs:
        if not x in seen:
            result.append(x)
            seen.add(x)
    return result


def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

    
def save( data, filename, message = None ):
    log_file = re.sub('[.].pickle$', '', filename) + '.log'
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write( f"{timestamp} {' '.join(sys.argv)}\n" )
        if message is not None:
            f.write( f"{timestamp} {message}\n" )
    with open(filename, 'wb') as f:
        pickle.dump(data,f)

        
############################################################

##
## The backtesting functions are similar to what we were using at DB
## (but they were in R, had a few more functionalities, and had been more intensively tested).
## In particular, I have often used the same function names...
## TODO: Remove the "db" prefix from all the function names...
##

def signal_backtest( signal, log_returns, date = None ):
    """
    Given a stock-level signal, build a strategy long the top quintile and short the bottom quintile, 
    and compute its performance (wealth, weights, turnover, performance and risk metrics, etc.)
    Inputs
      signal:      DataFrame, one row per stock, one column per date
      log_returns: DataFrame, one row per stock, one column per date, same size as "signal"; these are *trailing* log-returns
      date:        Threshold date for the performance computations: in-sample up to and including "date", out-of-sample afterwards
    Outputs
      weights:       List of id×date DataFrames, with the quintile portfolio weights: 0=bottom, 1, 2, 3, 4=top, long-short
      log-returns:   DataFrame, portfolio returns, one row per portfolio, one column per date
      ratio-returns: DataFrame, portfolio ratio returns
      prices:        DataFrame, portfolio wealth, initial wealth = 1
      performance:   DataFrame, performance metrics (columns), one portfolio per row, whole sample
      in-sample:     DataFrame, performance metrics (columns), one portfolio per row, in-sample
      out-of-sample: DataFrame, performance metrics (columns), one portfolio per row, out-of-sample
    """

    ## Check the arguments
    assert signal.shape == log_returns.shape, (
        f"Non-matching shapes: signal.shape={signal.shape}) ≠ log_returns.shape={log_returns.shape})"
    )
    dates = pd.to_datetime( signal.columns )

    ## Portfolio weights: list of weight matrices
    ws = create_baskets(signal)
    ws.append( ws[-1] - ws[0] )

    ## Turnover: Sum_i abs( w[i,t] - w[i,t] )
    turnover = [ np.abs((w.iloc[:,1:].values - w.iloc[:,:-1].values)).sum(axis=0) for w in ws ]  # Use "values" to prevent Pandas from aligning the dates...
    turnover = np.array(turnover)
    turnover = pd.DataFrame(turnover, columns = signal.columns[1:])

    ## Portfolio returns 
    rs = [ compute_portfolio_returns( w, np.expm1(log_returns) ) for w in ws ]  # Ratio-returns
    rs = [ np.log1p(r) for r in rs ]  # Log-returns

    ## Portfolio prices
    ps = [ np.exp(cumsum_na(r)) for r in rs ]                 # Log-price = cummulated log-returns
    ps = [ replace_last_leading_NaN_with_1(p) for p in ps ]   # "cumsum" is not the exact inverse of "diff" -- it discards the first value, 1: put it back

    result = {
        "dates":         dates,
        "weights":       ws,
        "log-returns":   pd.DataFrame( rs ),
        "ratio-returns": np.expm1( pd.DataFrame( rs ) ),
        "prices":        pd.DataFrame( ps ),
        "turnover":      turnover,
        "date":          date
    }

    ## Compute the performance on all the periods
    periods = { 'performance': dates == dates }
    ids = pd.DataFrame( { "portfolio": [ str(i+1) for i in range(5) ] + [ "LS" ] } )
    if date is not None:
        date = pd.to_datetime(date)
        periods['in-sample'    ] = dates <= date
        periods['out-of-sample'] = dates >  date 
    for period,i in periods.items():
        perf = [ analyze_returns( r[i], as_df = True ) for r in rs ]
        perf = pd.concat(perf).reset_index(drop=True)
        perf = perf.join(pd.DataFrame( turnover.iloc[ :, i[1:] ].median(axis=1), columns = ["Turnover"] ))    ## TODO: WHY DO I USE iloc?
        perf = ids.join(perf)
        result[period] = perf
        
    return result


def create_baskets(x, n=5):
    """
    Input:   Investment signal, one stock per row, one date per column
    Output:  List of weight matrices for the 5 quintile portfolios
    """
    y = fractiles(x,n)
    r = []
    for k in range(1,n+1):
        w = ( y == k ).astype( np.float64 )
        w = w / w.sum( axis=0 )
        w = w.fillna(0)
        r.append( w )
    return r


def fractiles_1(x, n=5):
    """
    Fractiles of a 1-dimensional object (list, numpy array, pandas dataframe).
    The output is a vector (or list, etc.) with elements 1,2,...,n (and np.nan): 
    the numbering does NOT start at 0.
    """
    assert type(x) == list or len(x.shape) == 1
    y = x.copy()
    p = np.linspace(0,100,n+1)           # 0, 1/5, 2/5, 3/5, 4/5, 1
    q = np.nanpercentile(x, p)           # Corresponding quantiles
    q[0], q[n] = q[0] - 1, q[n] + 1      # Make sure that the first (last) value is lower (larger) than the minimum (maximum), to avoid problems with strict inequalities
    z = [ bisect_left(q,u) for u in x ]  # Fractiles (what we want)
    # Put the data back into y, preserving any additional structure (e.g., the index, for a Pandas Series)
    if isinstance(y, pd.Series): y.iloc[:] = z
    else:                        y[:] = z
    # Put back the missing values
    if isinstance( x, list ): y = [ u if u != 0 else np.NaN for u in y ]
    else:                     y[ y == 0 ] = np.NaN
    return y


def fractiles(x, n=5):
    """
    Compute the fractiles of each column of x.
    """
    if type(x) == list or len(x.shape) == 1:
        ## 1-dimensional object
        return fractiles_1(x,n)
    y = x.copy()
    assert len(x.shape) == 2
    is_pandas = isinstance( x, pd.DataFrame )
    for i in range( x.shape[1] ):
        if is_pandas: y.iloc[:,i] = fractiles( x.iloc[:,i], n )
        else:         y     [:,i] = fractiles( x     [:,i], n )
    return y


def compute_portfolio_returns(
    weights,
    trailing_ratio_returns,
):
    """
    Trailing ratio returns of a portfolio.
    Inputs:  weights:               weight matrix, one row per stock, one column per date
             trailing_ratio_returs: stock returns, matrix of the same size
    Returns: vector of trailing ratio returns
    """
    assert weights.shape == trailing_ratio_returns.shape, "Not implemented: different dates, or different rebalancing frequencies"
    trailing_weights = LAG(weights, 1)
    r = trailing_weights * trailing_ratio_returns
    r.fillna(0, inplace=True)
    r = np.sum(r, axis=0)
    r[0] = np.NaN
    return r


def analyze_returns( ratio_returns, as_df = False ):
    """
    Performance metrics, from a time series of returns.
    The Pandas series should be indexed by dates (for annualization),
    but can be irregularly-spaced. 
    Returns: 
    - Total Number of Observations
    - Valid Observations
    - Start Date
    - End Date
    - Frequency
    - Cumulative Return
    - CAGR (Compounded Annualized Growth Rate)
    - Annualized Volatility
    - Information Ratio
    - Skewness (*)
    - Kurtosis (*)
    - Hit Ratio: Proportion of periods with positive returns
    - Maximum Drawdown        
    - Value-at-Risk 95% (*)
    - Expected Shortfall 95% (*)
    (*) Not annualized
    """
    assert isinstance( ratio_returns, pd.Series ), "Expecting a pandas.Series indexed by dates"

    r = ratio_returns.copy()

    result = {}
    result["Total Number of Observations"] = len(r)
    result["Valid Observations"          ] = np.isfinite(r).sum()

    if np.all( np.isnan(r) ):
        return result
    
    ## Remove missing values
    r = r[ np.isfinite(r) ]
    dates = pd.to_datetime( r.index )

    ## Dates
    result["Start Date"] = dates[ 0].to_pydatetime().date()
    result["End Date"]   = dates[-1].to_pydatetime().date()    
    result["Frequency"]  = periodicity(dates)

    ## Returns
    log_return = np.sum( coalesce( np.log1p(r), 0 ) )
    result["Cummulative Return"] = math.expm1( log_return )
    y = ( result["End Date"] - result["Start Date"] ).days / 365.25   # Number of years
    result["CAGR"] = math.expm1( log_return / y )
    
    ## Volatility (allow for irregularly-spaced observations)
    days = [ np.NaN] +  [ ns / 24 / 3600 / 1e9 for ns in np.diff(dates).tolist() ]
    days = np.diff(dates).astype(np.float64) / 24 / 3600 / 1e9
    x = np.log1p( r[1:] ) / np.sqrt( days )
    x = x[ ~ np.isnan(x) ]
    result["Annualized Volatility"] = math.sqrt(365.25) * np.std(x)

    result["Information Ratio"] = np.NaN
    if result["Annualized Volatility"] > 0: 
        result["Information Ratio"] = result["CAGR"] / result["Annualized Volatility"]

    ## Not annualized
    result["Skewness"]  = scipy.stats.skew(r)
    result["Kurtosis"]  = scipy.stats.kurtosis(r)

    result["Hit Ratio"] = np.sum( r > 0 ) / len( r )

    ## Maximum drawdown
    log_price = np.log1p(r).cumsum()
    last_peak = log_price.cummax()
    drawdown = log_price - last_peak    # log-returns, negative
    max_drawdown = drawdown.min()
    result["Maximum Drawdown"] = math.expm1( max_drawdown )  # Ratio-returns, still negative (it would make more sense, for me, to keep log-returns)

    ## VaR, CVaR
    VaR = np.percentile(r, .05)
    result["Value-at-Risk 95%"]     = VaR
    result["Expected Shortfall 95"] = np.mean( r[ r < VaR ] )

    if as_df: 
        result = pd.DataFrame( [result] )[ list(result.keys()) ]   # Keep the same order
    
    return result


def LAG(x, lag=1):
    """
    Shift an id×date matrix (or Pandas DataFrame) or a vector (or Pandas Series).
    Negative lags peer into the future. 
    The computations are straightforward, but we need to distinguish several case: 
    - 1-dimensional vs 2-dimensional data
    - positive vs negative lags
    - Pandas vs Numpy 
    """
    if lag == 0:
        return x.copy()
    if isinstance( x, list ):
        x = np.array(x)
    assert len(x.shape) in [1,2]
    nc = x.shape[-1]   # Number of elements (dim=1) or columns (dim=2)
    y = np.NaN * x
    if abs(lag) >= nc:
        return y
    if lag > 0:
        source = range(0, nc-lag)
        target = range(lag, nc)
    else:
        source = range(-lag, nc)
        target = range(0,nc+lag)    
    is_pandas = isinstance(x, pd.DataFrame) or isinstance( x, pd.Series )
    if len(x.shape) == 2:
        if is_pandas:
            y.values[ :, target ] = x.values[ :, source ]
        else:
            y[ :, target ] = x[ :, source ]
    else:
        if is_pandas:
            y.values[ target ] = x.values[ source ]
        else:
            y[ target ] = x[ source ]
    return y


def coalesce(*args):
    """
    Similar to SQL's COALESCE: 
    replace missing (or infinite) values in the first argument 
    with values in the second (or third, etc.).
    I do not use fillna() because: 
    - It only works with a constant, not an array;
    - It is specific to Pandas, it does not work with Numpy.
    """
    x = args[0].copy()
    is_pandas = isinstance(x, pd.DataFrame) or isinstance( x, pd.Series )
    for y in args[1:]:
        if 'values' in dir(y): a = y.values
        else:                  a = y
        if is_pandas: x.values[:] = np.where( np.isfinite(x.values), x.values, a )
        else:         x           = np.where( np.isfinite(x),        x,        a )
    return x


def data_frame_to_list(d, id_name = "id", date_name = "date"):
    """UNTESTED"""
    assert d.columns[0] == id_name,   f"The first column should be id_name='{id_name}'"
    assert d.columns[1] == date_name, f"The second column should me date_name='{date_name}'"
    ids   = sorted( unique( d[id_name]   ) )
    dates = sorted( unique( d[date_name] ) )
    empty = np.full( (len(ids),len(dates)), np.nan )
    empty = pd.DataFrame( empty, index = ids, columns = dates )
    def join( x, y ):
        z = np.zeros( shape=x.shape, dtype=np.int32 )
        r = {}
        for i,u in enumerate(y):
            r[u] = i
        for i in range(len(x)):
            z[i] = r[x[i]]
        return z
    i = join( d[id_name],   ids )
    j = join( d[date_name], dates )
    r = {}
    for key in d.columns[2:]:
        r[key] = empty.copy()
        r[key].values[i,j] = d[key].values
    return r

def data_frame_to_matrix(d, id_name = "id", date_name = "date"):
    assert len(d.columns) == 3
    return data_frame_to_list(d, id_name, date_name)[ d.columns[2] ]

def cumsum_na(x):
    """
    Cummulated sum (np.cumsum) preserving NaN.
    Contrary to np.nancumsum(), I do not replace the missing values with zero, but keep them as NaN.
    """
    assert isinstance(x, np.ndarray) or isinstance( x, pd.Series )
    i = np.isnan(x)
    y = x.copy()
    y[:] = np.nancumsum(x)
    y[i] = np.NaN
    return y

def replace_last_leading_NaN_with_1(x):
    """
    Replace the NaN before the first non-missing value with 1.
    """
    assert len(x.shape) == 1, "Expecting a vector, not a matrix"
    assert type(x) != np.ndarray, f"Not implemented: type(x)={type(x)}"
    assert isinstance( x, pd.Series )
    y = x.copy()
    i = np.isfinite(x)
    i = np.nonzero( LAG(i, -1).values )[0][0]
    if not np.isfinite( x[i] ): 
        y[i] = 1
    return y

def periodicity(dates):
    p = np.median( [ ns / 1e9 / 3600 / 24 for ns in np.diff(dates).astype('float') ] )  # Median number of days (from nanoseconds)
    periods = {
        "daily":       1,
        "weekly":      7,
        "fortnightly": 14,
        "monthly":     365.25/12,
        "quarterly":   365.25/4,
        "semiannual":  365.25/2,
        "annual":      365.25,
        "2y":          2 * 365.25,
        "3y":          3 * 365.25,
        "5y":          5 * 365.25,
        "10y":         10 * 365.25
    }
    # Closest period, on a logarithmic scale
    i = np.argmin( [ abs( math.log(p) - math.log(v) ) for v in periods.values() ] )
    return list( periods.keys() )[i]

############################################################

def latex(d, align = None, format={}, file=None):
    """
    Convert a Pandas DataFrame into LaTeX.
    Specify the column alignment as align="llrrr".
    The column formatting dictionary is indexed by column names and contains functions 
    to format each value, e.g., latex_signif, latex_decimal or latex_scientific.
    """
    r = []
    if align is None: 
        align = d.shape[1] * 'l'    
    r.append( r"\begin{tabular}{" + align + "}\n" )
    for j in range(d.shape[1]):
        if j > 0:
            r.append( " & " )
        r.append( texify( d.columns[j] ) )   # TODO: We may not always want to texify the headers
    r.append( r' \\' + "\n" )
    r.append( r'\hline' + "\n" )
    for i in range(d.shape[0]):
        for j in range(d.shape[1]): 
            if j > 0:
                r.append( " & " )
            value = d.iloc[i,j]
            if d.columns[j] in format:
                value = format[ d.columns[j] ]( value )
                value = str(value)
            elif isinstance(value, numbers.Number):
                value = f"${value}$"
            else: 
                value = str(value)
                value = texify(value)
            r.append( value )
        r.append( r' \\' + "\n" )
    r.append( r"\end{tabular}" )
    r = ''.join(r)    
    if file is None:
        return r
    with open(file, "w") as f:
        print(r, file=f)

def latex_scientific(u, signif=3):
    """
    Format a number in LaTeX, in scientific notation, 
    with the prescibed number of significant digits
    (this includes the digit before the decimal point).
    """
    if u == 0:
        return "$0$"
    if not np.isfinite(u):
        return r"\textsc{na}"
    if signif == 0: 
        e = round( math.log10( np.abs(u) ) )
        sign = "" if u > 0 else "-"
        return "$" + sign + "10^{" + str(e) + "}$"
    e = math.floor(math.log10(u))
    m = round( u / 10 ** e, signif )
    m = f"{ u / 10 ** e :.{signif-1}f}"
    return "$" + m + r"\cdot10^{" + str(e) + "}$"

def latex_signif(u, signif=3, scipen=0, round_integers=False):
    """
    Format a number in LaTeX, with a given number of significant digits. 
    Add zeroes at the end if needed.
    By default, integers are not rounded: 
    if you ask for 2 significant digits for 1234, it will remain 1234, and not 1200.
    If the scientific notation is shorter, it will be used: 
    the "scipen" argument specifies how much shorter if should be (in characters).
    """
    e = math.floor(math.log10(u))
    digits = signif - e - 1
    if round_integers and digits < 0:
        plain = f"{ round(u, digits) :.0f}"
    else: 
        digits = max(digits,0)
        plain = f"{u:.{digits}f}"
    scientific = latex_scientific(u,signif)
    a = len(plain)
    b = len( scientific.replace("$","").replace("{","").replace("}","").replace("\\cdot","").replace("^","") )
    if a > b + scipen:
        return scientific
    else:
        return plain

def latex_decimal(u, decimals=2):
    """
    Format a number with a prescribed number of decimals (zeroes if necessary).
    """
    return f"{u:.{decimals}f}"
    
def texify(s):
    s = s.replace( '\\', r'\textbackslash' )
    for character in [ '{', '}', '_', '^', '$', '%', '#', '&' ]:
        s = s.replace( character, '\\' + character )
    s = s.replace( r'\textbackslash', r'\textbackslash{}' )
    return s

############################################################

def corrplot(C, ax=None, vmin=-1, vmax=+1, cmap='RdBu', title=None, figsize=(12,12), order=False):
    """
    Plot a correlation matrix
    """
    if order:
        from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
        from scipy.spatial.distance import pdist, squareform
        i = leaves_list( ward( squareform( np.sqrt(1-C), checks=False ) ) )
        C = C.iloc[i,i]
    ax_was_None = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(C, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks( range(C.shape[1]) )
    ax.set_yticks( range(C.shape[0]) )
    ax.set_xticklabels( C.columns, rotation=90 )
    ax.set_yticklabels( C.index )
    if title is not None:
        ax.set_title(title)
    if ax_was_None:
        fig.tight_layout()
        plt.show()

def plot_lasso(coefs, n=20, ax=None):
    assert isinstance( coefs, pd.DataFrame ), f"Expecting a DataFrame of coefficients, with one row per predictor, got a {type(coefs)}"

    ## Reorder the coefficients
    def f(u):
        v = np.argwhere(u != 0)
        v = v.flatten()
        if len(v) == 0:
            return len(u)
        return v[0]
    i = np.apply_along_axis( f, 1, coefs )
    i = np.argsort(i)
    coefs = coefs.iloc[i,:]

    import copy
    cmap = copy.copy(matplotlib.cm.get_cmap("RdBu"))    
    cmap.set_bad('white')    # Plot the zeros as "white", not "grey" (after replacing them with nan)

    ax_was_None = ax is None
    if ax_was_None:
        fig, ax = plt.subplots(figsize=(10,5))
    tmp = coefs.iloc[:n,:].copy()
    m = np.max(np.abs(tmp.values))
    tmp[ tmp == 0 ] = np.nan
    ax.imshow(tmp, vmin=-m, vmax=m, cmap=cmap, aspect='auto', interpolation='none')
    ax.set_yticks(range(tmp.shape[0]))
    ax.set_yticklabels( tmp.index )
    ax.axes.yaxis.set_ticks_position("right")
    ax.spines['left'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.axes.xaxis.set_visible(False)
    if ax_was_None:
        fig.tight_layout()
        plt.show()  
