def test_mfrow():
    assert mfrow(1) == (1,1)
    assert mfrow(2) == (1,2)
    assert mfrow(3) == (2,2)
    assert mfrow(4) == (2,2)

def test_uniformize():
    a = sorted( uniformize( np.random.uniform(size=5) ) )
    assert np.abs( a - np.array([.1, .3, .5, .7, .9]) ).max() < 1e-10
    a = np.random.uniform(size=6)
    a[3] = np.NaN
    a = uniformize(a)
    assert np.isnan( a[3] )
    a = a[ np.isfinite(a) ]
    a = sorted(a)
    assert np.abs( a - np.array([.1, .3, .5, .7, .9]) ).max() < 1e-12

def test_unique():
    a = np.array([ 1,2,3,2,1,7,2,9 ])
    assert np.all( unique(a) == np.array([ 1,2,3,7,9 ]) )
    a = np.array([ 1, 3, 2, np.NaN, 2, 9, np.NaN, 7 ])
    unique(a)    # There are currently duplicated NaNs -- that was not intentional   

def test_save_load():
    x = np.array([1,2,3])
    save( x, "tmp/test.pickle", "This is a test" )
    y = load( "tmp/test.pickle" )
    assert np.all( x == y )

def test_signal_backtest():
    assert False, "NO TEST YET"   # TODO

def test_dbCreateBaskets():
    n_stocks = 10
    n_dates = 7
    signal = np.random.normal( size=(n_stocks,n_dates) )
    signal = pd.DataFrame( signal )
    r = dbCreateBaskets(signal, 5)
    assert len(r) == 5
    from functools import reduce
    import operator
    assert np.all( reduce(operator.add, r).sum() / 5 == 1 )

def test_dbFractiles_dim_1():
    a = np.random.normal(size=10)
    assert np.all( np.isin( dbFractiles(a), [1,2,3,4,5] ) )
    from collections import Counter
    assert np.all( np.array( list( Counter( dbFractiles(a) ).values() ) ) == 2 )
    a[2] = np.nan
    assert np.isnan( dbFractiles(a)[2] )    
    a = np.random.normal(size=10)
    a = sorted(a)
    assert dbFractiles(a) == [1,1, 2,2, 3,3, 4,4, 5,5]

def test_dbFractiles_dim_2():
    assert False, "NO TEST YET"   # TODO

def test_dbComputePortfolioReturns():
    n_stocks = 26
    n_dates = 7
    ids = LETTERS[:n_stocks]
    dates = pd.date_range(start="2020-01-01", periods=n_dates, freq='M')
    weights = np.random.uniform( size=(n_stocks,n_dates) )
    weights = pd.DataFrame(weights, index=ids, columns=dates)
    weights = weights / weights.sum()
    returns = weights.copy()
    returns[:] = .30 * np.random.normal( size=(n_stocks,n_dates) )
    returns = np.expm1(returns)
    r = dbComputePortfolioReturns(weights, returns)
    r
    ## TODO: Actual test...

def test_dbAnalyzeReturns():
    n_dates = 100
    dates = pd.date_range(start="2020-01-01", periods=n_dates, freq='M')
    returns = .2 * np.random.normal( size = n_dates )
    returns = np.expm1(returns)
    returns = pd.Series(returns, index=dates)
    r = dbAnalyzeReturns(returns, as_df=True)

def test_dbLag_dim_1():
    x = np.array([1,2,3,4,5])
    y = np.array( [ dbLag(x,lag) for lag in range(-6,+7) ] )
    print(y)  # More readable than the tests below
    assert np.all( y[2:7,0] == np.array([5,4,3,2,1]) )
    assert np.all( y[3:8,1] == np.array([5,4,3,2,1]) )
    assert np.all( y[4:9,2] == np.array([5,4,3,2,1]) )
    assert np.all( y[5:10,3] == np.array([5,4,3,2,1]) )
    assert np.all( y[6:11,4] == np.array([5,4,3,2,1]) )
    assert np.all( np.isnan( y[0,:] ) )
    assert np.all( np.isnan( y[1,:] ) )
    assert np.all( np.isnan( y[2,1:] ) )
    assert np.all( np.isnan( y[3,2:] ) )
    assert np.all( np.isnan( y[4,3:] ) )
    assert np.all( np.isnan( y[5,4:] ) )
    assert np.all( np.isnan( y[12,:] ) )
    assert np.all( np.isnan( y[11,:] ) )
    assert np.all( np.isnan( y[10,:-1] ) )
    assert np.all( np.isnan( y[9,:-2] ) )
    assert np.all( np.isnan( y[8,:-3] ) )
    assert np.all( np.isnan( y[7,:-4] ) )

def test_dbLag_dim_2():
    assert False, "NO TEST YET"   # TODO

def test_dbCoalesce():
    ## Dimension 1
    x = np.array([ np.nan,1,2,np.nan,4,np.nan ])
    assert np.all( dbCoalesce(x,0) == np.array([0,1,2,0,4,0]) )
    assert np.all( dbCoalesce( x, x[::-1] )[1:5] == np.array([1,2,2,4]) )
    assert np.all( dbCoalesce( x, x[::-1], np.array([5,4,3,2,1,0]) ) == np.array( [5,1,2,2,4,0] ) )
    ## Dimension 2
    assert False, "NO TEST YET"   #     TODO: 1, 2 or 3 arguments; same size of scalars
    assert False, "NO TEST YET"   #     TODO: vectors or matrices

def test_dbDataFrameToList():
    assert False, "NO TEST YET"   # TODO: Compare the old and the new function

def test_dbDataFrameToMatrix():
    assert False, "NO TEST YET"   # TODO

def test_cumsum_na():
    assert False, "NO TEST YET"   # TODO

def test_replace_last_leading_NaN_with_1():
    assert False, "NO TEST YET"   # TODO

def test_periodicity():
    assert periodicity( pd.date_range(start="2020-01-01", periods=100, freq='D') ) == 'daily'
    assert periodicity( pd.date_range(start="2020-01-01", periods=100, freq='B') ) == 'daily'   #  Business day
    assert periodicity( pd.date_range(start="2020-01-01", periods=100, freq='W') ) == 'weekly'
    assert periodicity( pd.date_range(start="2020-01-01", periods=100, freq='SM') ) == 'fortnightly'
    assert periodicity( pd.date_range(start="2020-01-01", periods=100, freq='M') ) == 'monthly'
    assert periodicity( pd.date_range(start="2020-01-01", periods=100, freq='6M') ) == 'semiannual'
    assert periodicity( pd.date_range(start="2020-01-01", periods=100, freq='A') ) == 'annual'
    assert periodicity( pd.date_range(start="2020-01-01", periods=100, freq='Y') ) == 'annual'
    assert periodicity( pd.date_range(start="1970-01-01", periods=20, freq='2Y') ) == '2y'
    assert periodicity( pd.date_range(start="1970-01-01", periods=20, freq='3Y') ) == '3y'
    assert periodicity( pd.date_range(start="1970-01-01", periods=20, freq='5Y') ) == '5y'
    assert periodicity( pd.date_range(start="1970-01-01", periods=20, freq='10Y') ) == '10y'

 def test_corrplot():
     x = np.random.normal( size=(100,26) )
     x = pd.DataFrame( x, columns = LETTERS )
     C = x.corr()
     corrplot(C, order = True)
     # No actual test
