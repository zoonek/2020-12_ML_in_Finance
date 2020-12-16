##
##  Convert the data-frame with all the data into a list of matrices,
##  one matrix per variable, with one stock per row, one date per column.
##
## Input:  raw/data_ml.csv
## Output: data/data_ml.pickle
##

from functions import *
from parameters import *

filename = "raw/data_ml.csv"
LOG( f"Reading {filename} [20 seconds]" )
d = pd.read_csv(filename)

LOG( "Convert the dates" )
d['date'] = pd.to_datetime( d['date'] )

LOG( "Convert to matrices [VERY LONG]" )
dd = dbDataFrameToList(d, id_name = 'stock_id', date_name = 'date')

dd['universe'].fillna(0, inplace=True)

LOG( "Save to data/data_ml.pickle" )
save( dd, "data/data_ml.pickle" )

LOG( "Done." )
