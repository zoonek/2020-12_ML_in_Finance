notebook: 1.ipynb data
	jupyter nbconvert --to notebook --execute 1.ipynb # LONG: 10 minutes

raw/data_ml.RData: 
	mkdir -p raw results models plots data tmp
	wget -nc -O raw/data_ml.RData https://github.com/shokru/mlfactor.github.io/raw/master/material/data_ml.RData

data: raw/data_ml.csv data/data_ml.pickle

raw/data_ml.csv: raw/data_ml.RData
	Rscript 1_Data.R

data/data_ml.pickle: raw/data_ml.csv
	python 1_Data.py # LONG

