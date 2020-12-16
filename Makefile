notebook:
	jupyter nbconvert --to notebook --execute 1.ipynb # LONG: 10 minutes

all: data notebook

data:
	mkdir -p raw results models plots data tmp
	wget -nc -O raw/data_ml.RData https://github.com/shokru/mlfactor.github.io/raw/master/material/data_ml.RData
	Rscript 1_Data.R
	python 1_Data.R # LONG

