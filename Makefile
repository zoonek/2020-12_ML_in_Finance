notebooks: data 1_Single_signals.ipynb 2_Linear_Model.ipynb 3_Shallow_Linear.ipynb 4_Information_Ratio.ipynb 4_Information_Ratio_Mini-Batches.ipynb 4_Information_Ratio_Mini-Batches_LS.ipynb 5_Deep_Linear.ipynb 5_Deep_Non-Linear.ipynb 6_Lattice_1.ipynb 6_Lattice_2.ipynb 6_Penalty.ipynb 7_Portfolio_Optimization.ipynb
	jupyter nbconvert --to notebook --execute 1_Single_signals.ipynb
	jupyter nbconvert --to notebook --execute 2_Linear_Model.ipynb
	jupyter nbconvert --to notebook --execute 3_Shallow_Linear.ipynb
	jupyter nbconvert --to notebook --execute 4_Information_Ratio.ipynb
	jupyter nbconvert --to notebook --execute 4_Information_Ratio_Mini-Batches.ipynb
	jupyter nbconvert --to notebook --execute 4_Information_Ratio_Mini-Batches_LS.ipynb
	jupyter nbconvert --to notebook --execute 5_Deep_Linear.ipynb
	jupyter nbconvert --to notebook --execute 5_Deep_Non-Linear.ipynb
	jupyter nbconvert --to notebook --execute 6_Lattice_1.ipynb
	jupyter nbconvert --to notebook --execute 6_Lattice_2.ipynb
	jupyter nbconvert --to notebook --execute 6_Penalty.ipynb
	jupyter nbconvert --to notebook --execute 7_Portfolio_Optimization.ipynb
	jupyter nbconvert --to notebook --execute 99_Results.ipynb


old_notebook: 1.ipynb data
	jupyter nbconvert --to notebook --execute 1.ipynb # LONG: 10 hours?

raw/data_ml.RData: 
	mkdir -p raw results models plots data tmp
	wget -nc -O raw/data_ml.RData https://github.com/shokru/mlfactor.github.io/raw/master/material/data_ml.RData

data: raw/data_ml.csv data/data_ml.pickle

raw/data_ml.csv: raw/data_ml.RData
	Rscript 1_Data.R

data/data_ml.pickle: raw/data_ml.csv
	python 1_Data.py # LONG

clean:
	rm -f 1.nbconvert.ipynb
	rm -f data/* results/* models/* plots/* tmp/*  # keep the "raw" directory
