#!/bin/bash
# Estimate local TE and AIS values for split data sets

python split_data.py
for pair in {30..31}
do
	python estimate_te_ais.py $pair
	python estimate_local_values.py $pair
done

for pair in {18..19}
do
	python estimate_te_ais.py $pair
	python estimate_local_values.py $pair
done

# Run bias correction and then generate_results_correlation_split_data.py in this folder.