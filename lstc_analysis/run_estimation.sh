#!/bin/bash
# Estimate local TE and AIS values

for pair in {1..17}
do
	python estimate_te_ais.py $pair
	python estimate_local_values.py $pair
	python estimate_pid.py $pair
done
