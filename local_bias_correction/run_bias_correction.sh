#!/bin/bash
# Run bias correction for local TE and AIS values

for pair in {1..17}
do
	python estimate_local_bias_correction.py $pair
done
