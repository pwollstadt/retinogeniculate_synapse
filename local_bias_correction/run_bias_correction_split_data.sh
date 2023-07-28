#!/bin/bash
# Run bias correction for local TE and AIS values

for pair in {30..31}
do
	python estimate_local_bias_correction.py $pair
done

for pair in {18..19}
do
	python estimate_local_bias_correction.py $pair
done
