#!/bin/bash
# Generate results and plots for LSTC analysis

python generate_results_correlation.py
python generate_results_isi.py
python generate_results_sta.py
python3 generate_results_tuples.py

python plots_for_paper.py
python plot_classification_relayed.py
