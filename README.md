## Code for the paper "Settling Time vs. Accuracy Tradeoffs for Clustering Big Data"

In this repository you can find the code that recreates the experiments for above-mentioned paper.
The code to create a coreset can be found in `make_coreset.py`.
The code to make the bar plots is in `experiment_script.py`, which will perform a series of experiments while varying requested parameters. The resulting bar plots can then be made using `read_metrics.py`.
The default setting will run comparisons of Fast Coresets, uniform sampling, lightweight coresets and welterweight coresets on the small datasets with coreset sizes `m=40k` and `m=80k`, for both the k-means and k-median settings.

The remaining experiments can be run using:
- `class_imbalance_experiments.py`: Evaluate performance of accelerated sampling methods vs. class imbalance (Table 6 in the paper).
- `compare_to_sens_sampling.py`: Evaluates sensitivity sampling vs. fast coresets (Table 2 in the paper).
- `log_delta_experiments.py`: Showcase of linear dependency on log Delta (Table 1 in the paper).

Please feel free to reach out to us (the authors of the paper) with any questions you may have.
