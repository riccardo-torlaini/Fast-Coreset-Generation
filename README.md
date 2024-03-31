## Code for the paper "Settling Time vs. Accuracy Tradeoffs for Clustering Big Data"

In this repository you can find the code that recreates the experiments for above-mentioned paper.

### General code outline
The primary function for making compressions is the `make_coreset.py` script. It contains code to create
- uniform compression
- lightweight coresets
- welterweight coresets
- fast coresets
- standard sensitivity-sampling coresets
- BICO coresets (we have simply incorporated the BICO software package as a sub-directory which may need to be independently installed)
- Stream-km++ coresets

The code to do fast-kmeans++ can be found in `fast_kmeans_plusplus.py`. It will create a multi-HST using `multi_hst.py` which in turn uses `hst.py`. The code for standard kmeans++ can be found in `kmeans_plusplus.py` and `kmeans_plusplus_slow.py`. The reason for the two methods is that we are comparing against a python-only implementation of Fast-Kmeans++, so we created versions of kmeans++ that are numpy-based (`kmeans_plusplus.py`) and python-based (`kmeans_plusplus_slow.py`). This was only done for fair speed comparisons. If one wants Fast-Kmeans++ to actually run faster than the numpy version of kmeans++, then one would need to make either a cython or C++ implementation.

### Recreating Experiments
The code to create a coreset can be found in `make_coreset.py`.
The code to make the bar plots is in `experiment_script.py`, which will perform a series of experiments while varying requested parameters. The resulting bar plots can then be made using `read_metrics.py`.
The default setting will run comparisons of Fast Coresets, uniform sampling, lightweight coresets and welterweight coresets on the small datasets with coreset sizes `m=40k` and `m=80k`, for both the k-means and k-median settings.

The remaining experiments can be run using:
- `class_imbalance_experiments.py`: Evaluate performance of accelerated sampling methods vs. class imbalance (Table 6 in the paper).
- `compare_to_sens_sampling.py`: Evaluates sensitivity sampling vs. fast coresets (Table 2 in the paper).
- `log_delta_experiments.py`: Showcase of linear dependency on log Delta (Table 1 in the paper).

Please feel free to reach out to us (the authors of the paper) with any questions you may have.
