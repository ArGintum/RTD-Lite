# RTD-Lite
 This repository provides official implementation of RTD-Lite topological metric from [paper](https://raw.githubusercontent.com/mlresearch/v258/main/assets/tulchinskii25a/tulchinskii25a.pdf) "RTD-Lite: Scalable Topological Analysis for Comparing Weighted Graphs in Learning Tasks",  which is accepted for presentation at [AISTATS 2025](https://aistats.org/aistats2025/)

*Repository is under construction*

## Abstract

Topological methods for comparing weighted graphs are valuable in various learning tasks but often suffer from computational inefficiency on large datasets. We introduce $\mbox{RTD-Lite}$, a scalable algorithm that efficiently compares topological features, specifically connectivity or cluster structures at arbitrary scales, of two weighted graphs with one-to-one correspondence between vertices. By leveraging minimal spanning trees in auxiliary graphs, RTD-Lite captures topological discrepancies with O($n^2$) time and memory complexity. This efficiency enables its application in tasks like dimensionality reduction and neural network training. Experiments on synthetic and real-world datasets demonstrate that $\mbox{RTD-Lite}$ effectively identifies topological differences while significantly reducing computation time compared to existing methods. Moreover, integrating RTD-Lite into neural network training as a loss function component enhances the preservation of topological structures in learned representations.

## Dependencies

Provided code requires Python 3.9.X version; additional packeges that required for the experiments are listed in `requirements.txt`.

## Usage

Notebooks in the `Experiments` folder contain code for reproducing main experiments from our paper:

- MNIST_UMAP.ipynb -  comparison MNIST low-dimensional representation by UMAP ('Comparing representations from UMAP' section)
- FMNIST+RTD_Lite -  optimization of RTD-Lite (example for FMNIST dataset)
- Rings.ipynd -  experiments on Rings dataset
- Cluster.ipynb - experiments on Clusters dataset


## Cite us

Please use the following BibTeX code to cite [our paper](https://raw.githubusercontent.com/mlresearch/v258/main/assets/tulchinskii25a/tulchinskii25a.pdf):

```
@InProceedings{tulchinskii2025rtdlite,
    title     = {RTD-Lite: Scalable Topological Analysis for Comparing Weighted Graphs in Learning Tasks},
    author    = {Tulchinskii, Eduard and Voronkova, Daria and Trofimov, Ilya and Burnaev, Evgeny and Barannikov, Serguei},
    booktitle = {Proceedings of The 28th International Conference on Artificial Intelligence and Statistics},
    pages     = {3826--3834},
    year      = {2025},
    volume    = {258},
    series    = {Proceedings of Machine Learning Research},
    month     = {03--05 May},
    publisher = {PMLR},
    url       = {https://proceedings.mlr.press/v258/tulchinskii25a.html}
}
```
