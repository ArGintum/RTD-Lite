# RTD-Lite
 This repository provides official implementation of RTD-Lite topological metric from paper "RTD-Lite: Scalable Topological Analysis for Comparing Weighted Graphs in Learning Tasks" accepted for AISTATS 2025 conference. 


## Abstract

Topological methods for comparing weighted graphs are valuable in various learning tasks but often suffer from computational inefficiency on large datasets. We introduce $\mbox{RTD-Lite}$, a scalable algorithm that efficiently compares topological features, specifically connectivity or cluster structures at arbitrary scales, of two weighted graphs with one-to-one correspondence between vertices. By leveraging minimal spanning trees in auxiliary graphs, RTD-Lite captures topological discrepancies with O($n^2$) time and memory complexity. This efficiency enables its application in tasks like dimensionality reduction and neural network training. Experiments on synthetic and real-world datasets demonstrate that $\mbox{RTD-Lite}$ effectively identifies topological differences while significantly reducing computation time compared to existing methods. Moreover, integrating RTD-Lite into neural network training as a loss function component enhances the preservation of topological structures in learned representations.

## Dependencies

Provided code requires Python 3.9.X version; additional packeges that required for the experiments are listed in `requirements.txt`.

## Usage

Notebook ... provides basic examples of usage of RTD-Lite.

## Cite us

```
to be done after publishing
```
