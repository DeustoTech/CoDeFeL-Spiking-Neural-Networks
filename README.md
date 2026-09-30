# Training Spiking Neural Networks

This repository accompanies the paper [*Spiking Neural Networks: a theoretical framework for Universal Approximation and training*](https://arxiv.org/abs/2509.21920).
It provides a compact, reproducible PyTorch implementation of a Spiking Neural Network (SNN) built from 
Leaky Integrate-and-Fire (LIF) neurons with threshold-reset dynamics. The notebook trains the model on a nonlinear 
binary-classification task and illustrates how surrogate gradients enable gradient-based optimization through 
spike and reset events.

![Scheme of the spiking neural network](SNN_circuit.png)

## Motivation

Unlike conventional neural networks, SNNs communicate through discrete events—spikes—rather than continuous activations. 
This event-driven design is biologically inspired and attractive for energy-efficient neuromorphic computation, 
real-time processing, and sparse signal representation.

Training SNNs is challenging because a spike is triggered by a discontinuous threshold operation and followed 
by a reset of the membrane potential. The paper addresses two complementary questions: whether this class of networks can 
approximate general nonlinear functions, and how their hybrid spike dynamics constrain information propagation through layers.

## Approach implemented

The implementation follows the paper's LIF-based architecture:
- An input LIF neuron converts each input sample into a spike train.
- Hidden LIF layers receive spike trains from the preceding layer, transform them into Gaussian-smoothed synaptic currents, emit spikes when their membrane potential reaches a threshold, and reset afterwards.
- A non-resetting output LIF layer produces continuous final-time potentials, which are passed to a trainable readout for binary classification.

The forward pass uses hard thresholding and reset dynamics. During backpropagation, the non-differentiable threshold is 
replaced by a smooth surrogate derivative, allowing PyTorch autograd to optimize the network parameters. 
The supplied experiment trains a one-hidden-layer SNN with eight hidden neurons on the `make_moons` dataset, 
using a 70%/5%/25% training, validation, and test split. It reports accuracy, precision, recall, and F1-score.

The associated paper develops the theoretical framework behind this model. It proves a constructive 
universal-approximation result for LIF SNNs on compact domains, establishes well-posedness of the hybrid dynamics, 
and analyses when spike counts remain stable, decrease across layers, or exceptionally increase through resonance 
or overlapping inputs. 

## Repository contents

| File | Purpose |
| --- | --- |
| `SNN.ipynb` | Complete implementation: LIF cells, surrogate-gradient training, `make_moons` data generation, and test evaluation. |
| `SNN_circuit.png` | Diagram of the SNN architecture used in the paper and this repository. |

## Requirements

Use Python 3.10 or newer with Jupyter and the following packages:

```bash
pip install jupyter torch numpy scikit-learn matplotlib
```

The notebook automatically uses a CUDA GPU when PyTorch detects one; otherwise it runs on CPU.

## Running the simulations

From this directory, launch Jupyter:

```bash
jupyter notebook
```

Open `SNN.ipynb` and run all cells from top to bottom. The notebook generates the `make_moons` dataset, trains the SNN, 
prints validation and test metrics, and displays a ROC curve. To explore a harder classification problem, change 
the `noise` argument in the `make_moons` call; the paper compares clean data with a 20% noise setting.

## Reference

U. Biccari, *Spiking Neural Networks: a theoretical framework for Universal Approximation and training*, 2025.
The manuscript is available on [arXiv:2509.21920](https://arxiv.org/abs/2509.21920).  

## Funding

This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2030 research and innovation 
programme (grant agreement No. 101096251, CoDeFeL).