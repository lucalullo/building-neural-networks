# Building Neural Networks

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)
![Fashion-MNIST](https://img.shields.io/badge/Fashion--MNIST-Dataset-2C3E50)
![Neural Networks](https://img.shields.io/badge/Neural%20Networks-From%20Scratch-8A2BE2)

A progressive educational project that shows how to build **neural networks from scratch**, starting from a single neuron for binary classification and evolving step by step toward multiclass classification, hidden layers, backpropagation, mini-batch training and a simple deep neural network.

The project uses the **Fashion-MNIST dataset** as a constant practical laboratory. Keeping the same dataset across the main multiclass versions makes each architectural and training change easier to inspect and compare.

The implementation intentionally avoids ready-made neural-network estimators. Weighted sums, activations, loss functions, analytical gradients, backpropagation, parameter updates and mini-batch training are implemented directly with **NumPy, Pandas and Matplotlib** so that the mechanics remain visible.

Version 5 completes the original five-stage roadmap with a simple deep neural network containing two hidden layers.

## Project roadmap

The project follows one main principle:

> **One version, one major concept.**

Version 1 introduces a **Simple Single Neuron**:

1. reduce Fashion-MNIST to two classes;
2. flatten each image into 784 input features;
3. normalize pixel values;
4. compute a weighted input with weights and bias;
5. apply the sigmoid activation;
6. measure binary cross-entropy loss;
7. compute analytical gradients;
8. update parameters with gradient descent;
9. convert the final output into a binary prediction.

Version 2 adds **Multiclass Classification**:

1. extend the problem to all 10 Fashion-MNIST classes;
2. one-hot encode the targets;
3. replace the single output neuron with 10 output neurons;
4. apply numerically stable softmax;
5. use multiclass cross-entropy;
6. train with full-batch gradient descent;
7. select the final class with `argmax`.

Version 3 adds a **Single Hidden Layer**:

1. introduce a 64-neuron hidden layer;
2. use ReLU activation;
3. initialize weights randomly with a He-style strategy;
4. propagate inputs through multiple layers;
5. compute output gradients;
6. backpropagate gradients through the hidden layer;
7. update both parameterized layers.

Version 4 adds **Mini-Batch Training**:

1. shuffle the training data at the start of each epoch;
2. divide the data into mini-batches of 64 samples;
3. run forward propagation for each mini-batch;
4. run backpropagation for each mini-batch;
5. update parameters after every batch;
6. clip probabilities inside the logarithm for numerical stability.

Version 5 adds a **Simple Deep Neural Network**:

1. introduce a second hidden layer;
2. use the architecture `784 → 128 → 64 → 10`;
3. keep ReLU hidden activations;
4. keep softmax output probabilities;
5. train with mini-batch gradient descent;
6. propagate gradients through both hidden layers;
7. combine the concepts introduced across the previous versions.

Each version preserves the previous implementation as much as possible so that the effect of the new concept can be studied in isolation.

Version 5 completes the original project roadmap.

![Building Neural Networks infographic](infographic.png)

## Current capabilities

The final implementation can:

- load the Fashion-MNIST training and test CSV datasets;
- flatten each 28×28 image into 784 input features;
- normalize pixel values from `0-255` to `0-1`;
- one-hot encode multiclass targets;
- initialize neural-network weights directly;
- compute weighted sums and biases;
- apply sigmoid and ReLU activations;
- compute numerically stable softmax probabilities;
- calculate binary and multiclass cross-entropy;
- perform forward propagation through multiple parameterized layers;
- compute analytical gradients;
- backpropagate gradients through hidden layers;
- train with full-batch gradient descent;
- train with mini-batch gradient descent;
- shuffle training data between epochs;
- update parameters after every mini-batch;
- predict multiclass labels with `argmax`;
- train a simple deep neural network without TensorFlow, PyTorch or scikit-learn models.

The final Version 5 architecture uses:

```text
Input features          → 784
First hidden layer      → 128 ReLU neurons
Second hidden layer     → 64 ReLU neurons
Output layer            → 10 softmax neurons
Training strategy       → mini-batch gradient descent
```

## Current architecture

```text
Fashion-MNIST CSV Data
    ↓
Pixel Features
28 × 28 → 784 values
    ↓
Normalization
0-255 → 0-1
    ↓
One-Hot Targets
10 classes
    ↓
Input Layer
784 features
    ↓
Hidden Layer 1
128 neurons + ReLU
    ↓
Hidden Layer 2
64 neurons + ReLU
    ↓
Output Layer
10 neurons
    ↓
Softmax
class probabilities
    ↓
Multiclass Cross-Entropy
    ↓
Backpropagation
output layer
    ↓
Backpropagation
hidden layer 2
    ↓
Backpropagation
hidden layer 1
    ↓
Mini-Batch Parameter Updates
    ↓
Repeat Across Epochs
    ↓
Argmax
    ↓
Fashion-MNIST Prediction
```

The key transition in Version 5 is that backpropagation no longer passes through only one hidden representation. Gradients are propagated through two hidden layers, allowing the project to demonstrate the mechanics of a deeper network while keeping every calculation explicit.

## Project versions

| Version | Main concept | Status |
|---|---|---|
| Version 1 | Simple Single Neuron | Complete |
| Version 2 | Multiclass Neural Network | Complete |
| Version 3 | Single Hidden Layer Network | Complete |
| Version 4 | Mini-Batch Neural Network | Complete |
| Version 5 | Simple Deep Neural Network | Complete |

## Version 1 - Simple Single Neuron

Version 1 establishes the complete learning loop with a single neuron.

Fashion-MNIST is reduced to a binary classification problem using only:

```text
0 → T-shirt/top
1 → Trouser
```

Each 28×28 image is flattened into **784 input features** and normalized from `0-255` to `0-1`.

The model implements:

- one neuron;
- weighted input `X @ w + b`;
- sigmoid activation;
- binary cross-entropy loss;
- gradient descent;
- a `0.5` classification threshold;
- loss-based stopping with a tolerance.

The weights and bias start at zero and are updated directly from the analytical gradients.

Saved notebook results:

```text
Training loss      → 0.0305
Training accuracy  → 98.97%
Test accuracy      → 99.15%
```

![Version 1](v1-simple-single-neuron/Version%201.png)

Version 1 is intentionally limited to two Fashion-MNIST classes so that the complete learning process remains visible before introducing multiclass classification.

## Version 2 - Multiclass Neural Network

Version 2 extends the model to all **10 Fashion-MNIST classes**.

The single output neuron becomes a linear output layer with 10 neurons. The model introduces:

- one-hot encoded targets;
- 10 output neurons;
- numerically stable softmax;
- multiclass cross-entropy;
- full-batch gradient descent;
- `argmax` class prediction.

There are still no hidden layers, so the model remains a linear classifier in pixel space.

Saved notebook results:

```text
Training loss      → 0.5092
Training accuracy  → 83.13%
Test accuracy      → 83.42%
```

![Version 2](v2-multiclass-neural-network/Version%202.png)

## Version 3 - Single Hidden Layer Network

Version 3 introduces the first non-linear hidden representation.

Architecture:

```text
784 inputs → 64 ReLU neurons → 10 softmax outputs
```

New concepts include:

- a 64-neuron hidden layer;
- ReLU activation;
- He-style random weight initialization;
- forward propagation through multiple layers;
- backpropagation through the output and hidden layers;
- fixed-epoch training.

The implementation explicitly computes every gradient needed to update both layers.

Saved notebook results:

```text
Training loss      → 0.5895
Training accuracy  → 79.03%
Test accuracy      → 79.02%
```

![Version 3](v3-single-hidden-layer-network/Version%203.png)

## Version 4 - Mini-Batch Neural Network

Version 4 keeps the same one-hidden-layer architecture but changes how the model is trained.

Architecture:

```text
784 inputs → 64 ReLU neurons → 10 softmax outputs
```

Training now introduces:

- random shuffling at the start of each epoch;
- mini-batches of 64 samples;
- forward propagation per mini-batch;
- backpropagation per mini-batch;
- parameter updates after every batch;
- clipped probabilities inside the logarithm for numerical stability.

This is a major practical step toward the training loop used by modern neural networks.

Saved notebook results:

```text
Training loss      → 0.0695
Training accuracy  → 95.75%
Test accuracy      → 87.28%
```

![Version 4](v4-mini-batch-neural-network/Version%204.png)

## Version 5 - Simple Deep Neural Network

Version 5 adds a second hidden layer and propagates gradients through the complete network.

Architecture:

```text
784 inputs → 128 ReLU neurons → 64 ReLU neurons → 10 softmax outputs
```

It combines the ideas developed in the earlier versions:

- normalized image inputs;
- one-hot targets;
- He-style initialization;
- ReLU hidden activations;
- softmax output probabilities;
- multiclass cross-entropy;
- mini-batch gradient descent;
- data shuffling;
- forward propagation through three parameterized layers;
- backpropagation through both hidden layers.

Saved notebook results:

```text
Training loss      → 0.0145
Training accuracy  → 99.59%
Test accuracy      → 89.92%
```

![Version 5](v5-simple-deep-neural-network/Version%205.png)

## Results

| Version | Model | Train accuracy | Test accuracy |
|---|---|---:|---:|
| V1 | Single neuron, binary | 98.97% | 99.15% |
| V2 | Linear multiclass network | 83.13% | 83.42% |
| V3 | One hidden layer, full-batch | 79.03% | 79.02% |
| V4 | One hidden layer, mini-batch | 95.75% | 87.28% |
| V5 | Two hidden layers, mini-batch | 99.59% | 89.92% |

The Version 1 result is not directly comparable with Versions 2-5 because it uses only two Fashion-MNIST classes, while the remaining versions solve the full 10-class problem.

The project is intentionally educational rather than benchmark-oriented, so the architectures and hyperparameters are kept simple and the versions are designed primarily to expose one new concept at a time.

## Documentation

The repository includes general documentation in both English and Italian:

- [General Project Report - English](project-report-en.pdf)
- [Relazione Generale del Progetto - Italiano](project-report-it.pdf)

Each version also contains:

- the complete Jupyter notebook;
- an English version report;
- an Italian version report;
- a version-specific infographic.

Version folders:

- [`v1-simple-single-neuron/`](v1-simple-single-neuron/)
- [`v2-multiclass-neural-network/`](v2-multiclass-neural-network/)
- [`v3-single-hidden-layer-network/`](v3-single-hidden-layer-network/)
- [`v4-mini-batch-neural-network/`](v4-mini-batch-neural-network/)
- [`v5-simple-deep-neural-network/`](v5-simple-deep-neural-network/)

## Repository structure

```text
building-neural-networks/
│
├── v1-simple-single-neuron/
│   ├── building-neural-networks.ipynb
│   ├── Relazione Versione 1 - Neurone Singolo Semplice.pdf
│   ├── Report Version 1 - Simple Single Neuron.pdf
│   └── Version 1.png
│
├── v2-multiclass-neural-network/
│   ├── building-neural-networks.ipynb
│   ├── Relazione Versione 2 - Rete Neurale Multiclasse.pdf
│   ├── Report Version 2 - Multiclass Neural Network.pdf
│   └── Version 2.png
│
├── v3-single-hidden-layer-network/
│   ├── building-neural-networks.ipynb
│   ├── Relazione Versione 3 - Rete con un Layer Nascosto.pdf
│   ├── Report Version 3 - Single Hidden Layer Network.pdf
│   └── Version 3.png
│
├── v4-mini-batch-neural-network/
│   ├── building-neural-networks.ipynb
│   ├── Relazione Versione 4 - Rete Neurale Mini-Batch.pdf
│   ├── Report Version 4 - Mini-Batch Neural Network.pdf
│   └── Version 4.png
│
├── v5-simple-deep-neural-network/
│   ├── building-neural-networks.ipynb
│   ├── Relazione Versione 5 - Rete Neurale Profonda Semplice.pdf
│   ├── Report Version 5 - Simple Deep Neural Network.pdf
│   └── Version 5.png
│
├── infographic.png
├── project-report-en.pdf
├── project-report-it.pdf
├── README.md
└── LICENSE
```

## Run locally

### Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- NumPy
- pandas
- Matplotlib
- Fashion-MNIST CSV training and test data

The notebooks were developed for the Kaggle Fashion-MNIST dataset environment and currently read the files from:

```python
PATH = "/kaggle/input/datasets/zalando-research/fashionmnist/"
```

Expected files:

```text
fashion-mnist_train.csv
fashion-mnist_test.csv
```

To run locally, install the required libraries:

```bash
pip install numpy pandas matplotlib jupyter
```

Clone the repository:

```bash
git clone <repository-url>
cd building-neural-networks
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Download the Fashion-MNIST CSV dataset, then update the `PATH` variable in the notebook to point to its local directory.

Open the version you want to study and run the cells in order.

For example, to study the final deep neural network implementation:

```text
v5-simple-deep-neural-network/building-neural-networks.ipynb
```

The dataset itself is not included in this repository.

## Development approach

The project follows one main principle:

> **One version, one major concept.**

Instead of starting with a complete deep-learning framework implementation, each version introduces only the next mechanism required to understand how neural networks learn.

The progression is deliberately cumulative:

```text
single neuron
    ↓
multiclass softmax
    ↓
hidden layer + backpropagation
    ↓
mini-batch training
    ↓
deeper neural network
```

Each completed version remains available as an independent learning resource.

The focus is algorithmic understanding rather than benchmark optimization. Fashion-MNIST provides a consistent practical dataset while the notebooks keep every important learning step explicit and inspectable.

## Roadmap

- [x] Version 1 - Simple Single Neuron
- [x] Version 2 - Multiclass Neural Network
- [x] Version 3 - Single Hidden Layer Network
- [x] Version 4 - Mini-Batch Neural Network
- [x] Version 5 - Simple Deep Neural Network

The original V1-V5 roadmap is complete.

## Current limitations

Version 5 is intentionally small and educational:

- the implementation is written for clarity rather than computational efficiency;
- neural-network layers are implemented directly with matrix operations rather than framework APIs;
- training loops are intentionally explicit;
- hyperparameters are manually selected rather than tuned systematically;
- there is no validation split or cross-validation;
- there is no regularization such as dropout or weight decay;
- there are no convolutional layers, even though Fashion-MNIST is an image dataset;
- there are no adaptive optimizers such as Adam or RMSprop;
- there is no learning-rate scheduling;
- the implementation is not intended to reproduce TensorFlow, PyTorch or another production deep-learning framework.

These limitations are intentional. The project was designed to expose the mechanics of neural-network learning clearly rather than maximize predictive performance or training speed.

## Project status

**Building Neural Networks is complete according to the V1-V5 structure originally planned for the project.**

The repository now contains the full educational path from a single binary neuron to a simple deep neural network trained with mini-batch gradient descent and backpropagation.

Version 1 establishes the basic learning loop with sigmoid activation, binary cross-entropy and analytical gradients.

Versions 2 through 4 progressively generalize the architecture and training process by adding multiclass softmax, a hidden layer with backpropagation and mini-batch gradient descent.

Version 5 closes the roadmap by showing how gradients can be propagated through two hidden layers inside a deeper network.

The project can remain finished in this form. Documentation, compatibility fixes or small maintenance updates may still be made when useful.

At the same time, the repository is intentionally not declared permanently frozen. If a future concept is worth studying with the same **one version, one major concept** philosophy, Building Neural Networks may one day receive additional versions beyond Version 5.

There is currently no required Version 6: any future continuation would be a new extension of an already completed project.

## License

This project is distributed under the [MIT License](LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
