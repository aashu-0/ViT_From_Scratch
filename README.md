# Vision Transformer (ViT) Implementation in PyTorch

This repository contains a PyTorch implementation of the Vision Transformer (ViT) as described in the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929).

## Overview

Vision Transformer (ViT) applies the transformer architecture to image classification by splitting an image into fixed-size patches, linearly embedding them, adding position embeddings, and feeding the resulting sequence to a standard Transformer encoder. This implementation follows the original paper and provides a working example on the CIFAR-10 dataset.

## Repository Structure

```
ViT/
├── README.md               # Project information and setup instructions
├── model.py                # ViT model implementation
├── custom_data.py          # Data loading and preprocessing
├── engine.py               # Training and evaluation functions
├── inference.py            # Script for running inference
└── requirements.txt        # Required dependencies
```

## Features

- Complete implementation of Vision Transformer for image classification
- Flexible patch size, embedding dimensions, and transformer configurations
- Training and evaluation pipeline
- Pre-trained model on CIFAR-10 subset (airplane, automobile, bird, cat)
- Inference script for using the trained model

## Installation

1. Clone the repository:
```bash
git clone 
cd ViT
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the ViT model on CIFAR-10:

```bash
python train.py --epochs 5
```

This will:
- Download the CIFAR-10 dataset (if not already downloaded)
- Create a subset with selected classes (airplane, automobile, bird, cat)
- Train the ViT model for the specified number of epochs
- Save the trained model as `vit_cifar10.pth`

## Inference

To run inference using a trained model:

```bash
python inference.py --image your_image.jpg --model vit_cifar10.pth
```

## Model Architecture

The implementation consists of the following key components:

1. **Patch Embedding**: Transforms the input image into patches and projects them to the embedding dimension
2. **Multi-Head Attention**: Self-attention mechanism for capturing relationships between patches
3. **MLP Block**: Feed-forward network applied to each token
4. **Transformer Block**: Combines attention and MLP with layer normalization and residual connections
5. **Classification Head**: Final layer for class prediction

## Colab Notebooks

For quick experimentation, here are Colab notebooks:
- [ViT Training Notebook](https://colab.research.google.com/drive/1dh-eKLrHXK3dWgeuscx-ZSATpCKQeCQ2?usp=sharing)
- [Pre-trained ViT](https://colab.research.google.com/drive/1A3q1R9P8IOF--qtULgrq2u94k3hlxVd1?usp=sharing)


## Acknowledgements

This implementation is based on the [original ViT paper](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.