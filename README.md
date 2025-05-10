Here’s the full content for your `README.md` file, including the overview and the link to the paper:

---

# NoProp: Training Neural Networks Without Backpropagation

This repository contains an implementation and analysis inspired by the research paper **"NoProp: Training Neural Networks Without Backpropagation or Forward-Propagation"**. The project demonstrates a novel approach to training neural networks by exploring alternative optimization methods, bypassing traditional backpropagation techniques.

For more information about the original paper, visit the [paper on AlphaXiv](https://www.alphaxiv.org/abs/2503.24322).

## Overview

The goal of this project is to implement and evaluate the ideas from the **NoProp** paper using PyTorch. It introduces a diffusion-based training method, where the training process relies on iterative noise reduction and denoising, guided by multi-layer perceptrons (MLPs) and convolutional neural networks (CNNs).

### Key Features

1. **Diffusion Process**:
   - The model is trained using a time-stepped diffusion process, where the noise is gradually reduced layer-by-layer.
   - At each step, the neural network predicts the denoised representation.

2. **CNN for Feature Extraction**:
   - A convolutional neural network is used to extract meaningful features from the data.
   - The CNN architecture includes multiple layers of convolutions, ReLU activations, pooling, and batch normalization for robust feature extraction.

3. **Denoising MLPs**:
   - A series of MLPs, each corresponding to a diffusion step, is used to denoise the representations produced at each stage.

4. **Custom Loss Metric**:
   - A unique loss function is implemented based on the signal-to-noise ratio (SNR) at each diffusion step, ensuring stability and convergence throughout training.

5. **Training and Evaluation**:
   - The model is trained on the MNIST dataset, with hyperparameters such as batch size, learning rate, and epochs configurable.
   - During evaluation, the model achieves high accuracy by reconstructing the denoised representations and classifying them effectively.

6. **Visualization**:
   - Random samples from the test dataset are visualized, showcasing the model's predictions alongside the true labels.

### Implementation Details

- **Dataset**: MNIST is used as the primary dataset for training and evaluation.
- **Framework**: The implementation is built using PyTorch, leveraging its flexibility for custom architectures.

### Results

- **Test Accuracy**: The model achieves a test accuracy of **98.78%**, demonstrating the effectiveness of the NoProp approach.
- **Visualization**: The `predict_and_plot_random_samples` function provides visual insights into the model's predictions on random test samples.

### How to Use

1. **Install Dependencies**:
   - Ensure you have Python installed with PyTorch and other required libraries (e.g., torchvision, matplotlib, numpy).
   - Install dependencies using:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run Training**:
   - To train the model, execute the cells in the Jupyter notebook (`NoProp.ipynb`).

3. **Evaluate Model**:
   - The evaluation section in the notebook calculates the test accuracy and visualizes predictions.

4. **Save and Load Models**:
   - The model can be saved and loaded using the provided functions for reuse and further experimentation.

5. **Visualization**:
   - Plot and analyze random test samples with predictions using the `predict_and_plot_random_samples` function.

### Acknowledgments

This implementation is inspired by the paper **"NoProp: Training Neural Networks Without Backpropagation or Forward-Propagation"**. Special thanks to the authors for introducing this innovative approach to training neural networks.

---
