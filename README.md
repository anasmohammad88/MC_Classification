# CIFAR-10 Image Classification with Custom Residual CNN

A custom deep learning project for multi-class image classification on the CIFAR-10 dataset using PyTorch.  
The model is built completely from scratch using residual convolutional blocks, Batch Normalization, GELU activations, Dropout2D regularization, and modern training techniques.

The final model achieves **~94% test accuracy** on CIFAR-10 without transfer learning or pretrained backbones.

---

# Features

- Custom residual CNN architecture
- Residual projection shortcuts
- GELU activation functions
- Batch Normalization
- Dropout2D regularization
- Adaptive Average Pooling
- Data augmentation pipeline
- TensorBoard logging
- Learning rate scheduling
- Checkpoint saving
- GPU training support

---

# Dataset

The project uses the CIFAR-10 dataset:

- 60,000 RGB images
- 10 classes
- Image size: 32×32

Classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

# Model Architecture

The architecture consists of:

- Convolutional stem
- Residual convolutional blocks
- Projection shortcuts for channel matching
- Progressive feature expansion:
  - 32 → 64 → 128 → 256 → 512
- Global average pooling
- Fully connected classification head

---

# Training Configuration

## Optimizer

- AdamW

## Loss Function

- CrossEntropyLoss with label smoothing

## Data Augmentation

- RandomCrop
- RandomHorizontalFlip
- Normalization

## Regularization

- Dropout2D
- Weight decay

---

# Results

| Metric | Value |
|---|---|
| Best Test Accuracy | 93.69% |
| Dataset | CIFAR-10 |
| Framework | PyTorch |

---



# Installation

```bash
pip install torch torchvision torchmetrics tensorboard matplotlib
```

---

# Run Training

```bash
python -m cnn.train
```

---

# TensorBoard

```bash
tensorboard --logdir logs
```

---

# Technologies Used

- Python
- PyTorch
- TorchVision
- TorchMetrics
- TensorBoard

# Results
  <img width="1018" height="497" alt="image" src="https://github.com/user-attachments/assets/5435f8a9-6667-41a8-8d8b-2c38ac225333" />
  <img width="1025" height="544" alt="image" src="https://github.com/user-attachments/assets/6f455348-38f2-41f9-82d7-33b724f80608" />

## Training Logs

```text
Epoch: 0, Train Loss: 1.6828, Eval Loss: 1.6260, Train Accuracy: 0.4555, Eval Accuracy: 0.4861, LR: 0.001000
Epoch: 1, Train Loss: 1.3030, Eval Loss: 1.3365, Train Accuracy: 0.6439, Eval Accuracy: 0.6335, LR: 0.001000
Epoch: 2, Train Loss: 1.1453, Eval Loss: 1.1380, Train Accuracy: 0.7198, Eval Accuracy: 0.7218, LR: 0.001000
Epoch: 3, Train Loss: 1.0430, Eval Loss: 1.0230, Train Accuracy: 0.7707, Eval Accuracy: 0.7826, LR: 0.001000
Epoch: 4, Train Loss: 0.9755, Eval Loss: 1.0018, Train Accuracy: 0.8033, Eval Accuracy: 0.7926, LR: 0.001000
Epoch: 5, Train Loss: 0.9245, Eval Loss: 0.9678, Train Accuracy: 0.8266, Eval Accuracy: 0.8014, LR: 0.001000
Epoch: 6, Train Loss: 0.8865, Eval Loss: 0.9323, Train Accuracy: 0.8440, Eval Accuracy: 0.8181, LR: 0.001000
Epoch: 7, Train Loss: 0.8529, Eval Loss: 0.8854, Train Accuracy: 0.8602, Eval Accuracy: 0.8412, LR: 0.001000
Epoch: 8, Train Loss: 0.8264, Eval Loss: 0.8655, Train Accuracy: 0.8704, Eval Accuracy: 0.8526, LR: 0.001000
Epoch: 9, Train Loss: 0.7976, Eval Loss: 0.8882, Train Accuracy: 0.8816, Eval Accuracy: 0.8448, LR: 0.001000
Epoch: 10, Train Loss: 0.7794, Eval Loss: 0.8284, Train Accuracy: 0.8894, Eval Accuracy: 0.8639, LR: 0.001000
Epoch: 11, Train Loss: 0.7580, Eval Loss: 0.8340, Train Accuracy: 0.8999, Eval Accuracy: 0.8664, LR: 0.001000
Epoch: 12, Train Loss: 0.7379, Eval Loss: 0.8109, Train Accuracy: 0.9099, Eval Accuracy: 0.8766, LR: 0.001000
Epoch: 13, Train Loss: 0.7211, Eval Loss: 0.8093, Train Accuracy: 0.9167, Eval Accuracy: 0.8728, LR: 0.001000
Epoch: 14, Train Loss: 0.7069, Eval Loss: 0.8151, Train Accuracy: 0.9213, Eval Accuracy: 0.8655, LR: 0.001000
Epoch: 15, Train Loss: 0.6920, Eval Loss: 0.7819, Train Accuracy: 0.9279, Eval Accuracy: 0.8857, LR: 0.001000
Epoch: 16, Train Loss: 0.6802, Eval Loss: 0.7701, Train Accuracy: 0.9335, Eval Accuracy: 0.8905, LR: 0.001000
Epoch: 17, Train Loss: 0.6677, Eval Loss: 0.7847, Train Accuracy: 0.9396, Eval Accuracy: 0.8860, LR: 0.001000
Epoch: 18, Train Loss: 0.6569, Eval Loss: 0.7738, Train Accuracy: 0.9431, Eval Accuracy: 0.8906, LR: 0.001000
Epoch: 19, Train Loss: 0.6474, Eval Loss: 0.7914, Train Accuracy: 0.9481, Eval Accuracy: 0.8809, LR: 0.001000
Epoch: 20, Train Loss: 0.6357, Eval Loss: 0.7472, Train Accuracy: 0.9533, Eval Accuracy: 0.9008, LR: 0.001000
Epoch: 21, Train Loss: 0.6263, Eval Loss: 0.7531, Train Accuracy: 0.9568, Eval Accuracy: 0.8996, LR: 0.001000
Epoch: 22, Train Loss: 0.6211, Eval Loss: 0.7247, Train Accuracy: 0.9594, Eval Accuracy: 0.9065, LR: 0.001000
Epoch: 23, Train Loss: 0.6092, Eval Loss: 0.7415, Train Accuracy: 0.9643, Eval Accuracy: 0.9044, LR: 0.001000
Epoch: 24, Train Loss: 0.6049, Eval Loss: 0.7284, Train Accuracy: 0.9658, Eval Accuracy: 0.9060, LR: 0.001000
Epoch: 25, Train Loss: 0.5966, Eval Loss: 0.7343, Train Accuracy: 0.9696, Eval Accuracy: 0.9054, LR: 0.001000
Epoch: 26, Train Loss: 0.5920, Eval Loss: 0.7305, Train Accuracy: 0.9717, Eval Accuracy: 0.9082, LR: 0.001000
Epoch: 27, Train Loss: 0.5874, Eval Loss: 0.7202, Train Accuracy: 0.9726, Eval Accuracy: 0.9128, LR: 0.001000
Epoch: 28, Train Loss: 0.5819, Eval Loss: 0.7374, Train Accuracy: 0.9747, Eval Accuracy: 0.9045, LR: 0.001000
Epoch: 29, Train Loss: 0.5788, Eval Loss: 0.7257, Train Accuracy: 0.9767, Eval Accuracy: 0.9093, LR: 0.001000
Epoch: 30, Train Loss: 0.5729, Eval Loss: 0.7146, Train Accuracy: 0.9797, Eval Accuracy: 0.9176, LR: 0.001000
Epoch: 31, Train Loss: 0.5728, Eval Loss: 0.7272, Train Accuracy: 0.9785, Eval Accuracy: 0.9109, LR: 0.001000
Epoch: 32, Train Loss: 0.5680, Eval Loss: 0.7083, Train Accuracy: 0.9801, Eval Accuracy: 0.9175, LR: 0.001000
Epoch: 33, Train Loss: 0.5629, Eval Loss: 0.7299, Train Accuracy: 0.9829, Eval Accuracy: 0.9097, LR: 0.001000
Epoch: 34, Train Loss: 0.5613, Eval Loss: 0.7072, Train Accuracy: 0.9829, Eval Accuracy: 0.9215, LR: 0.001000
Epoch: 35, Train Loss: 0.5586, Eval Loss: 0.7177, Train Accuracy: 0.9839, Eval Accuracy: 0.9158, LR: 0.001000
Epoch: 36, Train Loss: 0.5539, Eval Loss: 0.7151, Train Accuracy: 0.9861, Eval Accuracy: 0.9131, LR: 0.001000
Epoch: 37, Train Loss: 0.5524, Eval Loss: 0.7192, Train Accuracy: 0.9864, Eval Accuracy: 0.9149, LR: 0.001000
Epoch: 38, Train Loss: 0.5497, Eval Loss: 0.7232, Train Accuracy: 0.9875, Eval Accuracy: 0.9116, LR: 0.001000
Epoch: 39, Train Loss: 0.5493, Eval Loss: 0.7235, Train Accuracy: 0.9872, Eval Accuracy: 0.9152, LR: 0.001000
Epoch: 40, Train Loss: 0.5469, Eval Loss: 0.7190, Train Accuracy: 0.9878, Eval Accuracy: 0.9162, LR: 0.001000
Epoch: 41, Train Loss: 0.5493, Eval Loss: 0.7233, Train Accuracy: 0.9870, Eval Accuracy: 0.9144, LR: 0.001000
Epoch: 42, Train Loss: 0.5466, Eval Loss: 0.7109, Train Accuracy: 0.9884, Eval Accuracy: 0.9174, LR: 0.001000
Epoch: 43, Train Loss: 0.5449, Eval Loss: 0.7151, Train Accuracy: 0.9891, Eval Accuracy: 0.9183, LR: 0.001000
Epoch: 44, Train Loss: 0.5418, Eval Loss: 0.7116, Train Accuracy: 0.9898, Eval Accuracy: 0.9164, LR: 0.001000
Epoch: 45, Train Loss: 0.5391, Eval Loss: 0.7023, Train Accuracy: 0.9914, Eval Accuracy: 0.9203, LR: 0.000100
Epoch: 46, Train Loss: 0.5275, Eval Loss: 0.6767, Train Accuracy: 0.9952, Eval Accuracy: 0.9310, LR: 0.000100
Epoch: 47, Train Loss: 0.5225, Eval Loss: 0.6743, Train Accuracy: 0.9969, Eval Accuracy: 0.9319, LR: 0.000100
Epoch: 48, Train Loss: 0.5207, Eval Loss: 0.6728, Train Accuracy: 0.9971, Eval Accuracy: 0.9327, LR: 0.000100
Epoch: 49, Train Loss: 0.5201, Eval Loss: 0.6740, Train Accuracy: 0.9972, Eval Accuracy: 0.9312, LR: 0.000100
Epoch: 50, Train Loss: 0.5187, Eval Loss: 0.6733, Train Accuracy: 0.9979, Eval Accuracy: 0.9315, LR: 0.000100
Epoch: 51, Train Loss: 0.5177, Eval Loss: 0.6730, Train Accuracy: 0.9979, Eval Accuracy: 0.9316, LR: 0.000100
Epoch: 52, Train Loss: 0.5168, Eval Loss: 0.6719, Train Accuracy: 0.9982, Eval Accuracy: 0.9339, LR: 0.000100
Epoch: 53, Train Loss: 0.5169, Eval Loss: 0.6722, Train Accuracy: 0.9982, Eval Accuracy: 0.9327, LR: 0.000100
Epoch: 54, Train Loss: 0.5162, Eval Loss: 0.6715, Train Accuracy: 0.9985, Eval Accuracy: 0.9324, LR: 0.000100
Epoch: 55, Train Loss: 0.5154, Eval Loss: 0.6714, Train Accuracy: 0.9987, Eval Accuracy: 0.9328, LR: 0.000100
Epoch: 56, Train Loss: 0.5152, Eval Loss: 0.6704, Train Accuracy: 0.9986, Eval Accuracy: 0.9332, LR: 0.000100
Epoch: 57, Train Loss: 0.5148, Eval Loss: 0.6690, Train Accuracy: 0.9986, Eval Accuracy: 0.9338, LR: 0.000100
Epoch: 58, Train Loss: 0.5145, Eval Loss: 0.6692, Train Accuracy: 0.9988, Eval Accuracy: 0.9335, LR: 0.000100
Epoch: 59, Train Loss: 0.5140, Eval Loss: 0.6691, Train Accuracy: 0.9989, Eval Accuracy: 0.9342, LR: 0.000100
Epoch: 60, Train Loss: 0.5135, Eval Loss: 0.6687, Train Accuracy: 0.9989, Eval Accuracy: 0.9336, LR: 0.000100
Epoch: 61, Train Loss: 0.5131, Eval Loss: 0.6684, Train Accuracy: 0.9991, Eval Accuracy: 0.9336, LR: 0.000100
Epoch: 62, Train Loss: 0.5132, Eval Loss: 0.6665, Train Accuracy: 0.9991, Eval Accuracy: 0.9340, LR: 0.000100
Epoch: 63, Train Loss: 0.5129, Eval Loss: 0.6674, Train Accuracy: 0.9990, Eval Accuracy: 0.9335, LR: 0.000100
Epoch: 64, Train Loss: 0.5124, Eval Loss: 0.6676, Train Accuracy: 0.9992, Eval Accuracy: 0.9334, LR: 0.000100
Epoch: 65, Train Loss: 0.5122, Eval Loss: 0.6669, Train Accuracy: 0.9991, Eval Accuracy: 0.9345, LR: 0.000100
Epoch: 66, Train Loss: 0.5120, Eval Loss: 0.6666, Train Accuracy: 0.9991, Eval Accuracy: 0.9338, LR: 0.000100
Epoch: 67, Train Loss: 0.5122, Eval Loss: 0.6695, Train Accuracy: 0.9990, Eval Accuracy: 0.9351, LR: 0.000100
Epoch: 68, Train Loss: 0.5117, Eval Loss: 0.6669, Train Accuracy: 0.9992, Eval Accuracy: 0.9348, LR: 0.000100
Epoch: 69, Train Loss: 0.5116, Eval Loss: 0.6679, Train Accuracy: 0.9992, Eval Accuracy: 0.9355, LR: 0.000100
Epoch: 70, Train Loss: 0.5113, Eval Loss: 0.6678, Train Accuracy: 0.9993, Eval Accuracy: 0.9352, LR: 0.000100
Epoch: 71, Train Loss: 0.5110, Eval Loss: 0.6666, Train Accuracy: 0.9993, Eval Accuracy: 0.9338, LR: 0.000100
Epoch: 72, Train Loss: 0.5106, Eval Loss: 0.6687, Train Accuracy: 0.9993, Eval Accuracy: 0.9333, LR: 0.000100
Epoch: 73, Train Loss: 0.5107, Eval Loss: 0.6663, Train Accuracy: 0.9996, Eval Accuracy: 0.9345, LR: 0.000100
Epoch: 74, Train Loss: 0.5105, Eval Loss: 0.6697, Train Accuracy: 0.9993, Eval Accuracy: 0.9337, LR: 0.000100
Epoch: 75, Train Loss: 0.5106, Eval Loss: 0.6682, Train Accuracy: 0.9992, Eval Accuracy: 0.9340, LR: 0.000100
Epoch: 76, Train Loss: 0.5103, Eval Loss: 0.6682, Train Accuracy: 0.9996, Eval Accuracy: 0.9349, LR: 0.000100
Epoch: 77, Train Loss: 0.5102, Eval Loss: 0.6670, Train Accuracy: 0.9993, Eval Accuracy: 0.9353, LR: 0.000100
Epoch: 78, Train Loss: 0.5095, Eval Loss: 0.6694, Train Accuracy: 0.9995, Eval Accuracy: 0.9333, LR: 0.000100
Epoch: 79, Train Loss: 0.5097, Eval Loss: 0.6667, Train Accuracy: 0.9994, Eval Accuracy: 0.9351, LR: 0.000100
Epoch: 80, Train Loss: 0.5097, Eval Loss: 0.6665, Train Accuracy: 0.9996, Eval Accuracy: 0.9359, LR: 0.000100
Epoch: 81, Train Loss: 0.5093, Eval Loss: 0.6669, Train Accuracy: 0.9994, Eval Accuracy: 0.9359, LR: 0.000100
Epoch: 82, Train Loss: 0.5094, Eval Loss: 0.6667, Train Accuracy: 0.9995, Eval Accuracy: 0.9357, LR: 0.000100
Epoch: 83, Train Loss: 0.5092, Eval Loss: 0.6666, Train Accuracy: 0.9996, Eval Accuracy: 0.9342, LR: 0.000100
Epoch: 84, Train Loss: 0.5090, Eval Loss: 0.6666, Train Accuracy: 0.9996, Eval Accuracy: 0.9356, LR: 0.000100
Epoch: 85, Train Loss: 0.5089, Eval Loss: 0.6671, Train Accuracy: 0.9995, Eval Accuracy: 0.9351, LR: 0.000100
Epoch: 86, Train Loss: 0.5087, Eval Loss: 0.6664, Train Accuracy: 0.9995, Eval Accuracy: 0.9346, LR: 0.000100
Epoch: 87, Train Loss: 0.5092, Eval Loss: 0.6675, Train Accuracy: 0.9993, Eval Accuracy: 0.9338, LR: 0.000100
Epoch: 88, Train Loss: 0.5088, Eval Loss: 0.6671, Train Accuracy: 0.9993, Eval Accuracy: 0.9351, LR: 0.000100
Epoch: 89, Train Loss: 0.5087, Eval Loss: 0.6658, Train Accuracy: 0.9995, Eval Accuracy: 0.9359, LR: 0.000100
Epoch: 90, Train Loss: 0.5086, Eval Loss: 0.6670, Train Accuracy: 0.9996, Eval Accuracy: 0.9355, LR: 0.000100
Epoch: 91, Train Loss: 0.5083, Eval Loss: 0.6663, Train Accuracy: 0.9998, Eval Accuracy: 0.9362, LR: 0.000100
Epoch: 92, Train Loss: 0.5080, Eval Loss: 0.6654, Train Accuracy: 0.9998, Eval Accuracy: 0.9369, LR: 0.000100
Epoch: 93, Train Loss: 0.5081, Eval Loss: 0.6675, Train Accuracy: 0.9996, Eval Accuracy: 0.9353, LR: 0.000100
Epoch: 94, Train Loss: 0.5079, Eval Loss: 0.6669, Train Accuracy: 0.9996, Eval Accuracy: 0.9350, LR: 0.000100
Epoch: 95, Train Loss: 0.5077, Eval Loss: 0.6670, Train Accuracy: 0.9998, Eval Accuracy: 0.9344, LR: 0.000100
Epoch: 96, Train Loss: 0.5075, Eval Loss: 0.6678, Train Accuracy: 0.9997, Eval Accuracy: 0.9359, LR: 0.000100
Epoch: 97, Train Loss: 0.5079, Eval Loss: 0.6677, Train Accuracy: 0.9995, Eval Accuracy: 0.9343, LR: 0.000100
Epoch: 98, Train Loss: 0.5076, Eval Loss: 0.6687, Train Accuracy: 0.9997, Eval Accuracy: 0.9345, LR: 0.000100
Epoch: 99, Train Loss: 0.5076, Eval Loss: 0.6657, Train Accuracy: 0.9996, Eval Accuracy: 0.9348, LR: 0.000100
...
Best Evaluation Accuracy: 0.9369
```
 
