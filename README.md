# ImageText-GRU: Deep Learning Architecture for Multimodal Classification and Caption Synthesis

## Overview
ImageText-GRU is a novel deep learning architecture designed for multimodal learning tasks, including image-text classification and caption synthesis. Unlike traditional attention-based models, this project leverages Gated Recurrent Units (GRUs) to efficiently process paired image-text data while maintaining strong performance.

## Features
- **Multimodal Classification**: Classifies paired image-text inputs into predefined categories.
- **Caption Synthesis**: Generates descriptive text for image inputs.
- **Cross-Modal Fusion**: Integrates visual and textual features effectively using GRUs.
- **Computational Efficiency**: Offers a lightweight alternative to transformer-based models.
- **Benchmarking**: Evaluated against state-of-the-art models like CLIP and ViT+BERT.

## Repository Contents
- `project_code.ipynb`: Jupyter Notebook containing the implementation of the ImageText-GRU model.
- `README.md`: This documentation file.

## Installation
To run the project, ensure you have the following dependencies installed:

```bash
pip install torch torchvision transformers numpy pandas matplotlib
```

## Usage
1. Clone this repository:
   ```bash
   git clone [https://github.com/niranjan-ellur/ImageText-GRU-Deep-Learning-Architecture-for-Multimodal-Classification-and-Caption-Synthesis.git]
   cd ImageText-GRU
   ```
2. Open `project_code.ipynb` in Jupyter Notebook or Google Colab.
3. Run the notebook to preprocess data, train the model, and evaluate results.

## Model Architecture
- **Image Processing**: Extracts features using a pre-trained Vision Transformer (ViT) model.
- **Text Processing**: Tokenizes text and generates embeddings using a BERT-based encoder.
- **GRU-based Fusion**: Merges textual and visual embeddings through GRU layers for classification and caption generation.
- **Task-Specific Outputs**:
  - A fully connected layer for classification.
  - A caption generation module with an object-action extraction mechanism.

## Evaluation
The model was trained and tested on the MuCR dataset, achieving:
- **Classification Accuracy**: 93.75%
- **ROC AUC Score**: 99.28%
- **Caption Similarity Score**: 0.6291

## Result
![image](https://github.com/user-attachments/assets/f827fd5b-86fb-4ea2-bbf1-3a1564ff43fc)







