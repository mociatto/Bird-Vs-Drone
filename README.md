# Birds vs Drones: Multimodal Vertical Federated Learning with Privacy Analysis

<div align="center">

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A research project investigating systematic metadata extraction from image-only datasets and privacy vulnerabilities in Vertical Federated Learning (VFL) architectures for bird vs drone classification.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Key Features](#key-features)
- [Research Notebooks](#research-notebooks)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Privacy Analysis](#privacy-analysis)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project presents a comprehensive research pipeline that addresses the challenge of leveraging image-only datasets in federated learning environments. The system automatically extracts rich tabular metadata from images using Large Language Models (LLMs) and programmatic computer vision techniques, then implements a **Vertical Federated Learning (VFL)** architecture to train a multimodal classification system for distinguishing birds from drones.

### Research Objectives

1. **Automated Metadata Extraction**: Develop a systematic pipeline to extract structured tabular features from image-only datasets using LLM APIs
2. **Multimodal VFL Implementation**: Design and implement a vertical federated learning architecture with separate image and tabular clients
3. **Privacy Vulnerability Assessment**: Evaluate the VFL system's susceptibility to privacy attacks including Membership Inference Attacks (MIA) and Attribute Inference Attacks (AIA)
4. **Performance Benchmarking**: Compare standalone models vs. federated learning approaches

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Image-Only Dataset                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          Automated Metadata Extraction Pipeline             │
│  • LLM-based Feature Extraction (Google Gemini API)         │
│  • Programmatic CV Features (SAM, Color, Texture)           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Generated Tabular Dataset                      │
│  (Object Detection, Colors, Shapes, Textures, etc.)         │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Image Client   │     │ Tabular Client  │
│  (EfficientNet) │     │  (Dense NN)     │
│   512D Embed    │     │   128D Embed    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌─────────────────────┐
         │    VFL Server       │
         │  (Fusion Layer)     │
         │  256D → Binary      │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Classification     │
         │  (Bird vs Drone)    │
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Privacy Attacks    │
         │  • MIA              │
         │  • AIA              │
         └─────────────────────┘
```

---

## Key Features

### 1. Intelligent Metadata Extraction
- **LLM-Powered Analysis**: Utilizes Google Gemini API for intelligent visual feature extraction
- **Computer Vision Techniques**: Implements Segment Anything Model (SAM) for object segmentation and programmatic feature extraction
- **Rich Feature Set**: Extracts colors, shapes, textures, object counts, spatial distributions, and more

### 2. Multimodal VFL Architecture
- **Dual-Client Design**: Separate neural networks for image and tabular data processing
- **Privacy-Preserving**: Clients only share embeddings, not raw data
- **Attention-Based Fusion**: Intelligent combination of multimodal embeddings at server level
- **Transfer Learning**: Leverages pre-trained EfficientNetB0 for image features

### 3. Comprehensive Privacy Analysis
- **Membership Inference Attacks (MIA)**: Evaluates if an attacker can determine whether a specific sample was in the training set
- **Attribute Inference Attacks (AIA)**: Assesses vulnerability to inferring sensitive attributes from embeddings
- **Quantitative Metrics**: Provides numerical evaluation of privacy leakage in VFL architecture

### 4. Robust Evaluation Framework
- **Multiple Baselines**: Compares VFL performance against standalone image and tabular models
- **Ensemble Methods**: Implements advanced ensemble techniques for tabular classification
- **Uncertainty Quantification**: Analyzes prediction confidence and model uncertainty

---

## Research Notebooks

The research is structured as a series of progressive Jupyter notebooks, each building upon the previous work:

### 1. Metadata Extraction Pipeline
[![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mostafaanoosha/birds-vs-drones-metadata-extraction)

**Notebook**: `scripts/birds-vs-drones-metadata-extraction.ipynb`

This notebook implements the automated feature extraction pipeline that transforms image-only datasets into rich tabular data.

**Key Components**:
- Configuration of Google Gemini API for visual language model inference
- Implementation of Segment Anything Model (SAM) for precise object segmentation
- Programmatic feature extraction (color histograms, texture analysis, shape descriptors)
- Intelligent caching system to optimize API usage
- Data assembly and validation pipeline

**Extracted Features**:
- Object detection and counting
- Color distribution and dominance
- Shape characteristics (aspect ratio, compactness, symmetry)
- Texture features (edge density, smoothness)
- Spatial distribution metrics
- Background analysis

---

### 2. Baseline Image Classification
[![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mostafaanoosha/bird-vs-drone-classification)

**Notebook**: `scripts/bird-vs-drone-classification.ipynb`

Establishes baseline performance using a standalone deep learning image classifier.

**Key Components**:
- Transfer learning with pre-trained EfficientNetB0
- Advanced data augmentation strategies
- Comprehensive training with early stopping and learning rate scheduling
- Detailed evaluation metrics and visualization
- Prediction confidence analysis
- Uncertainty estimation and visualization

**Performance Metrics**:
- Training/validation accuracy curves
- Confusion matrix analysis
- Per-class performance breakdown
- Confidence distribution analysis
- Misclassification analysis

---

### 3. Tabular Data Classification
[![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mostafaanoosha/birds-vs-drones-tabular-classifier)

**Notebook**: `scripts/birds-vs-drones-tabular-classifier.ipynb`

Trains and evaluates machine learning models on the extracted tabular metadata to establish tabular baseline performance.

**Key Components**:
- Advanced feature engineering with interaction features
- Comprehensive model benchmarking (LightGBM, XGBoost, Random Forest, Neural Networks)
- Feature importance analysis
- Cross-validation and hyperparameter tuning
- Advanced ensemble model creation
- Detailed comparative evaluation

**Models Evaluated**:
- LightGBM Classifier
- XGBoost Classifier
- Random Forest Classifier
- Neural Network (Dense)
- Ensemble combinations

---

### 4. VFL Pipeline with Privacy Attack Analysis
[![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mostafaanoosha/vfl-pipeline-bird-vs-drone)

**Notebook**: `scripts/vfl-pipeline-bird-vs-drone.ipynb`

The main research contribution: implements a complete VFL system and evaluates its privacy vulnerabilities.

**Key Components**:

**VFL Architecture**:
- **Image Client**: EfficientNetB0 → Global Average Pooling → 512D embedding
- **Tabular Client**: Dense Network → 128D embedding
- **VFL Server**: Attention-based fusion → 256D → Binary classification
- Synchronized training with gradient aggregation

**Privacy Attack Implementation**:
- **Membership Inference Attack (MIA)**:
  - Shadow model training approach
  - Attack model training on member/non-member embeddings
  - Evaluation of attack success rate
  
- **Attribute Inference Attack (AIA)**:
  - Attribute prediction from embeddings
  - Analysis of what information leaks through shared embeddings
  - Quantification of privacy loss

**Performance Analysis**:
- Training convergence visualization
- Comparison with baseline standalone models
- Privacy-utility trade-off analysis
- Attack success rate quantification

**Results**:
- VFL system achieves **>80% accuracy** on bird vs drone classification
- Privacy vulnerability assessment provides quantitative metrics on information leakage
- Demonstrates the trade-off between model performance and privacy preservation

---

## Methodology

### Phase 1: Data Preparation
1. **Dataset Acquisition**: Birds vs Drones image dataset from Kaggle
2. **Metadata Extraction**: 
   - LLM-based visual analysis using Google Gemini API
   - Programmatic feature extraction using SAM and OpenCV
3. **Feature Engineering**: Creation of derived features and interaction terms
4. **Data Validation**: Quality checks and consistency verification

### Phase 2: Baseline Model Development
1. **Image Classifier**: Transfer learning with EfficientNetB0
2. **Tabular Classifier**: Ensemble of tree-based and neural network models
3. **Performance Benchmarking**: Establish baseline metrics for comparison

### Phase 3: VFL System Implementation
1. **Client Architecture Design**: 
   - Image client with CNN-based feature extraction
   - Tabular client with dense neural network
2. **Server Architecture**: Fusion layer with attention mechanism
3. **Training Protocol**: Synchronized federated learning with privacy-preserving gradient aggregation

### Phase 4: Privacy Vulnerability Assessment
1. **Attack Scenario Design**: Define threat models for MIA and AIA
2. **Attack Implementation**: Train attack models to exploit VFL vulnerabilities
3. **Evaluation**: Quantify privacy leakage through attack success rates
4. **Analysis**: Investigate privacy-utility trade-offs

---

## Installation

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (recommended for VFL training)
- Google Gemini API key (for metadata extraction)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/mociatto/Bird-Vs-Drone.git
cd Bird-Vs-Drone
```

2. **Install dependencies**:
```bash
pip install -q -U google-generativeai segment-anything transformers lightgbm
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn xgboost opencv-python
```

### Required Libraries
- **Deep Learning**: TensorFlow 2.x, Keras
- **Machine Learning**: scikit-learn, LightGBM, XGBoost
- **Computer Vision**: OpenCV, Segment Anything (SAM)
- **LLM Integration**: google-generativeai
- **Data Science**: pandas, numpy, matplotlib, seaborn
- **Utilities**: PIL, tqdm

---

## Usage

### 1. Metadata Extraction
Run the metadata extraction notebook to generate tabular features from images:

```python
# Configure API
GEMINI_API_KEY = "your-api-key-here"
USE_VLM_API = True  # Set to False to skip API calls and use cache
USE_SAM_MODEL = True  # Set to False to skip SAM processing

# Run extraction pipeline
# This will generate feature files in the working directory
```

### 2. Baseline Training
Train baseline models to establish performance benchmarks:

```python
# Image classifier
# Run bird-vs-drone-classification.ipynb

# Tabular classifier
# Run birds-vs-drones-tabular-classifier.ipynb
```

### 3. VFL Training and Privacy Analysis
Execute the complete VFL pipeline with privacy attack evaluation:

```python
# Run vfl-pipeline-bird-vs-drone.ipynb
# This will:
# 1. Train the VFL system
# 2. Evaluate classification performance
# 3. Conduct MIA and AIA attacks
# 4. Generate privacy vulnerability reports
```

---

## Results

### Classification Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Image Baseline (EfficientNetB0) | ~95% | - | - | - |
| Tabular Baseline (Ensemble) | ~75-80% | - | - | - |
| **VFL System (Image + Tabular)** | **>80%** | - | - | - |

### Key Findings

1. **Metadata Quality**: LLM-based extraction produces semantically rich features that complement programmatic CV features
2. **VFL Effectiveness**: The multimodal VFL architecture successfully combines image and tabular modalities while preserving local data privacy
3. **Privacy-Utility Trade-off**: The system achieves strong classification performance but exhibits measurable privacy vulnerabilities
4. **Attack Success**: Privacy attacks (MIA and AIA) demonstrate that VFL architectures can leak sensitive information through shared embeddings

---

## Privacy Analysis

### Threat Model

The research evaluates two primary privacy attacks:

#### 1. Membership Inference Attack (MIA)
- **Goal**: Determine if a specific sample was used in training
- **Method**: Shadow model approach with attack classifier
- **Metric**: Attack accuracy (higher = more privacy leakage)

#### 2. Attribute Inference Attack (AIA)
- **Goal**: Infer sensitive attributes from shared embeddings
- **Method**: Train attribute predictor on embeddings
- **Metric**: Attribute prediction accuracy

### Privacy Vulnerabilities Discovered

The VFL system exhibits privacy leakage through:
1. **Embedding Information Leakage**: Shared embeddings contain recoverable information about original data
2. **Training Set Membership**: Attackers can distinguish training samples with above-random accuracy
3. **Attribute Recovery**: Certain features can be partially reconstructed from embeddings

### Privacy-Utility Trade-off

The research demonstrates a fundamental tension:
- **Higher Embedding Dimensions** → Better classification accuracy but more privacy leakage
- **Lower Embedding Dimensions** → Better privacy but reduced utility
- **Fine-tuning vs. Frozen Features** → Fine-tuning improves accuracy but increases vulnerability

### Mitigation Strategies

Potential defenses (not implemented in this baseline research):
- Differential privacy mechanisms
- Secure multi-party computation
- Homomorphic encryption
- Dimensionality reduction with privacy constraints
- Noise injection in embeddings

---

## Dataset

**Source**: [Birds vs Drones Dataset on Kaggle](https://www.kaggle.com/datasets/...)

### Dataset Structure
```
dataset/
├── train/
│   ├── bird/
│   └── drone/
└── test/
    ├── bird/
    └── drone/
```

### Dataset Statistics
- **Total Images**: ~4,000+ images
- **Classes**: Binary (Bird, Drone)
- **Image Format**: JPEG/PNG
- **Resolution**: Variable (resized to 224x224 for model input)
- **Split**: 80% train, 20% test

### Generated Metadata Features
The extraction pipeline generates ~20-30 tabular features per image including:
- Object detection results (bounding boxes, confidence scores)
- Color features (dominant colors, color histograms, color variance)
- Shape features (aspect ratio, compactness, eccentricity, symmetry)
- Texture features (edge density, contrast, smoothness)
- Spatial features (object centrality, size ratios)
- Background analysis (complexity, color distribution)

---

## Technologies Used

### Deep Learning Frameworks
- **TensorFlow/Keras**: Primary deep learning framework
- **EfficientNetB0**: Pre-trained CNN for image feature extraction
- **Custom Neural Architectures**: VFL client and server networks

### Machine Learning Libraries
- **scikit-learn**: Classical ML algorithms and evaluation
- **LightGBM**: Gradient boosting for tabular data
- **XGBoost**: Alternative gradient boosting implementation

### Computer Vision
- **Segment Anything Model (SAM)**: Advanced object segmentation
- **OpenCV**: Image processing and feature extraction
- **PIL**: Image manipulation

### LLM Integration
- **Google Gemini API**: Visual language model for intelligent feature extraction
- **Custom prompting strategies**: Optimized for structured metadata generation

### Data Science Stack
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Visualization
- **tqdm**: Progress tracking

---

## Contributing

Contributions are welcome! This is a research project and there are many directions for extension:

### Potential Research Extensions
1. **Privacy Preservation**: Implement differential privacy or secure aggregation
2. **Attack Robustness**: Develop defense mechanisms against MIA/AIA
3. **Model Architectures**: Experiment with different fusion strategies
4. **Dataset Expansion**: Apply to other multimodal classification problems
5. **Horizontal Federated Learning**: Compare with HFL approaches
6. **Cross-Silo Federation**: Extend to multi-organization scenarios

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Acknowledgments

- **Dataset**: Birds vs Drones dataset from Kaggle community
- **Pre-trained Models**: EfficientNetB0 from TensorFlow Hub
- **Segment Anything**: Meta AI Research
- **Google Gemini**: Visual language model for metadata extraction
- **Research Community**: Federated learning and privacy-preserving ML research

---

## Related Work

This research builds upon and contributes to several areas:

1. **Vertical Federated Learning**: Privacy-preserving machine learning across feature-partitioned data
2. **Privacy Attacks in ML**: Understanding vulnerabilities in federated systems
3. **Multimodal Learning**: Combining vision and tabular data for improved predictions
4. **Automated Feature Engineering**: LLM-based metadata extraction from unstructured data

---

## Contact

For questions, suggestions, or collaboration opportunities:

- **GitHub Issues**: [Open an issue](https://github.com/mociatto/Bird-Vs-Drone/issues)
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ for privacy-preserving AI research

</div>
