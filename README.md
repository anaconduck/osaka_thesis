# Master's Thesis: Multimodal Deep Learning for Alzheimer's Disease Classification

## Overview
This repository contains the source code for a novel **Multimodal Deep Learning Architecture** designed to classify the progression of Alzheimer's Disease. The model integrates structural MRI images and Genetic data (SNP) using an advanced intermediate fusion strategy. 

The primary objectives of the classification tasks are split into two distinct, clinical gold-standard pipelines:
1. **AD vs NC**: Differentiating Alzheimer's Disease (AD) patients from Normal Controls (NC).
2. **pMCI vs sMCI**: Differentiating Progressive Mild Cognitive Impairment (pMCI) from Stable Mild Cognitive Impairment (sMCI).

## Advanced Model Architecture
The project pushes the boundaries of standard multimodal approaches by utilizing:
*   **Unimodal Extractors**: 
    *   **ResNet-18** for extracting high-dimensional spatial features from structural MRI images.
    *   **Transformer Encoder Blocks (x2)** for capturing sequential dependencies and complex relationships in genetic data (SNP).
*   **Feature Fusion (Cross -> Self Attention)**: Rather than simple concatenation, the architecture routes the extracted features through a **Cross-Modal Attention** block, splits them, and processes them through **Self-Attention** blocks before final concatenation.
*   **Auxiliary Losses**: The network architecture prevents gradient vanishing and stabilizes training by computing three parallel losses: Main Output Loss, MRI-specific Loss, and SNP-specific Loss.

## Curriculum Learning Integration
To further improve model convergence on highly complex patient cases (particularly the challenging pMCI vs sMCI task), the training pipeline implements a **Loss-based Curriculum Learning** approach via a Custom Keras Sequence Generator (`CurriculumDataGenerator`):
1. **Warm-Up Phase**: The model trains exclusively on the easiest 30% of patient data (lowest cross-entropy loss) for the first 10 epochs.
2. **Dynamic Scaling**: The model evaluates and sorts the dataset by prediction loss at the end of each epoch, gradually introducing "Medium" and "Hard" patient cases as training progresses.

## Usage
Each task directory (`ad_vs_nc` and `pmci_vs_smci`) acts independently. You can run the advanced multimodal pipeline directly via the terminal:

```bash
# Example: Training the advanced fusion model for pMCI vs sMCI
python src/models/pmci_vs_smci/train_advanced_modalities.py
```

For a visual demonstration of how Curriculum Learning splits patients into *Easy, Medium*, and *Hard* categories during the warm-up epochs, please refer to the `Curriculum_Learning_Simulation.ipynb` notebook.

---
*This research is conducted as part of an ongoing Master's Thesis under the guidance of SANKEN Lab, Osaka University.*
