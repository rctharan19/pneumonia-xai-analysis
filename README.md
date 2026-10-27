# Explainable AI for Pneumonia Classification

A comparative study of **Grad-CAM, LIME, and SHAP** for explaining deep-learning predictions on chest X-ray images using the RSNA Pneumonia Detection Challenge dataset.

## Project Overview

Deep-learning models can classify medical images, but their predictions are often difficult to interpret. This project evaluates three Explainable AI (XAI) techniques to investigate how they highlight image regions associated with pneumonia classification.

## Objectives

- Develop a chest X-ray pneumonia classification pipeline using ResNet-18.
- Generate explanations using Grad-CAM, LIME, and SHAP.
- Evaluate explanation localization against available ground-truth bounding boxes.
- Compare XAI methods using quantitative metrics and paired statistical tests.
- Conduct a limited two-seed reproducibility check.

## Methodology

1. DICOM image preprocessing and label preparation.
2. Training and validation of a ResNet-18 classifier.
3. Classification threshold selection using validation data.
4. Generation of Grad-CAM, LIME, and SHAP explanations.
5. Evaluation using IoU, Dice, coverage, and precision.
6. Pairwise Wilcoxon signed-rank tests with Holm multiple-comparison correction.

## Experimental Results

The primary test evaluation achieved:

| Metric | Result |
|---|---:|
| Accuracy | 0.7639 |
| Precision | 0.4862 |
| Sensitivity | 0.8392 |
| Specificity | 0.7420 |
| F1-score | 0.6157 |
| ROC-AUC | 0.8786 |

Among the evaluated explanation methods, SHAP had the highest mean localization scores descriptively. Grad-CAM and SHAP were not significantly different on every metric after multiple-comparison correction. LIME had lower localization scores than the other two methods across the four reported metrics.

## Repository Contents

- `Pneumonia_XAI_Final_Run.ipynb` — experiment notebook.
- `requirements.txt` — Python dependencies.
- `notebooks/` — notebook documentation.

## How to Run

1. Download or clone this repository.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Obtain the RSNA Pneumonia Detection Challenge dataset through its official source.
4. Configure the dataset paths in the notebook.
5. Run the notebook in an environment with sufficient memory and compute resources.

The original dataset is not included in this repository.

## Limitations and Responsible Use

This project is an experimental research study, not a clinically validated diagnostic system. Localization metrics measure overlap with annotated regions and do not, by themselves, establish clinical correctness or explanation faithfulness. Results may vary with data splits, preprocessing, model training, and explanation settings.

## Technologies

Python · PyTorch · ResNet-18 · Grad-CAM · LIME · SHAP · NumPy · Pandas · Scikit-learn · SciPy

## Dataset Attribution

Dataset: [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

Please follow the dataset provider's terms and conditions.

## Results and Visualizations

### ROC Curve

The classifier achieved a test ROC-AUC of 0.8786.

![ROC Curve](images/roc_curve.png)

### Confusion Matrix

Test-set classification results for the ResNet-18 pneumonia classifier.

![Confusion Matrix](images/confusion_matrix.png)

### Explainability Comparison

Qualitative comparison of Grad-CAM, LIME, and SHAP explanations against ground-truth bounding boxes.

![Grad-CAM, LIME and SHAP comparison](images/xai_qualitative_grid.png)

---

**Author:** Tharan R C

