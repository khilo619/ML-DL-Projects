# Breast Cancer Classification with K-Nearest Neighbors

This project demonstrates the application of the K-Nearest Neighbors (KNN) algorithm for classifying breast cancer tumors as benign or malignant using the Breast Cancer Wisconsin (Diagnostic) Dataset. The notebook covers data preprocessing, exploratory data analysis, model selection, hyperparameter tuning, feature selection, dimensionality reduction, and performance evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Key Steps](#key-steps)
- [Results](#results)
- [References](#references)

---

## Overview

The goal of this project is to build a robust KNN classifier for breast cancer diagnosis. The workflow includes:

- Data cleaning and preprocessing
- Exploratory data analysis and visualization
- Feature scaling and selection
- Model training and hyperparameter tuning (K selection)
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- Dimensionality reduction with PCA and visualization in 2D and 3D

---

## Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- **File:** `breast cancer.csv`
- **Features:** 30 numeric features from digitized images of fine needle aspirate (FNA) of breast mass
- **Target:** Diagnosis (`M` = malignant, `B` = benign, encoded as 1 and 0)

---

## Project Structure

```
KNN-classifier/
│
├── main.ipynb           # Jupyter notebook with all code and analysis
├── breast cancer.csv    # Dataset file (not included in repo, see Dataset section)
└── README.md            # Project documentation
```

---

## Requirements

- Python 3.11+
- Jupyter Notebook
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

**Install requirements:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## How to Run

1. Clone the repository and navigate to the project folder:
    ```bash
    git clone <your-repo-url>
    cd KNN-classifier
    ```

2. Place the `breast cancer.csv` dataset in the project directory.

3. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. Open `main.ipynb` and run the cells sequentially.

---

## Key Steps

- **Data Preprocessing:**  
  Remove unnecessary columns, encode categorical variables, handle missing values and duplicates, and scale features.

- **Exploratory Data Analysis:**  
  Outlier detection, class distribution, and feature correlation analysis.

- **Model Training and Tuning:**  
  - Split data into training, validation, and test sets with stratification.
  - Tune the number of neighbors (K) using validation accuracy and cross-validation.
  - Evaluate model performance on validation and test sets.

- **Feature Selection:**  
  Remove highly correlated features to improve model generalization.

- **Dimensionality Reduction:**  
  Apply PCA for 2D and 3D visualization of the dataset.

- **Evaluation:**  
  Report accuracy, precision, recall, F1-score, and visualize confusion matrices.

---

## Results

- The best K value is selected based on validation and cross-validation accuracy.
- Feature selection further improves model performance and reduces overfitting.
- PCA visualizations provide insights into class separability in lower dimensions.
- The final model achieves high accuracy and balanced precision/recall on the test set.

---

## References

- [UCI ML Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [KNeighborsClassifier User Guide](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
