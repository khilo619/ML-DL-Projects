# Breast Cancer Classification using Machine Learning

This project applies various machine learning algorithms to classify breast cancer tumors as benign or malignant using the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29). The notebook demonstrates data preprocessing, exploratory data analysis, model training, hyperparameter tuning, and evaluation using both Support Vector Machines (SVM) and Neural Networks (MLPClassifier with different optimizers).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Key Steps](#key-steps)
- [Results](#results)
- [References](#references)

---

## Project Overview

The goal of this project is to build and compare machine learning models for breast cancer diagnosis. The workflow includes:

- Data loading and cleaning
- Exploratory data analysis and visualization
- Feature encoding and scaling
- Model training with SVM and Neural Networks (MLPClassifier)
- Hyperparameter tuning using GridSearchCV
- Model evaluation using accuracy, confusion matrix, and classification report

---

## Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- **File:** `breast cancer.csv`
- **Features:** 30 numeric features computed from digitized images of fine needle aspirate (FNA) of breast mass
- **Target:** Diagnosis (`M` = malignant, `B` = benign)

---

## Project Structure

```
Assignment-3/
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
    cd Assignment-3
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
  Remove unnecessary columns, encode categorical variables, handle missing values, and scale features.

- **Exploratory Data Analysis:**  
  Visualize class distribution, feature distributions, and correlations.

- **Model Training:**  
  - **SVM:** Hyperparameter tuning with GridSearchCV.
  - **Neural Network (MLPClassifier):** Tested with Adam, SGD, and LBFGS optimizers.

- **Evaluation:**  
  Accuracy, confusion matrix, and classification report for each model.

---

## Results

- All models achieved high accuracy on the test set.
- The best neural network configuration (LBFGS optimizer) slightly outperformed SVM in this dataset.
- Visualizations and detailed reports are available in the notebook.

---

## References

- [UCI ML Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [MLPClassifier User Guide](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

---

**Author:**
