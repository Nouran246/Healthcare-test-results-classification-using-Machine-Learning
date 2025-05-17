# Healthcare Test Results Classification Using Machine Learning

## Overview

This project aims to classify healthcare test results (e.g., normal vs. abnormal) using a variety of machine learning algorithms. By leveraging patient laboratory test data, we attempt to predict the outcome of these tests using supervised learning approaches. The models developed can assist in early diagnosis and automated screening in the healthcare sector.

We experimented with multiple models including Logistic Regression, Random Forest, SVM, XGBoost, Naive Bayes, Neural Networks, and an ensemble voting classifier.

---

## Dataset and Preprocessing

### Source

The dataset consists of healthcare test results with both numerical and categorical features. Labels represent whether the result is normal or abnormal.

### Preprocessing Steps

* **Handling Missing Values**: Missing entries were either dropped or imputed.
* **Encoding Categorical Variables**: Used label encoding and one-hot encoding.
* **Feature Scaling**: Applied `StandardScaler` to normalize numeric features.
* **Data Splitting**: Split into training and testing sets using `train_test_split`.
* **Optional**: PCA for dimensionality reduction and visualization.

Preprocessed datasets are saved in the `Preprocessed Datasets/` directory.

---

## Models Used

| Model                       | Description                                               |
| --------------------------- | --------------------------------------------------------- |
| **Logistic Regression**     | Baseline linear model for binary classification.          |
| **Random Forest**           | Ensemble of decision trees using bagging.                 |
| **XGBoost**                 | Gradient boosting algorithm, best accuracy achieved.      |
| **SVM (RBF Kernel)**        | Effective in high-dimensional spaces.                     |
| **Multinomial Naive Bayes** | Based on Bayes’ theorem, assumes feature independence.    |
| **Gaussian Naive Bayes**    | Assumes Gaussian distribution of features.                |
| **MLP Classifier**          | Multi-layer perceptron (neural network).                  |
| **Voting Classifier**       | Soft voting ensemble combining the top-performing models. |

---

## Hyperparameter Tuning

Hyperparameters were optimized using `GridSearchCV` with 5-fold cross-validation. Examples:

* **Random Forest**: Number of estimators, max depth.
* **SVM**: C, gamma, kernel.
* **XGBoost**: Learning rate, max depth, n\_estimators.
* **MLP**: Hidden layer sizes, activation functions.

---

## Evaluation Metrics

The following metrics were used for model evaluation:

* **Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
* **Confusion Matrix**

---

## Results

| Model                  | Accuracy | Precision | Recall   | F1-Score  |
| ---------------------- | -------- | --------- | -------- | --------- |
| Logistic Regression    | 0.80     | 0.81      | 0.79     | 0.80      |
| Random Forest          | 0.85     | 0.86      | 0.85     | 0.855     |
| XGBoost                | **0.88** | **0.89**  | **0.88** | **0.885** |
| SVM (RBF Kernel)       | 0.82     | 0.83      | 0.82     | 0.825     |
| Multinomial NB         | 0.75     | 0.74      | 0.76     | 0.75      |
| Gaussian NB            | 0.78     | 0.77      | 0.79     | 0.78      |
| MLP Classifier         | 0.84     | 0.85      | 0.84     | 0.845     |
| Soft Voting Classifier | 0.87     | 0.88      | 0.87     | 0.875     |

> ✅ **Best Model**: XGBoost performed the best overall, achieving the highest accuracy and F1-score.

---

## Visualizations

* **PCA Plot**: Visualizes data in reduced dimensions for class separability.
* **Confusion Matrix**: Illustrates true positives, false positives, false negatives, and true negatives for each classifier.
* **Metric Plots**: Bar plots for comparing model performance metrics.

All visualizations can be found in the corresponding notebook.

---

## Project Structure

```
Healthcare-test-results-classification-using-Machine-Learning/
│
├── Original Dataset/                # Raw data files
├── Preprocessed Datasets/          # Cleaned and processed files
├── Notebooks/
│   └── Healthcare-Classification.ipynb  # Main Jupyter notebook
├── Models/                          # Saved model files (if any)
├── requirements.txt                # Python dependencies
└── README.md                       # Project overview
```

---

## How to Run

1. **Clone the repository**

```bash
git clone https://github.com/Nouran246/Healthcare-test-results-classification-using-Machine-Learning.git
cd Healthcare-test-results-classification-using-Machine-Learning
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Open the Jupyter Notebook**

```bash
jupyter notebook Notebooks/Healthcare-Classification.ipynb
```

4. **Run all cells**
   You can train models, visualize results, and evaluate their performance step-by-step.

---

## Contributors

* **Nouran Hassan (@Nouran246)** – Project Lead
* Contributions from team members for preprocessing, model building, and documentation

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for more information.

