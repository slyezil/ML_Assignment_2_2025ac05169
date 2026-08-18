# Handwritten Digit Classification - ML Assignment 2

## A. Problem statement

The objective of this project is to build and compare five machine-learning classification models that recognize handwritten digits from 0 to 9. Each input sample is an 8 x 8 grayscale digit image represented by 64 numerical pixel-intensity features. All five models are trained on the same dataset and evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The trained models are demonstrated through an interactive Streamlit application. The application allows a user to upload test data in CSV format, select a model, view evaluation metrics, inspect a confusion matrix and classification report, and download model predictions.

## B. Dataset description

**Dataset:** Optical Recognition of Handwritten Digits  
**Public repository:** UCI Machine Learning Repository  
**Implementation used:** `sklearn.datasets.load_digits()`

The scikit-learn copy used in this project contains:

| Property | Value |
|---|---:|
| Number of instances | 1,797 |
| Number of input features | 64 |
| Number of classes | 10 |
| Target classes | Digits 0 to 9 |
| Image representation | 8 x 8 pixels |
| Feature range | 0 to 16 |
| Missing values | None |
| Classification type | Multiclass |

Each feature represents the intensity of one pixel in an 8 x 8 image. The feature names in the generated CSV follow the format `pixel_r0_c0` through `pixel_r7_c7`. The target column is named `target`.

Dataset references:

- https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

## C. Project links

- **GitHub Repository Link:**  https://github.com/slyezil/ML_Assignment_2_2025ac05169.git
- **Live Streamlit App Link:** https://mlassignment22025ac05169-op2cm6kdcxsm5tjwjo5ntx.streamlit.app/


## D. Models used and evaluation results

### Models

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Multinomial Naive Bayes Classifier
5. Random Forest Classifier - Ensemble Model

### Experimental setup

- Train-test split: 80% training and 20% testing
- Split type: Stratified
- Random state: 42
- Training rows: 1,437
- Test rows: 360
- Precision, Recall, and F1 averaging: Weighted
- Multiclass AUC: Weighted One-vs-Rest
- The same train-test split is used for every model
- Feature scaling is included inside the relevant model pipelines

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9722 | 0.9991 | 0.9724 | 0.9722 | 0.9722 | 0.9692 |
| Decision Tree | 0.8250 | 0.9028 | 0.8241 | 0.8250 | 0.8237 | 0.8057 |
| kNN | 0.9667 | 0.9950 | 0.9675 | 0.9667 | 0.9664 | 0.9631 |
| Naive Bayes | 0.8889 | 0.9911 | 0.8908 | 0.8889 | 0.8881 | 0.8769 |
| Random Forest (Ensemble) | 0.9694 | 0.9992 | 0.9701 | 0.9694 | 0.9692 | 0.9662 |

## E. Observations about model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression produced the best overall result. Its Accuracy, F1, and MCC are all above 0.96, which indicates that the scaled pixel features can be separated effectively using a linear multiclass decision boundary. |
| Decision Tree | The Decision Tree gave the lowest result among the five models. A single tree can learn nonlinear rules, but it is sensitive to the training data and may overfit local pixel patterns. |
| kNN | kNN performed strongly after feature standardization. Similar handwritten digits are close to one another in the scaled feature space, so neighbor-based classification works well for this dataset. |
| Naive Bayes | Multinomial Naive Bayes trained quickly and achieved a high AUC, but its Accuracy and F1 were lower than Logistic Regression, kNN, and Random Forest. The assumption that input features are conditionally independent is restrictive because nearby pixels in a digit image are related. |
| Random Forest (Ensemble) | Random Forest produced excellent Accuracy, AUC, F1, and MCC values. Combining many decision trees reduced the variance of a single Decision Tree and made the model more stable. |
| Overall winner for the dataset | **Logistic Regression** is the overall winner because it achieved the highest Accuracy, F1 Score, and MCC among the five implemented models. |

