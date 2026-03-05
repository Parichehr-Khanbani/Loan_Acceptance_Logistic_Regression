# Loan_Acceptance_Logistic_Regression
***A machine learning project focused on predicting customer acceptance of personal loan offers using interpretable classification models, advanced preprocessing, and threshold optimization.***
<p align="center">
  <img src="https://github.com/Parichehr-Khanbani/Loan_Acceptance_Logistic_Regression/blob/main/assets/personal-loans.webp" alt="Loan Acceptance Banner" width="100%">
</p>

This project develops a complete end-to-end modeling pipeline starting from exploratory data analysis and feature transformation to model comparison, evaluation, and testing. The main objective is not only maximizing predictive performance but also understanding how financial and demographic factors interact to influence loan acceptance.

# 1. Project Overview

*  Built a full supervised learning pipeline for binary classification.
*  Applied feature engineering and feature transformation.
*  Compared multiple regularized linear models.
*  Performed hyperparameter tuning using cross-validation.
*  Evaluated models using ROC-AUC, Precision-Recall, Recall, and F1 metrics.
*  Optimized decision threshold based on business-oriented trade-offs.

# 2. Dataset

The dataset contains 5000 samples and 13 features of customer financial, demographic, and banking behavior attributes used to predict whether a customer will accept a personal loan offer.

| Column Name            | Description                                                                                         |
| ---------------------- | --------------------------------------------------------------------------------------------------- |
| **ID**                 | Unique customer identification number (Customer ID).                                                |
| **Age**                | Age of the customer (in years).                                                                     |
| **Experience**         | Number of years of professional work experience of the customer.                                    |
| **Income**             | Annual income of the customer (in thousands of dollars).                                            |
| **ZIP Code**           | Customer’s residential ZIP code.                                                                    |
| **Family**             | Number of family members.                                                                           |
| **CCAvg**              | Average monthly credit card spending (in thousands of dollars).                                     |
| **Education**          | Education level of the customer: 1 = High School, 2 = Undergraduate, 3 = Graduate/Advanced Degree.  |
| **Mortgage**           | Value of the customer’s home mortgage (if any).                                                     |
| **Personal Loan**      | **Target variable**: Indicates whether the customer has accepted a personal loan (1 = Yes, 0 = No). |
| **Securities Account** | Indicates whether the customer has a securities/investment account with the bank (1 = Yes, 0 = No). |
| **CD Account**         | Indicates whether the customer has a Certificate of Deposit (CD) account (1 = Yes, 0 = No).         |
| **Online**             | Indicates whether the customer uses online banking services (1 = Yes, 0 = No).                      |
| **CreditCard**         | Indicates whether the customer owns a credit card issued by the bank (1 = Yes, 0 = No).             |

# 3. Workflow Summary

The project follows a structured machine learning workflow:

**3.1. Data Exploration & Preprocessing**
*  Exploratory Data Analysis (EDA)
*  Distribution analysis
*  Feature transformation for skewed variables
*  Handling zero-inflated mortgage values
*  Feature reduction based on relevance

**3.2. Outlier Analysis**
*  Univariate analysis on numerical variables
*  Bivariate pairplot inspection
*  Validation that no extreme outliers remained after preprocessing

**3.3. Modeling Pipeline**
*  Scaling using MinMaxScaler
*  Regularization via ElasticNet penalties
*  Polynomial feature expansion for interaction modeling

# 4. Models Implemented

Four models were trained and compared:
*  Model 1 — SGDClassifier
*  Model 2 — Logistic Regression (ElasticNet)
*  Model 3 — Polynomial (Degree 2) + Logistic Regression
*  Model 4 — Polynomial (Degree 3) + Logistic Regression

Final Selected Model:
Polynomial Degree-2 Logistic Regression Pipeline.

# 5. Performance Interpretation & Conclusion

The final model demonstrates strong discrimination ability with high ROC-AUC and recall while maintaining balanced precision after threshold tuning. Interaction terms — especially those involving income, education, and spending behavior — significantly improved predictive performance compared to purely linear models.

Threshold analysis showed that adjusting the decision boundary provides better control over false positives and false negatives depending on business priorities. The final pipeline achieves robust generalization on the unseen test set, indicating effective preprocessing, regularization, and model selection.
