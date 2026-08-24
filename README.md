# LoanTap Personal Loan — Credit Risk Analysis

## Problem Statement

LoanTap is building a credit underwriting layer for its Personal Loan product. The goal of this analysis is to use an applicant's financial and credit profile to predict whether a loan is likely to be **Fully Paid** or **Charged Off**.

The business problem is not simply about getting the highest accuracy. A missed defaulter can result in an NPA and financial loss, while incorrectly rejecting a good borrower can mean lost revenue. Therefore, the model needs to balance **default detection with the cost of false positives**. 

---

## Dataset

* **Dataset:** LoanTap Personal Loan Dataset
* **Records:** 396,030
* **Original features:** 27
* **Target:** `loan_status`
* **0:** Fully Paid
* **1:** Charged Off

The dataset contains information related to loan amount, interest rate, installment, credit grade, employment, income, debt-to-income ratio, credit utilization, credit history and other borrower characteristics. 

---

## Methodology

The analysis follows a complete credit-risk modelling workflow:

* Data understanding and initial inspection
* Missing-value analysis and treatment
* Removal of high-cardinality / low-utility columns
* Feature engineering
* Ordinal and one-hot encoding
* Outlier treatment using IQR-based capping
* Exploratory Data Analysis
* Class imbalance analysis
* Multicollinearity check using VIF
* Stratified train-test split
* Feature scaling using `StandardScaler`
* Baseline Logistic Regression
* SMOTE for minority-class oversampling
* Logistic Regression with SMOTE
* Threshold analysis
* ROC-AUC and Precision-Recall analysis
* Feature coefficient interpretation
* Stratified K-Fold cross-validation

The original dataset contains an approximately **80:20 split between Fully Paid and Charged Off loans**, making class imbalance a major modelling consideration. 

---

## Data Preparation

Several preprocessing decisions were made before modelling:

* `address`, `emp_title`, and `title` were removed because of their extremely high cardinality.
* `emp_length` was converted into an ordinal numerical feature.
* Missing `mort_acc` values were filled using the median within `home_ownership` groups.
* Other small amounts of missing data were handled using median imputation.
* Binary flags were created for public records, mortgage accounts and bankruptcies.
* `term` was converted from text into numerical months.
* `credit_age_years` was engineered from the credit-line and loan-issue dates.
* Extreme values in key numerical variables were capped using the IQR method.
* Categorical variables were encoded using one-hot encoding.
* Numerical features were standardized before Logistic Regression. 

---

## Key Findings

### 1. Accuracy is misleading for this problem

The baseline Logistic Regression achieved approximately **80.6% accuracy** and a ROC-AUC of **0.711**.

At first glance, the accuracy looks reasonable. However, the model detected only around **8% of actual defaulters**.

This means that approximately 92% of genuine defaulters were being classified as Fully Paid — a serious problem for a lending business. 

### 2. SMOTE significantly improves default detection

After applying SMOTE to the training data:

| Metric                | Baseline |     SMOTE |
| --------------------- | -------: | --------: |
| Accuracy              |     ~80% |      ~66% |
| ROC-AUC               |    ~0.71 |     ~0.71 |
| Charged Off Precision |     ~52% |      ~32% |
| Charged Off Recall    |      ~8% |  **~64%** |
| Charged Off F1        |    ~0.14 | **~0.42** |

The most important improvement is the increase in **Charged Off recall from ~8% to ~64%**.

Although accuracy falls, the model becomes far more useful from a credit-risk perspective because it catches substantially more real defaulters. 

### 3. SMOTE changes the operating point, not the underlying discrimination

ROC-AUC remains around **0.71** before and after SMOTE.

This means SMOTE substantially improves the classification threshold and minority-class detection, but does not fundamentally improve the model's ability to separate risky and safe borrowers.

This suggests that better input features and richer credit data would be needed to materially improve the model's discriminatory power. 

### 4. The strongest default-risk signals

Based on the standardized Logistic Regression coefficients, the major features associated with increased default probability include:

* `sub_grade`
* `dti`
* `term`
* `revol_util`
* Public-record / bankruptcy flags

Factors associated with lower default probability include:

* `annual_inc`
* `credit_age_years`
* `mort_acc_flag`

`sub_grade` is the strongest signal in the model. 

---

## Business Recommendations

### Focus on recall, not raw accuracy

For credit underwriting, accuracy alone can hide a model that misses most defaulters. **Charged Off recall and the precision-recall tradeoff** should be treated as key operational metrics. 

### Calibrate the decision threshold

The default 0.5 threshold should not automatically be treated as the optimal business threshold.

A lower threshold such as **0.3–0.4** could be considered when minimizing NPA losses is the priority, while a higher threshold may be appropriate when customer acquisition and revenue are more important.

The final threshold should be based on the actual financial cost of:

* Missing a defaulter
* Rejecting a good borrower



### Strengthen underwriting around key risk signals

* Use `sub_grade` as a major underwriting signal.
* Give additional attention to applicants with high DTI.
* Review borrowers with high credit utilization.
* Perform enhanced checks when public-record or bankruptcy indicators are present.
* Consider shorter terms or lower loan amounts for borrowers with very short credit histories.
* Apply additional underwriting criteria to small-business loan purposes.
* Consider income-to-installment checks during underwriting. 

### Enrich the data

The model's ROC-AUC of ~0.71 suggests there is room for better discrimination. Additional information such as **credit bureau data, payment history and real-time income verification** could improve the model. 

---

## Final Model

The **Logistic Regression + SMOTE** model is selected as the preferred model for this case study.

It is not the model with the highest accuracy. It is preferred because it provides a much better balance for the actual business problem by increasing default detection from approximately **8% to 64%**.

The model should be considered a **first-tier underwriting screen**, with borderline applications passed to manual review rather than making every lending decision automatically. 

---

## Limitations

* The dataset has a significant class imbalance.
* SMOTE improves minority-class recall but also increases false positives.
* ROC-AUC remains around 0.71, showing that the model's underlying discrimination is limited.
* There is an inherent precision-recall tradeoff that cannot be completely solved without better features or additional data.
* Geographic information was removed through the dropping of `address`, although location can potentially influence credit risk through local economic and housing conditions.
* The model should not be used as the sole decision-maker for borderline applications. 

---

## Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Statsmodels
* Imbalanced-learn

---

## Repository Structure

```text
loantap-credit-risk-underwriting/
├── README.md
├── requirements.txt
├── data/
│   └── logistic_regression.csv
├── notebooks/
│   └── LoanTap_Logistic_Regression.ipynb
└── results/
    └── findings_summary.txt
```

---

## How to Run

1. Clone the repository.

```bash
git clone <repository-url>
```

2. Install the required dependencies.

```bash
pip install -r requirements.txt
```

3. Place the dataset inside the `data/` folder.

4. Open:

```text
notebooks/LoanTap_Logistic_Regression.ipynb
```

5. Run the notebook from top to bottom.

---

## Final Takeaway

The biggest lesson from this case study is that **a good credit-risk model is not necessarily the model with the highest accuracy**.

The baseline model achieved around 80% accuracy but detected only 8% of actual defaulters. After addressing class imbalance with SMOTE, default recall increased to approximately 64%, making the model considerably more useful for LoanTap's underwriting objective.

The remaining challenge is deciding **how much risk LoanTap is willing to accept**. That decision should ultimately be driven by the financial cost of NPAs versus the revenue lost from rejecting good borrowers, with the classification threshold calibrated accordingly. 
