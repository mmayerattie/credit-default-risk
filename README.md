# Credit Card Default Risk Segmentation

Customer analytics group project (IE University, 4th year, Business Data & Business Analytics). We segment 30,000 credit card clients of a Taiwanese bank by risk profile and test how well default can be predicted, with and without payment history.

**Team:** Mayer Attie, Ian Fernandez, Jose Bouza, Ivan Ryazantsev

## Key results

| | |
|---|---|
| Portfolio default rate | 22.1% (6,636 of 30,000 clients missed the next payment) |
| Risk segments (K-Means, k=4) | Default rates from **12.7%** to **60.5%**, a gap of nearly 5x |
| Best model | Random Forest, AUC **0.770**; top 10% of scores captures 31% of defaults (lift 3.07x) |
| Demographics-only model | AUC **0.605**, barely better than chance |
| Simple rule | `MAX_DELAY > 0` and `PAY_0 >= 2` reaches AUC 0.761 with no ML infrastructure |

The segmentation and the behavioral insights are the main deliverable. The prediction model is a supporting tool.

### The four segments

| Segment | Clients | Default rate | Profile |
|---|---|---|---|
| Low Risk | 5,510 (18%) | 12.7% | High limits (~NT$311K), low utilization, rarely late |
| Low-Moderate Risk | 9,375 (31%) | 15.9% | Moderate limits (~NT$207K), 4.5% utilization, pay early |
| Moderate-High Risk | 11,379 (38%) | 19.3% | Low limits (~NT$91K), 64% utilization, minimum payments |
| High Risk | 3,701 (12%) | 60.5% | Low limits (~NT$88K), late 4.5 of 6 months |

### Main findings

- **Behavior beats demographics.** Default rises from 11.7% with zero late months to 29.8% with one and 70.3% with six. All demographic correlations with default are below |r| = 0.05.
- **The first missed payment is the key moment.** Risk nearly triples there, which makes it the best point for intervention.
- **New clients can't be screened with this data.** With only application-time features (age, gender, education, marital status, credit limit), the best AUC is 0.605. Screening at onboarding needs external data such as bureau scores or income.

## Models compared

| Model | AUC |
|---|---|
| Random Forest | 0.770 |
| XGBoost | 0.763 |
| Decision Tree | 0.761 |
| Neural Network (MLP) | 0.734 |
| Logistic Regression | 0.720 |

Class imbalance is handled with SMOTE on the training split only. All splits use `random_state=42`.

## Recommendations

1. Run monthly risk scoring (0-100 per client) and feed it into the CRM.
2. Use the two-condition decision-tree rule as an immediate alert, no ML needed.
3. Target the Moderate-High Risk segment (38% of clients) with structured outreach at the first missed payment: email day 1, SMS day 7, call day 14, letter day 21.
4. Treat segments differently: reward reliable clients, monitor revolvers, assign specialists to delinquent accounts.

**Financial estimate (assumptions stated in the notebook):** NT$322M exposure at default, about NT$241M expected loss at 75% loss-given-default, and NT$12-24M potentially avoided if early intervention cuts delinquency by 5-10%.

## Repository structure

```
.
├── notebooks/
│   ├── 01_credit_default_risk_segmentation.ipynb   # Main analysis: cleaning, features, EDA, clustering, models, rules, recommendations
│   └── 02_onboarding_risk_model.ipynb              # Demographics-only model (what the bank knows at application time)
├── requirements.txt
└── README.md
```

## How to run

The data is not stored in the repo. The notebooks download it from the UCI repository (dataset ID 350) through `ucimlrepo`, so an internet connection is needed on the first run.

```bash
git clone https://github.com/mmayerattie/credit-default-risk-segmentation.git
cd credit-default-risk-segmentation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run `01` first, then `02`.

## Methodology

1. **Cleaning and features.** 35 duplicates removed and undocumented categorical codes recoded. Engineered features: `AVG_UTIL`, `AVG_PAY_RATIO`, `MAX_DELAY`, `NUM_LATE`, `BILL_GROWTH`.
2. **Exploration.** Default rate by payment behavior and by demographic group, plus correlations.
3. **Segmentation.** K-Means on behavioral features, with k chosen using silhouette scores.
4. **Prediction.** Logistic Regression, Decision Tree, Random Forest, XGBoost and MLP, evaluated on a stratified 80/20 split using AUC and lift. SHAP is used for interpretation.
5. **Onboarding test.** The same pipeline restricted to the 5 application-time features.
6. **Interpretable rules.** A shallow decision tree distilled into a simple alert rule.

## Limitations

- The target is failure to pay in a single month (October 2005), not long-term creditworthiness.
- The strongest predictors measure past delinquency. For clients already months late the model is accurate, but the loss has largely happened.
- One bank, one country, six months of data from 2005.
- Loss figures depend on stated assumptions (75% loss-given-default, 5-10% intervention effect) and exclude implementation costs.

## Data

Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2). Dataset: [UCI Machine Learning Repository, Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

## Tech stack

Python 3, Jupyter, pandas, NumPy, scikit-learn, XGBoost, SHAP, imbalanced-learn, matplotlib, seaborn.
