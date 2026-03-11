# Customer Churn Prediction — Nigerian Fintech

A end-to-end machine learning project that predicts which customers are likely to churn for a Nigerian fintech company, identifies the key drivers behind churn, and delivers actionable business recommendations.

## Problem Statement
Customer churn is one of the most costly challenges in fintech. This project answers: *can we predict which customers will leave before they do?*

## Dataset
- 1,000 simulated Nigerian fintech customer records
- 13 features including tenure, app logins, support calls, savings plan status, and account balance
- 12% churn rate

## Models Built
| Model | Accuracy | AUC Score |
|-------|----------|-----------|
| Logistic Regression | 89.0% | 0.948 |
| Decision Tree | 87.5% | 0.871 |

✅ Logistic Regression selected as best model (AUC: 0.948)

## Key Findings
- New customers (0–6 months tenure) churn at the highest rate
- High customer support calls are a leading churn signal
- Customers with an active savings plan churn significantly less
- Low app engagement precedes departure

## Business Recommendations
1. Structured onboarding programme for new customers
2. Proactive savings plan activation campaign
3. Improve support resolution speed
4. Re-engagement push notifications for dormant users
5. Deploy churn risk scoring in production pipelines

## Tools & Libraries
- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Jupyter / VS Code

## Files
| File | Description |
|------|-------------|
| `Python script.py` | Full analysis and ML pipeline |
| `fintech_customers.csv` | Simulated dataset |

## Author
**Ayodeji Toluwani**
AI / Data Intern — NCAIR
[LinkedIn](#) | ayodejitoluwani@gmail.com
