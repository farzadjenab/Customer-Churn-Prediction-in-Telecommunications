# 📉 Customer Churn Prediction in Telecommunications
### A Comparative Study of Logistic Regression, Random Forest, and XGBoost
 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-red)
![License](https://img.shields.io/badge/License-MIT-green)
 
---
 
## 📋 Overview
 
This repository contains the full code, dataset, and research paper for a machine learning study on **customer churn prediction** in the telecommunications industry. The project evaluates and compares three supervised learning algorithms — Logistic Regression, Random Forest, and XGBoost — using the publicly available IBM Telco Customer Churn dataset.
 
The companion research paper is included as [`MAIN.pdf`](./MAIN.pdf).
 
---
 
## 🗂️ Repository Structure
 
```
├── improved_churn_analysis.ipynb   # Full analysis notebook (EDA, modeling, evaluation)
├── Telco-Customer-Churn.csv        # Dataset (7,043 customers, 21 features)
├── MAIN.pdf                        # Published research paper
└── README.md
```
 
---
 
## 📊 Dataset
 
**Source:** [IBM Community — Telco Customer Churn](https://community.ibm.com/community/user/businessanalytics/blogs/stevenmacko/2019/07/11/telco-customer-churn-1113)
 
| Property | Value |
|---|---|
| Records | 7,043 customers |
| Features | 21 variables |
| Target | `Churn` (Yes / No) |
| Class balance | ~26.5% churned, ~73.5% retained |
 
**Feature categories:**
- **Demographics** — gender, senior citizen status, partner, dependents
- **Account info** — tenure, contract type, payment method, paperless billing
- **Services** — phone, internet, online security, tech support, streaming TV/movies
- **Billing** — monthly charges, total charges
---
 
## 🔬 Methodology
 
### Preprocessing
- Converted `TotalCharges` from string to numeric; removed 11 rows with missing values
- Dropped non-informative `customerID` column
- Applied **Label Encoding** for binary variables and **One-Hot Encoding** for nominal multi-class variables
- Handled class imbalance via `class_weight='balanced'` (LR & RF) and `scale_pos_weight` (XGBoost)
- Applied `StandardScaler` within a scikit-learn `Pipeline` for Logistic Regression
- **80/20 stratified train-test split** with `random_state=42`
### Models
 
| Model | Key Configuration |
|---|---|
| Logistic Regression | L2 regularization, SAGA solver, `class_weight='balanced'`, max_iter=5000 |
| Random Forest | 200 trees, Gini criterion, `class_weight='balanced_subsample'` |
| XGBoost | 200 estimators, lr=0.05, max_depth=5, `scale_pos_weight≈2.77` |
 
### Evaluation
- Hold-out test set metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- **5-fold stratified cross-validation** (mean ± std)
- **McNemar's test** for statistical significance of pairwise model differences
---
 
## 📈 Results
 
### Hold-out Test Set Performance
 
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.7264 | 0.4909 | **0.7968** | 0.6075 | **0.8350** |
| Random Forest | **0.7655** | **0.5495** | 0.6524 | 0.5966 | 0.8188 |
| XGBoost | 0.7441 | 0.5122 | 0.7888 | **0.6211** | 0.8333 |
 
### 5-Fold Cross-Validation (Mean ± Std)
 
| Model | CV Accuracy | CV Recall | CV F1 | CV ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | 0.7513 ± 0.0048 | **0.8020 ± 0.0160** | 0.6315 ± 0.0073 | **0.8459 ± 0.0054** |
| Random Forest | **0.7778 ± 0.0081** | 0.6475 ± 0.0275 | 0.6075 ± 0.0161 | 0.8297 ± 0.0068 |
| XGBoost | 0.7616 ± 0.0086 | 0.7619 ± 0.0235 | **0.6294 ± 0.0138** | 0.8448 ± 0.0046 |
 
> **Key finding:** Logistic Regression achieved the best overall performance (CV ROC-AUC = 0.8459 ± 0.0054). All pairwise differences were statistically significant at α = 0.05 via McNemar's test.


<img width="1200" height="900" alt="roc_curves_combined" src="https://github.com/user-attachments/assets/de01347f-fb5b-4e24-a0c1-cb06517bf680" />


---
 
## 🔑 Key Churn Drivers (Feature Importance)
 
Based on standardized Logistic Regression coefficients:
 
| Feature | Effect on Churn | Insight |
|---|---|---|
| **Tenure** | ↓ Strong negative | Longer-tenured customers are far less likely to churn |
| **InternetService — Fiber Optic** | ↑ Strong positive | Fiber subscribers show higher churn, possibly due to pricing sensitivity |
| **Contract — Two Year** | ↓ Strong negative | Long-term contracts are the strongest structural retention lever |
| **Contract — One Year** | ↓ Negative | One-year plans also substantially reduce churn vs. month-to-month |
| **Monthly Charges** | ↑ Positive | Higher bills correlate with increased churn probability |
| **PaymentMethod — Electronic Check** | ↑ Positive | Electronic check users churn at 45.3% vs. ~16% for auto-payment users |


 <img width="1350" height="900" alt="feature_importance_LR" src="https://github.com/user-attachments/assets/ec441e4c-930a-45c3-855a-069bd653342d" />

---
 
## 💡 Business Recommendations
 
1. **Contract upgrade programs** — Offer discounts/loyalty incentives to migrate month-to-month subscribers to annual or biennial plans
2. **Early lifecycle engagement** — Focus onboarding and retention efforts on the first 12 months, when churn risk is highest
3. **Fiber optic retention** — Proactively engage high-value fiber subscribers with tailored value-add offers
4. **Pricing review** — Reassess pricing for high monthly charge segments and bundle in support/security services
5. **CRM integration** — Embed the churn model in CRM pipelines to generate real-time risk scores for customer service teams
---
 
## ⚙️ Getting Started
 
### Requirements
 
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```
 
### Run the Notebook
 
```bash
git clone https://github.com/farzadjenab/Customer-Churn-Prediction-in-Telecommunications.git
cd Customer-Churn-Prediction-in-Telecommunications
jupyter notebook improved_churn_analysis.ipynb
```
 
---
 
## 📄 Citation
 
If you use this work, please cite the accompanying paper:
 
```bibtex
@article{moharami2024churn,
  title     = {Predicting Customer Churn in Telecommunication Industry Using Machine Learning Algorithms:
               A Comparative Study of Logistic Regression, Random Forest, and XGBoost},
  author    = {Moharami Jenab, Farzad},
  year      = {2024}
}
```
 
---
 
## 📚 References
 
Key references from the paper:
 
- Chang et al. (2024) — *Algorithms*, 17(6), 231
- Sikri et al. (2024) — *Scientific Reports*, 14, 13097
- Alotaibi & Haq (2024) — *ETASR*, 14(3), 14572–14578
- Breiman (2001) — Random Forests — *Machine Learning*, 45(1), 5–32
- Chen & Guestrin (2016) — XGBoost — *KDD 2016*
Full reference list available in [`MAIN.pdf`](./MAIN.pdf).
 
---
 
## 📝 License
 
This project is licensed under the [MIT License](LICENSE).
