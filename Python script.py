import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# ── Styling ──────────────────────────────────────────────────────────────────
BLUE   = "#1F4E79"
LIGHT  = "#2E75B6"
ACCENT = "#F4A21E"
RED    = "#C0392B"
GREEN  = "#27AE60"
BG     = "#F7F9FC"
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': BG,
                     'axes.facecolor': BG, 'axes.spines.top': False,
                     'axes.spines.right': False})

df = pd.read_csv('/Users/chugga/Desktop/Churn Analysis/fintech_customers.csv')

# ── 1. CHURN OVERVIEW ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(BG)
fig.suptitle('Customer Churn Overview', fontsize=16, fontweight='bold', color=BLUE, y=1.02)

# Donut
churn_counts = df['churned'].value_counts()
wedges, texts, autotexts = axes[0].pie(
    churn_counts, labels=['Retained', 'Churned'],
    autopct='%1.1f%%', startangle=90,
    colors=[GREEN, RED], wedgeprops=dict(width=0.5),
    textprops={'fontsize': 12})
autotexts[0].set_color('white'); autotexts[1].set_color('white')
axes[0].set_title('Overall Churn Rate', fontweight='bold', color=BLUE)

# Churn by account type
churn_by_type = df.groupby('account_type')['churned'].mean() * 100
bars = axes[1].bar(churn_by_type.index, churn_by_type.values,
                   color=[LIGHT, ACCENT, RED], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, churn_by_type.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold', color=BLUE)
axes[1].set_title('Churn Rate by Account Type', fontweight='bold', color=BLUE)
GRAY = '#555555'
axes[1].set_ylabel('Churn Rate (%)', color=GRAY)
axes[1].set_ylim(0, churn_by_type.max() * 1.3)

# Churn by state
churn_by_state = df.groupby('state')['churned'].mean() * 100
churn_by_state = churn_by_state.sort_values(ascending=True)
colors_state = [RED if v == churn_by_state.max() else LIGHT for v in churn_by_state.values]
axes[2].barh(churn_by_state.index, churn_by_state.values, color=colors_state, edgecolor='white')
axes[2].set_title('Churn Rate by State', fontweight='bold', color=BLUE)
axes[2].set_xlabel('Churn Rate (%)', color=GRAY)
plt.tight_layout()
plt.savefig('/Users/chugga/Desktop/Churn Analysis/chart1_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1 saved")

# ── 2. KEY DRIVERS ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Key Churn Drivers — Exploratory Analysis', fontsize=15,
             fontweight='bold', color=BLUE)

churned = df[df['churned'] == 1]
retained = df[df['churned'] == 0]

# Tenure
axes[0,0].hist(retained['tenure_months'], bins=20, alpha=0.7, color=GREEN, label='Retained', edgecolor='white')
axes[0,0].hist(churned['tenure_months'], bins=20, alpha=0.7, color=RED, label='Churned', edgecolor='white')
axes[0,0].set_title('Tenure vs Churn', fontweight='bold', color=BLUE)
axes[0,0].set_xlabel('Tenure (Months)'); axes[0,0].legend()

# Support calls
avg_calls = df.groupby('churned')['customer_support_calls'].mean()
bars = axes[0,1].bar(['Retained', 'Churned'], avg_calls.values, color=[GREEN, RED], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, avg_calls.values):
    axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:.1f}', ha='center', fontweight='bold', color=BLUE)
axes[0,1].set_title('Avg Support Calls vs Churn', fontweight='bold', color=BLUE)
axes[0,1].set_ylabel('Avg Calls per Customer')

# App logins
axes[1,0].hist(retained['app_logins_per_month'], bins=20, alpha=0.7, color=GREEN, label='Retained', edgecolor='white')
axes[1,0].hist(churned['app_logins_per_month'], bins=20, alpha=0.7, color=RED, label='Churned', edgecolor='white')
axes[1,0].set_title('App Logins per Month vs Churn', fontweight='bold', color=BLUE)
axes[1,0].set_xlabel('App Logins / Month'); axes[1,0].legend()

# Savings plan
sp = df.groupby(['savings_plan_active', 'churned']).size().unstack()
sp.plot(kind='bar', ax=axes[1,1], color=[GREEN, RED], edgecolor='white', linewidth=1.5)
axes[1,1].set_xticklabels(['No Savings Plan', 'Has Savings Plan'], rotation=0)
axes[1,1].set_title('Savings Plan vs Churn', fontweight='bold', color=BLUE)
axes[1,1].set_ylabel('Number of Customers')
axes[1,1].legend(['Retained', 'Churned'])

plt.tight_layout()
plt.savefig('/Users/chugga/Desktop/Churn Analysis/chart2_drivers.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2 saved")

# ── 3. ML MODELS ─────────────────────────────────────────────────────────────
le = LabelEncoder()
df_model = df.copy()
df_model['state_enc'] = le.fit_transform(df_model['state'])
df_model['account_enc'] = le.fit_transform(df_model['account_type'])

features = ['age', 'tenure_months', 'monthly_transactions', 'avg_transaction_value_ngn',
            'account_balance_ngn', 'loan_taken', 'savings_plan_active',
            'customer_support_calls', 'app_logins_per_month', 'referrals_made',
            'state_enc', 'account_enc']

X = df_model[features]
y = df_model['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr.fit(X_train_s, y_train)
lr_pred = lr.predict(X_test_s)
lr_prob = lr.predict_proba(X_test_s)[:,1]
lr_acc = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_prob)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_prob = dt.predict_proba(X_test)[:,1]
dt_acc = accuracy_score(y_test, dt_pred)
dt_auc = roc_auc_score(y_test, dt_prob)

print(f"\nLogistic Regression — Accuracy: {lr_acc:.2%}, AUC: {lr_auc:.3f}")
print(f"Decision Tree       — Accuracy: {dt_acc:.2%}, AUC: {dt_auc:.3f}")
print("\nLogistic Regression Report:")
print(classification_report(y_test, lr_pred, target_names=['Retained','Churned']))
print("\nDecision Tree Report:")
print(classification_report(y_test, dt_pred, target_names=['Retained','Churned']))

# ── 4. MODEL COMPARISON CHART ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold', color=BLUE)

# ROC curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)
axes[0].plot(fpr_lr, tpr_lr, color=LIGHT, lw=2, label=f'Logistic Regression (AUC={lr_auc:.2f})')
axes[0].plot(fpr_dt, tpr_dt, color=ACCENT, lw=2, label=f'Decision Tree (AUC={dt_auc:.2f})')
axes[0].plot([0,1],[0,1],'--', color='grey', lw=1)
axes[0].set_title('ROC Curve', fontweight='bold', color=BLUE)
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].legend(fontsize=9)

# Accuracy & AUC bar
metrics = ['Accuracy', 'AUC Score']
lr_scores = [lr_acc, lr_auc]
dt_scores = [dt_acc, dt_auc]
x = np.arange(len(metrics)); w = 0.35
axes[1].bar(x - w/2, lr_scores, w, label='Logistic Regression', color=LIGHT, edgecolor='white')
axes[1].bar(x + w/2, dt_scores, w, label='Decision Tree', color=ACCENT, edgecolor='white')
axes[1].set_xticks(x); axes[1].set_xticklabels(metrics)
axes[1].set_ylim(0, 1.1); axes[1].set_title('Accuracy & AUC', fontweight='bold', color=BLUE)
axes[1].legend(fontsize=9)
for i, (l, d) in enumerate(zip(lr_scores, dt_scores)):
    axes[1].text(i - w/2, l + 0.02, f'{l:.2f}', ha='center', fontsize=9, fontweight='bold')
    axes[1].text(i + w/2, d + 0.02, f'{d:.2f}', ha='center', fontsize=9, fontweight='bold')

# Feature importance (Decision Tree)
importances = pd.Series(dt.feature_importances_, index=features).sort_values(ascending=True).tail(8)
colors_fi = [RED if v == importances.max() else LIGHT for v in importances.values]
axes[2].barh(importances.index, importances.values, color=colors_fi, edgecolor='white')
axes[2].set_title('Top Feature Importances\n(Decision Tree)', fontweight='bold', color=BLUE)
axes[2].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('/Users/chugga/Desktop/Churn Analysis/chart3_models.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3 saved")

# ── 5. SAVE METRICS ──────────────────────────────────────────────────────────
import json
metrics_out = {
    "total_customers": len(df),
    "churned": int(df['churned'].sum()),
    "churn_rate": f"{df['churned'].mean():.1%}",
    "lr_accuracy": f"{lr_acc:.1%}",
    "lr_auc": round(lr_auc, 3),
    "dt_accuracy": f"{dt_acc:.1%}",
    "dt_auc": round(dt_auc, 3),
    "top_feature": importances.index[-1],
    "lr_report": classification_report(y_test, lr_pred, target_names=['Retained','Churned'], output_dict=True),
    "dt_report": classification_report(y_test, dt_pred, target_names=['Retained','Churned'], output_dict=True),
}
with open('/Users/chugga/Desktop/Churn Analysis/metrics.json', 'w') as f:
    json.dump(metrics_out, f)
print("\nAll done. Metrics saved.")
