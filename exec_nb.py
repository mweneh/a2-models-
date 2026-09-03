import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Load data
df = pd.read_csv('./data/a1_cleaned_data.csv')
df['ts'] = pd.to_datetime(df['ts'])
df.sort_values('ts', inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Time range: {df['ts'].min()} to {df['ts'].max()}")

print("Target distribution:")
print(df['is_fraud'].value_counts(normalize=True))

# Identify numerical and categorical features
LEAKY_COLS = ['manual_review_score', 'settlement_status']
TARGET = 'is_fraud'
TIME_COL = 'ts'
GROUP_COL = 'customer_id'

IGNORE_COLS = ['txn_id', 'msisdn', 'reg_id', 'account_name', 'agent_id', 'device_id', 'counterparty', 'amount', 'gps_lat', 'gps_lon', 'balance_after', 'amount_signed'] + LEAKY_COLS + [TARGET, TIME_COL, GROUP_COL]
FEATURES = [c for c in df.columns if c not in IGNORE_COLS]
NUMERICAL = ['amount_abs', 'log_amt', 'hr_sin', 'hr_cos', 'dow_sin', 'dow_cos', 'payday_dist', 'gps_missing', 'recency_days', 'frequency', 'monetary_out', 'monetary_in', 'n_counterparty', 'cashout_ratio', 'out_in_ratio']
CATEGORICAL = ['txn_type', 'region', 'segment']

print(f"Using {len(FEATURES)} features.")

split_idx = int(len(df) * 0.8)
split_date = df['ts'].iloc[split_idx]
print(f"Split Date: {split_date}")

# Ensure we don't split in the middle of a day or timestamp exactly, but here we just strictly split by time
dev_df = df[df['ts'] < split_date].copy()
test_df = df[df['ts'] >= split_date].copy()

print(f"Development Set: {len(dev_df)} rows")
print(f"Final Test Set: {len(test_df)} rows (LOCKED until the very end)")

from sklearn.model_selection import TimeSeriesSplit

class GroupTimeSeriesSplit:
    '''Custom CV that splits by time while ensuring groups do not overlap between train and test. 
       Actually, to strictly prevent future peaking, test must be strictly after train.
       To avoid group leakage, if a customer is in test, their past data can't be in train.
       We implement a simple blocked time split, and purge any overlapping customers from train.'''
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        
    def split(self, X, y=None, groups=None):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        for train_idx, test_idx in tscv.split(X):
            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])
            
            # Purge any train index whose group is in test
            overlap = train_groups.intersection(test_groups)
            
            clean_train_idx = [i for i in train_idx if groups.iloc[i] not in overlap]
            yield np.array(clean_train_idx), test_idx

cv = GroupTimeSeriesSplit(n_splits=3)
# Let's test it
X_dev = dev_df[FEATURES]
y_dev = dev_df[TARGET]
groups_dev = dev_df[GROUP_COL]

for i, (tr, te) in enumerate(cv.split(X_dev, y_dev, groups_dev)):
    tr_groups = set(groups_dev.iloc[tr])
    te_groups = set(groups_dev.iloc[te])
    intersect = tr_groups.intersection(te_groups)
    
    tr_max_time = dev_df['ts'].iloc[tr].max()
    te_min_time = dev_df['ts'].iloc[te].min()
    print(f"Fold {i+1}: Train={len(tr)}, Val={len(te)} | Overlap={len(intersect)} | TrainMax={tr_max_time}, ValMin={te_min_time}")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import time

def build_preprocessor():
    num = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', RobustScaler()),
    ])
    cat = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='UNK')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer([('num', num, NUMERICAL), ('cat', cat, CATEGORICAL)])

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'LightGBM': LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1, verbose=-1)
}

results = []

for name, clf in models.items():
    pipe = Pipeline([('prep', build_preprocessor()), ('clf', clf)])
    
    pr_aucs = []
    roc_aucs = []
    
    t0 = time.time()
    for tr, te in cv.split(X_dev, y_dev, groups_dev):
        X_tr, y_tr = X_dev.iloc[tr], y_dev.iloc[tr]
        X_te, y_te = X_dev.iloc[te], y_dev.iloc[te]
        
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict_proba(X_te)[:, 1]
        
        pr_aucs.append(average_precision_score(y_te, preds))
        roc_aucs.append(roc_auc_score(y_te, preds))
        
    t1 = time.time()
    results.append({
        'Model': name,
        'Mean PR-AUC': np.mean(pr_aucs),
        'Mean ROC-AUC': np.mean(roc_aucs),
        'Time (s)': t1 - t0
    })

baseline_df = pd.DataFrame(results)
print(baseline_df)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Let's say LightGBM was best, or we use LightGBM for speed
lgb_base = LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1, verbose=-1)
lgb_cw = LGBMClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
smote = SMOTE(random_state=42)

pipelines = {
    'Baseline': Pipeline([('prep', build_preprocessor()), ('clf', lgb_base)]),
    'Class Weighting': Pipeline([('prep', build_preprocessor()), ('clf', lgb_cw)]),
    'SMOTE (Inside Fold)': ImbPipeline([('prep', build_preprocessor()), ('smote', smote), ('clf', lgb_base)])
}

imb_results = []
for name, pipe in pipelines.items():
    pr_aucs = []
    for tr, te in cv.split(X_dev, y_dev, groups_dev):
        X_tr, y_tr = X_dev.iloc[tr], y_dev.iloc[tr]
        X_te, y_te = X_dev.iloc[te], y_dev.iloc[te]
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict_proba(X_te)[:, 1]
        pr_aucs.append(average_precision_score(y_te, preds))
    imb_results.append({'Strategy': name, 'Mean PR-AUC': np.mean(pr_aucs)})
    
print("Imbalance Handling Results:")
print(pd.DataFrame(imb_results))

# Note: This is an intentionally BAD practice.
X_dev_prep = build_preprocessor().fit_transform(X_dev)
X_resampled, y_resampled = smote.fit_resample(X_dev_prep, y_dev)

# Because we lost the original grouping during SMOTE, we must do a naive split to demonstrate standard leakage
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=False) # Naive split without time or groups

leaky_aucs = []
for tr, te in kf.split(X_resampled):
    clf = LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1, verbose=-1)
    clf.fit(X_resampled[tr], y_resampled.iloc[tr])
    preds = clf.predict_proba(X_resampled[te])[:, 1]
    leaky_aucs.append(average_precision_score(y_resampled.iloc[te], preds))

imb_results.append({'Strategy': 'SMOTE (Before CV / LEAKY)', 'Mean PR-AUC': np.mean(leaky_aucs)})

leakage_df = pd.DataFrame(imb_results)
print(leakage_df)

import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 30.0) # Exploring class weighting
    }
    
    clf = LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
    pipe = Pipeline([('prep', build_preprocessor()), ('clf', clf)])
    
    scores = []
    for tr, te in cv.split(X_dev, y_dev, groups_dev):
        pipe.fit(X_dev.iloc[tr], y_dev.iloc[tr])
        preds = pipe.predict_proba(X_dev.iloc[te])[:, 1]
        scores.append(average_precision_score(y_dev.iloc[te], preds))
        
    return np.mean(scores)

# Uncomment below to run tuning. In practice this takes time.
study = optuna.create_study(direction='maximize', study_name='lgbm_tune', storage='sqlite:///results/optuna_study.db', load_if_exists=True)
study.optimize(objective, n_trials=60)
print('Best params:', study.best_params)

# For now we will use a dummy best_params to continue
best_params = study.best_params # {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.05, 'num_leaves': 31, 'scale_pos_weight': 12.5}
print("Using best params:", best_params)
final_clf = LGBMClassifier(**best_params, random_state=42, n_jobs=-1, verbose=-1)
final_pipe = Pipeline([('prep', build_preprocessor()), ('clf', final_clf)])

oof_preds = np.zeros(len(X_dev))
for tr, te in cv.split(X_dev, y_dev, groups_dev):
    final_pipe.fit(X_dev.iloc[tr], y_dev.iloc[tr])
    oof_preds[te] = final_pipe.predict_proba(X_dev.iloc[te])[:, 1]
    
# Remove instances that were never in a test fold (due to our strict chronological purge)
mask = oof_preds > 0
valid_oof_y = y_dev[mask]
valid_oof_preds = oof_preds[mask]

from sklearn.metrics import brier_score_loss
uncalibrated_brier = brier_score_loss(valid_oof_y, valid_oof_preds)
print(f"Uncalibrated Brier Score: {uncalibrated_brier:.4f}")

from sklearn.calibration import calibration_curve, IsotonicRegression
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import numpy as np

# Apply Isotonic Regression on the valid OOF
iso = IsotonicRegression(out_of_bounds='clip')
calibrated_preds = iso.fit_transform(valid_oof_preds, valid_oof_y)

calibrated_brier = brier_score_loss(valid_oof_y, calibrated_preds)
print(f"Calibrated Brier Score: {calibrated_brier:.4f}")

# Plot reliability diagrams
prob_true_uncal, prob_pred_uncal = calibration_curve(valid_oof_y, valid_oof_preds, n_bins=10)
prob_true_cal, prob_pred_cal = calibration_curve(valid_oof_y, calibrated_preds, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(prob_pred_uncal, prob_true_uncal, marker='o', label='Uncalibrated')
plt.plot(prob_pred_cal, prob_true_cal, marker='s', label='Isotonic Calibrated')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Reliability Diagram')
plt.legend()


FN_COST = 10000
FP_COST = 800

thresholds = np.linspace(0.01, 0.99, 100)
costs = []

for t in thresholds:
    preds_binary = (calibrated_preds >= t).astype(int)
    fn = np.sum((valid_oof_y == 1) & (preds_binary == 0))
    fp = np.sum((valid_oof_y == 0) & (preds_binary == 1))
    total_cost = (fn * FN_COST) + (fp * FP_COST)
    costs.append(total_cost)

optimal_idx = np.argmin(costs)
optimal_threshold = thresholds[optimal_idx]
optimal_cost = costs[optimal_idx]

# Calculate cost at 0.5 naive threshold
naive_binary = (calibrated_preds >= 0.5).astype(int)
naive_fn = np.sum((valid_oof_y == 1) & (naive_binary == 0))
naive_fp = np.sum((valid_oof_y == 0) & (naive_binary == 1))
naive_cost = (naive_fn * FN_COST) + (naive_fp * FP_COST)

print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"Cost at 0.5 threshold: KES {naive_cost:,.2f}")
print(f"Cost at optimal threshold: KES {optimal_cost:,.2f}")
print(f"Savings per {len(valid_oof_y)} applications: KES {naive_cost - optimal_cost:,.2f}")
print(f"Savings per 1,000 applications: KES {(naive_cost - optimal_cost) / len(valid_oof_y) * 1000:,.2f}")

plt.figure(figsize=(8, 6))
plt.plot(thresholds, costs, label='Total Cost', color='purple')
plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.2f})')
plt.axvline(0.5, color='gray', linestyle=':', label='Naive Threshold (0.5)')
plt.xlabel('Threshold')
plt.ylabel('Total Cost (KES)')
plt.title('Cost vs Threshold Curve')
plt.legend()


# Train final model on the entirety of Development set
final_pipe.fit(X_dev, y_dev)

# Predict on Final Test Set
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# 1. Uncalibrated predictions
test_preds_uncal = final_pipe.predict_proba(X_test)[:, 1]

# 2. Apply saved calibrator
test_preds_cal = iso.transform(test_preds_uncal)

# 3. Apply optimal threshold
test_preds_binary = (test_preds_cal >= optimal_threshold).astype(int)

# Evaluate cost on test set
test_fn = np.sum((y_test == 1) & (test_preds_binary == 0))
test_fp = np.sum((y_test == 0) & (test_preds_binary == 1))
test_total_cost = (test_fn * FN_COST) + (test_fp * FP_COST)
print(f"Final Test Set Total Cost: KES {test_total_cost:,.2f}")

# Add predictions back to a copy of the test dataframe for analysis
fair_df = test_df.copy()
fair_df['pred_prob'] = test_preds_cal
fair_df['approved'] = (test_preds_binary == 0).astype(int) # Approval means NO fraud (predicted 0)

print("Subgroup Fairness Analysis (by Segment):")
segment_stats = fair_df.groupby('segment').apply(lambda x: pd.Series({
    'Approval Rate': x['approved'].mean(),
    'FNR (Fraud missed)': np.sum((x[TARGET] == 1) & (x['approved'] == 1)) / (np.sum(x[TARGET] == 1) + 1e-9)
}))
print(segment_stats)

print("Subgroup Fairness Analysis (by Region):")
region_stats = fair_df.groupby('region').apply(lambda x: pd.Series({
    'Approval Rate': x['approved'].mean(),
    'FNR (Fraud missed)': np.sum((x[TARGET] == 1) & (x['approved'] == 1)) / (np.sum(x[TARGET] == 1) + 1e-9)
}))
print(region_stats)

import shap


# Extract preprocessor and fitted estimator
prep = final_pipe.named_steps['prep']
clf = final_pipe.named_steps['clf']

# Transform a sample of test data for SHAP
X_test_transformed = prep.transform(X_test.sample(1000, random_state=42))

# Use TreeExplainer for LightGBM
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_transformed)

# Global SHAP (Feature Importance)
# Note: we need feature names after one-hot encoding
cat_features = prep.transformers_[1][1].named_steps['onehot'].get_feature_names_out(CATEGORICAL)
all_features = NUMERICAL + list(cat_features)

print("Global Explanations:")
shap.summary_plot(shap_values, X_test_transformed, feature_names=all_features)

print("Local Explanation for a rejected transaction (fraud detected):")
rejected_idx = np.where(test_preds_binary == 1)[0]
if len(rejected_idx) > 0:
    idx = rejected_idx[0]
    expected_v = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    shap_v = shap_values[1][idx] if isinstance(shap_values, list) else shap_values[idx]
    print(shap.force_plot(expected_v, shap_v, X_test_transformed[idx], feature_names=all_features))
else:
    print("No rejected transactions in sample.")
