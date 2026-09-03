import pandas as pd
import numpy as np

# Reconstruct the process to get the test metrics
df = pd.read_csv('./data/a1_cleaned_data.csv')
df['ts'] = pd.to_datetime(df['ts'])
df.sort_values('ts', inplace=True)
df.reset_index(drop=True, inplace=True)

TARGET = 'is_fraud'
IGNORE_COLS = ['txn_id', 'msisdn', 'reg_id', 'account_name', 'agent_id', 'device_id', 'counterparty', 'amount', 'gps_lat', 'gps_lon', 'balance_after', 'amount_signed', 'manual_review_score', 'settlement_status', TARGET, 'ts', 'customer_id']
FEATURES = [c for c in df.columns if c not in IGNORE_COLS]
CATEGORICAL = [c for c in FEATURES if df[c].dtype == 'object' or df[c].dtype == 'string' or df[c].dtype.name == 'string']
NUMERICAL = [c for c in FEATURES if c not in CATEGORICAL]

# Chronological split
test_size = int(len(df) * 0.2)
dev_df = df.iloc[:-test_size]
test_df = df.iloc[-test_size:]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

def build_preprocessor():
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('num', num_pipe, NUMERICAL), ('cat', cat_pipe, CATEGORICAL)])

best_params = {'n_estimators': 164, 'max_depth': 6, 'learning_rate': 0.001328015443289555, 'num_leaves': 56, 'scale_pos_weight': 5.14678758265047}
final_clf = LGBMClassifier(**best_params, random_state=42, n_jobs=1, verbose=-1)
final_pipe = Pipeline([('prep', build_preprocessor()), ('clf', final_clf)])

from sklearn.model_selection import train_test_split
dev_train, dev_val = train_test_split(dev_df, test_size=0.2, random_state=42)
final_pipe.fit(dev_train[FEATURES], dev_train[TARGET])
oof_preds = final_pipe.predict_proba(dev_val[FEATURES])[:, 1]
dev_target_val = dev_val[TARGET]

from sklearn.calibration import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(oof_preds, dev_target_val)

final_pipe.fit(dev_df[FEATURES], dev_df[TARGET])
test_preds_uncal = final_pipe.predict_proba(X_test)[:, 1]
test_preds_cal = iso.transform(test_preds_uncal)

optimal_threshold = 0.0694
test_preds_binary = (test_preds_cal >= optimal_threshold).astype(int)

test_pr_auc = average_precision_score(y_test, test_preds_cal)
test_roc_auc = roc_auc_score(y_test, test_preds_cal)
test_brier = brier_score_loss(y_test, test_preds_cal)

test_fn = np.sum((y_test == 1) & (test_preds_binary == 0))
test_fp = np.sum((y_test == 0) & (test_preds_binary == 1))
test_total_cost = (test_fn * 10000) + (test_fp * 800)

naive_binary = (test_preds_cal >= 0.5).astype(int)
naive_fn = np.sum((y_test == 1) & (naive_binary == 0))
naive_fp = np.sum((y_test == 0) & (naive_binary == 1))
naive_cost = (naive_fn * 10000) + (naive_fp * 800)

approval_rate = np.mean(test_preds_binary == 0)
fnr = np.sum((y_test == 1) & (test_preds_binary == 0)) / np.sum(y_test == 1)

print(f"Test PR-AUC: {test_pr_auc:.4f}")
print(f"Test ROC-AUC: {test_roc_auc:.4f}")
print(f"Test Brier: {test_brier:.4f}")
print(f"Test Approval Rate: {approval_rate * 100:.2f}%")
print(f"Test FNR: {fnr * 100:.2f}%")
print(f"Test Total Cost: {test_total_cost}")
print(f"Test Naive Cost: {naive_cost}")

print("Done")

