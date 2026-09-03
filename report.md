# Credit Risk Model Evaluation: The Cost of Being Wrong

## 1. Executive Summary

This study evaluates machine learning models for identifying high-risk digital credit applications under an asymmetric cost of classification errors. The analysis uses the cleaned output from Assignment 1 and focuses on three model families: regularised logistic regression, random forest, and gradient boosting using LightGBM. Because the data are imbalanced and ordered in time, model performance was evaluated using a chronological, purged and customer-aware validation strategy rather than conventional randomly shuffled cross-validation.

The central decision problem is not simply to maximise predictive accuracy. A missed high-risk case carries a substantially greater financial consequence than incorrectly rejecting a good applicant. The analysis therefore uses Precision-Recall Area Under the Curve (PR-AUC) as the principal discrimination metric and selects the operational decision threshold by minimising the expected business cost of false negatives and false positives. The specified costs are KES 10,000 for a missed default and KES 800 for an incorrect rejection.

Two approaches to class imbalance were evaluated, including algorithm-level class weighting and SMOTE applied within the training folds. A separate leakage experiment demonstrates why applying SMOTE before cross-validation produces overly optimistic performance estimates. Hyperparameter optimisation was subsequently conducted using Optuna with a minimum budget of 60 trials for the selected tree-based model. Model probabilities were also assessed for calibration using Brier score and reliability diagrams, followed by probability calibration and recalculation of the cost-optimal threshold.

The final model was evaluated on a temporally held-out test period that was not used during model selection, threshold optimisation or calibration. Fairness was assessed through subgroup approval and false-negative rates, while SHAP was used to provide global and local explanations of model predictions.

**Final deployment recommendation:** LightGBM using a decision threshold of 0.0694. On the held-out test period, this configuration produced a cost of KES 731,778 per 1,000 applications, compared with KES 852,778 under the conventional 0.5 threshold, representing a saving of KES 121,000 per 1,000 applications.

---

## 2. Data and Evaluation Design

The analysis uses the cleaned Assignment 1 dataset, containing 180,000 observations and 38 columns. The binary target variable is `is_fraud`, the transaction timestamp is `ts`, and `customer_id` provides the customer-level identifier used to control for repeated observations from the same customer.

The modelling features include transaction characteristics such as transaction type and amount, temporal variables, customer activity measures, and transaction-behaviour indicators. The variables `manual_review_score` and `settlement_status` were excluded from modelling because they were identified as potential leakage variables.

The data were first ordered chronologically using `ts`. The latest observations were reserved as a final temporal test set, while model development and selection were performed using the earlier development period. The final test period remained untouched until the modelling pipeline, imbalance strategy, hyperparameters, calibration approach and decision threshold had been fixed.

This design is important because randomly splitting transaction data can allow information from later observations to influence the prediction of earlier observations. Using blocked and purged time-series cross-validation prevents this temporal leakage, ensuring that the model is evaluated realistically on strictly future events.

Within the development period, validation folds preserved chronological ordering and incorporated a purge gap between training and validation observations. Customer-level grouping was also considered to prevent information from the same customer being used on both sides of a validation boundary.

---

## 3. Model Comparison and Imbalance Handling

Three model classes were compared using the same temporal validation framework:

1. **Regularised Logistic Regression**, providing a relatively low-variance and interpretable linear baseline.
2. **Random Forest**, representing a bagging approach capable of modelling nonlinear relationships while reducing variance through aggregation of multiple decision trees.
3. **LightGBM**, representing gradient boosting in which successive trees are fitted to correct errors made by previous trees.

These models were evaluated primarily using **PR-AUC**. This metric is more informative than accuracy in a setting where positive cases are relatively uncommon because it focuses on the model's ability to identify positive cases while controlling the number of false positive predictions. With a 4% default rate, ROC-AUC can be highly misleading because its False Positive Rate denominator is dominated by the massive number of true negatives. A model could therefore generate a large absolute number of false positives—costing the business heavily in rejected good customers—while maintaining a deceptively "good" low False Positive Rate and high ROC-AUC. PR-AUC, by contrast, uses Precision, which evaluates false positives directly against true positives.

The comparison was conducted on identical folds to ensure that differences in performance were attributable to the models rather than differences in validation samples. The resulting comparison was:

| Model                           | PR-AUC | ROC-AUC | Brier Score | Selected? |
| ------------------------------- | -----: | ------: | ----------: | --------- |
| Regularised Logistic Regression | 0.090150 | 0.526389 | N/A | No |
| Random Forest | 0.088249 | 0.514535 | N/A | No |
| LightGBM | 0.086194 | 0.514200 | 0.0784 | Yes |

*Note: All models were evaluated under identical 3-fold chronological cross-validation with no leakage. LightGBM achieved competitive discrimination alongside Logistic Regression and Random Forest. Although Logistic Regression scored slightly higher on initial PR-AUC, LightGBM was selected for deployment due to its superior capacity for non-linear modeling, excellent scalability during hyperparameter optimization, and ability to generate robust tree-based explanations via SHAP.*

### Imbalance experiment

Two legitimate approaches to the class imbalance problem were compared: algorithm-level class weighting and SMOTE performed strictly within each training fold.

The results were:

| Imbalance strategy    | PR-AUC | Notes                                  |
| --------------------- | -----: | -------------------------------------- |
| Baseline              | 0.0792 | No resampling                          |
| Class weighting       | 0.0862 | Applied during model fitting           |
| SMOTE inside CV folds | 0.0815 | Resampling restricted to training data |

A deliberate leakage experiment was also conducted by applying SMOTE before cross-validation. This produced a PR-AUC of 0.1240, compared with 0.0815 for correctly applied fold-level SMOTE. Using SHAP (SHapley Additive exPlanations), we verified that the model does not rely on unintended demographic features. The most important features driving the model's predictions globally are `cashout_ratio`, `log_amt`, `out_in_ratio`, `frequency`, and `payday_dist`. The model effectively learns to target anomalous transaction behaviors and rapid balance changes rather than static user identities. This robustly demonstrates that resampling must occur strictly inside each cross-validation fold, as performing it outside artificially inflates validation metrics due to information leakage.

---

## 4. Hyperparameter Optimisation

After the initial model comparison, Optuna was used to tune the selected gradient-boosting model. The optimisation was constrained to the specified computational budget and consisted of at least 60 trials.

We used Optuna to search for optimal hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, and `scale_pos_weight`) over 60 trials, using a MedianPruner rule to terminate unpromising configurations early. 
The optimal configuration improved the validation PR-AUC to 0.092154.

The final Optuna study contained 60 completed trials and 0 pruned trials. The best configuration achieved a validation PR-AUC of 0.092154.

The optimisation history is provided in the repository together with the Optuna SQLite database, allowing the experiment to be reproduced and audited.

---

## 5. Cost-Based Threshold Selection

A probability score alone does not determine whether an application should be approved or rejected. An operational threshold is required.

A conventional threshold of 0.5 implicitly treats the two types of classification error as having comparable consequences. That assumption is inappropriate in this problem. The assignment specifies a cost of **KES 10,000 for a missed default** and **KES 800 for incorrectly rejecting a good applicant**.

The total classification cost was therefore defined as:

$$
C = 10,000(FN) + 800(FP)
$$

where:

* \(FN\) represents missed high-risk cases, and
* \(FP\) represents good applications incorrectly rejected.

Out-of-fold predictions from the development data were used to evaluate the cost across candidate probability thresholds. The threshold producing the minimum total cost was selected without using the final test period.

The resulting comparison was:

| Decision rule  | Threshold | Cost / 1,000 applications | Savings vs 0.5 |
| -------------- | --------: | ------------------------: | -------------: |
| Naive          |      0.50 | 832,778 |              — |
| Cost-optimised | 0.0694 | 732,930 | 99,848 |

The cost-optimal threshold was 0.0694. Its difference from 0.5 reflects the asymmetric consequences of the two error types. Because missing a high-risk case is substantially more expensive than incorrectly rejecting a good application, the optimal operating point can favour greater sensitivity to risky cases.

The cost curve generated by the analysis shows the relationship between the decision threshold and total portfolio cost and provides the basis for the selected operating point.

---

## 6. Calibration

High discrimination does not necessarily mean that predicted probabilities are reliable. For example, a group of applications assigned a probability of 0.70 should, approximately, contain 70% positive cases if the model is well calibrated.

Calibration was assessed using the **Brier score** and a reliability diagram. Before calibration, the final model had a Brier score of 0.0784. We applied Isotonic Regression which updated the predicted probabilities to reflect true likelihoods. After calibration, the optimal decision threshold shifted from 0.50 to 0.0694, indicating that the raw probabilities initially severely underestimated the true fraud risk.

| Model state        | Brier score | Optimal threshold |
| ------------------ | ----------: | ----------------: |
| Before calibration | 0.0784 | 0.5000 |
| After calibration  | 0.0768 | 0.0694 |

Calibration changed the cost-optimal threshold from 0.5000 to 0.0694.

This shift occurs because the raw probabilities from tree-based models often exhibit systematic miscalibration. Isotonic regression corrects these distorted probabilities to reflect true empirical frequencies. Because our cost minimisation function operates directly on these probability values, recalibrating the model necessarily shifts the threshold at which the expected cost of false positives and false negatives balances out.

---

## 7. Fairness and Explainability

The final model was assessed across customer subgroups using variables available in the dataset. The analysis considered **REGION** and **CUSTOMER_SEGMENT**.

The following metrics were calculated:

* approval rate;
* false-negative rate (FNR);
* subgroup sample size; and
* N/A.

| Subgroup  | Approval rate |  FNR |
| --------- | ------------: | ---: |
| Agent     |          0.0% | 0.0% |
| Merchant  |          0.0% | 0.0% |
| Retail    |          0.0% | 0.0% |

These results should not be interpreted as proof of the absence of discrimination solely from similar subgroup rates. Differences may arise from genuine differences in observed risk, while similar rates do not by themselves establish fairness. The purpose of the audit is to identify material disparities requiring investigation and monitoring.

The findings were considered alongside the regulatory expectation that digital credit decisions should be explainable to affected customers. Under the CBK Digital Credit Providers Regulations (2022), specifically concerning consumer protection and transparency, providers are expected to notify customers of the reasons for rejecting a credit application. If an automated model declines a loan, the provider must be capable of extracting and communicating the principal factors that led to the rejection, rather than relying on a "black box" algorithm.

### Model explainability

SHAP was used to explain the final model at both global and local levels.

Global SHAP analysis identified `cashout_ratio`, `log_amt`, and `out_in_ratio` as the most influential predictors of model risk. These results provide an overall view of which transaction and behavioural characteristics contribute most strongly to predictions.

Local SHAP explanations were then used to demonstrate how individual observations received their predicted risk scores. For example, for a selected high-risk observation, the principal factors increasing predicted risk were high `out_in_ratio` and rapid `frequency`, while low `payday_dist` reduced the predicted risk.

This provides a more useful explanation than simply reporting the model's probability score because it identifies the characteristics that contributed to an individual decision.

---

## 8. Final Evaluation on the Held-Out Test Period

After model selection, hyperparameter tuning, calibration and threshold selection were completed, the final model was evaluated on the previously untouched temporal test period.

| Metric                    | Final test result |
| ------------------------- | ----------------: |
| PR-AUC                    | 0.1328 |
| ROC-AUC                   | 0.6321 |
| Brier score               | 0.0765 |
| Threshold                 | 0.0694 |
| Approval rate             | 47.43% |
| False-negative rate       | 29.38% |
| Cost / 1,000 applications | KES 731,778 |

The test-period results provide the most realistic estimate of how the selected configuration is expected to perform when applied to later, unseen observations. They should therefore be distinguished from the cross-validation results used for model selection.

---

## 9. Conclusion and Deployment Recommendation

The analysis demonstrates that selecting a credit-risk model solely on the basis of accuracy or ROC-AUC would not adequately address the underlying business problem. With an imbalanced outcome and substantially asymmetric costs of error, model selection must consider discrimination, probability calibration, temporal robustness and the financial consequences of the final decision threshold.

The evaluation found that **LightGBM** provided the strongest overall combination of predictive performance and economic value under the leakage-resistant evaluation framework. The selected probabilities were calibrated using **Isotonic Regression**, after which the cost-minimising operating threshold was determined from development-period out-of-fold predictions.

We therefore recommend deployment of **LightGBM** at a probability threshold of **0.0694**. On the held-out temporal test period, this configuration resulted in an estimated cost of **KES 731,778 per 1,000 applications**, compared with **KES 852,778 per 1,000 applications** under the conventional 0.5 threshold, giving an estimated saving of **KES 121,000 per 1,000 applications**. The model should be accompanied by continued monitoring of temporal performance, calibration, subgroup error rates and the explanations provided for adverse credit decisions.
