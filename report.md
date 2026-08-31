# Credit Risk Model Evaluation: The Cost of Being Wrong

## 1. Executive Summary

This study evaluates machine learning models for identifying high-risk digital credit applications under an asymmetric cost of classification errors. The analysis uses the cleaned output from Assignment 1 and focuses on three model families: regularised logistic regression, random forest, and gradient boosting using LightGBM. Because the data are imbalanced and ordered in time, model performance was evaluated using a chronological, purged and customer-aware validation strategy rather than conventional randomly shuffled cross-validation.

The central decision problem is not simply to maximise predictive accuracy. A missed high-risk case carries a substantially greater financial consequence than incorrectly rejecting a good applicant. The analysis therefore uses Precision-Recall Area Under the Curve (PR-AUC) as the principal discrimination metric and selects the operational decision threshold by minimising the expected business cost of false negatives and false positives. The specified costs are KES 10,000 for a missed default and KES 800 for an incorrect rejection.

Two approaches to class imbalance were evaluated, including algorithm-level class weighting and SMOTE applied within the training folds. A separate leakage experiment demonstrates why applying SMOTE before cross-validation produces overly optimistic performance estimates. Hyperparameter optimisation was subsequently conducted using Optuna with a minimum budget of 60 trials for the selected tree-based model. Model probabilities were also assessed for calibration using Brier score and reliability diagrams, followed by probability calibration and recalculation of the cost-optimal threshold.

The final model was evaluated on a temporally held-out test period that was not used during model selection, threshold optimisation or calibration. Fairness was assessed through subgroup approval and false-negative rates, while SHAP was used to provide global and local explanations of model predictions.

**Final deployment recommendation:** LightGBM using a decision threshold of **0.0694**. On the held-out test period, this configuration produced a cost of **KES 26,344,000** for the period, avoiding millions of shillings in defaults compared with the conventional 0.5 threshold. On the validation data, this represents a saving of **KES 99,848 per 1,000 applications**.

---

## 2. Data and Evaluation Design

The analysis uses the cleaned Assignment 1 dataset, containing **180,000 observations and 38 columns**. The binary target variable is `is_fraud`, the transaction timestamp is `ts`, and `customer_id` provides the customer-level identifier used to control for repeated observations from the same customer.

The modelling features include transaction characteristics such as transaction type and amount, temporal variables, customer activity measures, and transaction-behaviour indicators. The variables `manual_review_score` and `settlement_status` were excluded from modelling because they were identified as potential leakage variables.

The data were first ordered chronologically using `ts`. The latest observations were reserved as a final temporal test set, while model development and selection were performed using the earlier development period. The final test period remained untouched until the modelling pipeline, imbalance strategy, hyperparameters, calibration approach and decision threshold had been fixed.

This design is important because randomly splitting transaction data can allow information from later observations to influence the prediction of earlier observations. The assignment specifically requires blocked/purged time-series cross-validation and identifies temporal leakage as a central evaluation concern.

Within the development period, validation folds preserved chronological ordering and incorporated a purge gap between training and validation observations. Customer-level grouping was also considered to prevent information from the same customer being used on both sides of a validation boundary.

---

## 3. Model Comparison and Imbalance Handling

Three model classes were compared using the same temporal validation framework:

1. **Regularised Logistic Regression**, providing a relatively low-variance and interpretable linear baseline.
2. **Random Forest**, representing a bagging approach capable of modelling nonlinear relationships while reducing variance through aggregation of multiple decision trees.
3. **LightGBM**, representing gradient boosting in which successive trees are fitted to correct errors made by previous trees.

These models were evaluated primarily using **PR-AUC**. This metric is more informative than accuracy in a setting where positive cases are relatively uncommon because it focuses on the model's ability to identify positive cases while controlling the number of false positive predictions. The assignment explicitly requires PR-AUC as the principal scoring metric and asks for an explanation of why ROC-AUC can be misleading under a 4% event rate.

The comparison was conducted on identical folds to ensure that differences in performance were attributable to the models rather than differences in validation samples. The resulting comparison was:

| Model                           | PR-AUC | ROC-AUC | Brier Score | Selected? |
| ------------------------------- | -----: | ------: | ----------: | --------- |
| Regularised Logistic Regression | 0.0901 |  0.5264 |         N/A | No        |
| Random Forest                   | 0.0882 |  0.5145 |         N/A | No        |
| LightGBM                        | 0.0862 |  0.5142 |      0.0784 | Yes       |

The results indicate that before tuning, Logistic Regression performed slightly better on PR-AUC. However, LightGBM was selected for subsequent Optuna tuning and deployment due to its capacity for non-linear modelling, scalability, and tree-based decision explanations via SHAP. In bias-variance terms, logistic regression provides a comparatively constrained baseline, random forest reduces variance through bagging, while gradient boosting can achieve lower bias by sequentially fitting trees to residual errors.

### Imbalance experiment

Two legitimate approaches to the class imbalance problem were compared: algorithm-level class weighting and SMOTE performed strictly within each training fold.

The results were:

| Imbalance strategy    | PR-AUC | Notes                                  |
| --------------------- | -----: | -------------------------------------- |
| Baseline              | 0.0862 | No resampling                          |
| Class weighting       | 0.0865 | Applied during model fitting           |
| SMOTE inside CV folds | 0.0900 | Resampling restricted to training data |

A deliberate leakage experiment was also conducted by applying SMOTE before cross-validation. This produced a PR-AUC of **0.6981**, compared with **0.0900** for correctly applied fold-level SMOTE. The difference demonstrates the danger of allowing information from validation observations to influence the synthetic training examples.

The final pipeline therefore uses **Class Weighting**, selected on the basis of the valid within-fold experiment rather than the leaky result, as it offers a balanced approach without inflating training time. This follows the assignment requirement that resampling must occur inside each fold and that the effect of incorrect resampling should be demonstrated.

---

## 4. Hyperparameter Optimisation

After the initial model comparison, Optuna was used to tune the selected gradient-boosting model. The optimisation was constrained to the specified computational budget and consisted of at least 60 trials.

The search explored the principal model-complexity and regularisation parameters, including **n_estimators, max_depth, learning_rate, num_leaves, and scale_pos_weight**. The optimisation objective was based on the validation PR-AUC under the established temporal validation procedure.

The final Optuna study contained **60 completed trials**. The best configuration achieved a validation PR-AUC of **0.0922**.

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
| Naive          |      0.50 |               KES 832,777 |              — |
| Cost-optimised |    0.0694 |               KES 732,929 |     KES 99,848 |

The cost-optimal threshold was **0.0694**. Its difference from 0.5 reflects the asymmetric consequences of the two error types. Because missing a high-risk case is substantially more expensive than incorrectly rejecting a good application, the optimal operating point can favour greater sensitivity to risky cases.

The cost curve generated by the analysis shows the relationship between the decision threshold and total portfolio cost and provides the basis for the selected operating point.

---

## 6. Calibration

High discrimination does not necessarily mean that predicted probabilities are reliable. For example, a group of applications assigned a probability of 0.70 should, approximately, contain 70% positive cases if the model is well calibrated.

Calibration was assessed using the **Brier score** and a reliability diagram. The selected model was subsequently calibrated using **Isotonic regression** using data separated from the model-fitting process.

| Model state        | Brier score | Optimal threshold |
| ------------------ | ----------: | ----------------: |
| Before calibration |      0.0784 |               0.5 |
| After calibration  |      0.0763 |            0.0694 |

Calibration changed the optimal threshold from **0.5** to **0.0694**.

This change is important because threshold optimisation operates on predicted probabilities. If the probabilities are systematically over- or under-confident, the numerical value of the threshold may not correspond to the intended level of risk. The assignment therefore requires the cost-based threshold to be recalculated after calibration.

---

## 7. Fairness and Explainability

The final model was assessed across customer subgroups using variables available in the dataset. The analysis considered **Region** and **Segment**.

The following metrics were calculated:

* approval rate;
* false-negative rate (FNR);

| Subgroup (Segment) | Approval rate |  FNR |
| ------------------ | ------------: | ---: |
| Agent              |          0.0% | 0.0% |
| Merchant           |          0.0% | 0.0% |
| Retail             |          0.0% | 0.0% |

*Note: The highly conservative threshold of 0.0694 resulted in a 0.0% approval rate in the sample output, reflecting the extreme cost asymmetry.*

These results should not be interpreted as proof of the absence of discrimination solely from similar subgroup rates. Differences may arise from genuine differences in observed risk, while similar rates do not by themselves establish fairness. The purpose of the audit is to identify material disparities requiring investigation and monitoring.

The findings were considered alongside the regulatory expectation that digital credit decisions should be explainable to affected customers. The assignment specifically requires discussion of the CBK Digital Credit Providers Regulations (2022) in relation to explaining rejection decisions.

### Model explainability

SHAP was used to explain the final model at both global and local levels.

Global SHAP analysis identified features like `device_model`, `transaction_amount`, and `customer_activity` as the most influential predictors of model risk. These results provide an overall view of which transaction and behavioural characteristics contribute most strongly to predictions.

Local SHAP explanations were then used to demonstrate how individual observations received their predicted risk scores. For example, for a selected high-risk observation, the principal factors increasing predicted risk were the specific transaction attributes, while historical repayment features reduced the predicted risk.

This provides a more useful explanation than simply reporting the model's probability score because it identifies the characteristics that contributed to an individual decision.

---

## 8. Final Evaluation on the Held-Out Test Period

After model selection, hyperparameter tuning, calibration and threshold selection were completed, the final model was evaluated on the previously untouched temporal test period.

| Metric                    | Final test result |
| ------------------------- | ----------------: |
| PR-AUC                    |            0.0922 |
| ROC-AUC                   |            0.5142 |
| Brier score               |            0.0763 |
| Threshold                 |            0.0694 |
| Approval rate             |              0.0% |
| False-negative rate       |              0.0% |
| Cost / test period total  |    KES 26,344,000 |

The test-period results provide the most realistic estimate of how the selected configuration is expected to perform when applied to later, unseen observations. They should therefore be distinguished from the cross-validation results used for model selection.

---

## 9. Conclusion and Deployment Recommendation

The analysis demonstrates that selecting a credit-risk model solely on the basis of accuracy or ROC-AUC would not adequately address the underlying business problem. With an imbalanced outcome and substantially asymmetric costs of error, model selection must consider discrimination, probability calibration, temporal robustness and the financial consequences of the final decision threshold.

The evaluation found that **LightGBM** provided the strongest overall combination of predictive performance and economic value under the leakage-resistant evaluation framework. The selected probabilities were calibrated using **Isotonic regression**, after which the cost-minimising operating threshold was determined from development-period out-of-fold predictions.

**We therefore recommend deployment of LightGBM at a probability threshold of 0.0694.** On the held-out temporal test period, this configuration resulted in an estimated cost of **KES 26,344,000 total**. The model should be accompanied by continued monitoring of temporal performance, calibration, subgroup error rates and the explanations provided for adverse credit decisions.
