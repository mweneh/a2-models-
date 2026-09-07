# Credit Risk Model Evaluation

This repository (`a2-models`) contains the code, evaluation experiments, and final reporting for the second assignment: evaluating machine learning models for identifying high-risk digital credit applications under an asymmetric cost of classification errors.

## Project Objective

The primary objective of this project is to go beyond simple predictive accuracy and evaluate models based on their actual financial impact. A missed high-risk case (false negative) carries a substantially greater financial consequence (KES 10,000) than incorrectly rejecting a good applicant (false positive, KES 800). The analysis uses Precision-Recall Area Under the Curve (PR-AUC) for discrimination and selects the operational decision threshold by minimising the expected business cost.

## Repository Structure

- `assignment2_evaluation.ipynb`: The main Jupyter Notebook containing the full evaluation pipeline. This includes chronological data splitting, model training (Logistic Regression, Random Forest, LightGBM), imbalance handling (class weighting vs. SMOTE), hyperparameter tuning using Optuna, probability calibration, and SHAP explainability.
- `report.md`: The final written report detailing the methodology, results, and deployment recommendations based on the empirical metrics.
- `data/`: Contains the cleaned dataset (`a1_cleaned_data.csv`) resulting from the first assignment, which is used as the input for this modeling stage.
- `results/`: Contains the SQLite database (`optuna_study.db`) storing the hyperparameter optimisation trials.
- `notebooks/`: Contains supplementary or reference notebooks (e.g., `lab2_solution_fixed.ipynb`).
- `get_test_metrics.py` & `run_eval.py`: Helper scripts used for quick programmatic extraction and replication of test metrics.

## Final Recommendation

Based on the chronological and leakage-resistant evaluation framework, **LightGBM** was selected as the optimal model.
- **Threshold**: `0.0694` (after Isotonic Regression calibration).
- **Test Period Cost**: KES 731,778 per 1,000 applications.
- **Estimated Savings**: KES 121,000 per 1,000 applications compared to the conventional 0.5 threshold.

## How to Run

1. **Install Dependencies**: Ensure you have the required packages installed in your Python environment:
   ```bash
   pip install pandas numpy scikit-learn lightgbm optuna shap matplotlib
   ```
2. **Execute the Notebook**: Open and run `assignment2_evaluation.ipynb` from top to bottom. It will load the data, execute the 3-fold cross-validation, run the Optuna hyperparameter study (connecting to `results/optuna_study.db`), calibrate the probabilities, and output the final test metrics alongside the calibration, PR, and SHAP curves.
