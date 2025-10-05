# Insurance-Claim-Risk-Prediction

This project develops a machine learning model to predict the likelihood of a customer filing an auto insurance claim. By identifying high-risk policyholders, the insurance company can implement proactive risk management strategies, such as adjusting premiums or offering safety programs, to mitigate potential losses.

***

## 📝 Problem Statement

Fraudulent and high-risk claims are a significant financial burden on the insurance industry. The business objective is to create a predictive model that can accurately distinguish between low-risk and high-risk customers based on their demographic, vehicle, and policy information. This will enable the company to price policies more accurately and reduce financial exposure to fraudulent claims.

### Key Challenges
* **Class Imbalance**: The dataset exhibits a significant class imbalance, with only about 6.4% of policies resulting in a claim. This requires special techniques to prevent the model from simply predicting "no claim" for every case.
* **Data Predictiveness**: The available features, while relevant, have limited predictive power. This makes it challenging to achieve a very high level of accuracy. As a result, the project aims to build the *best possible model with the current data*, acknowledging that the final performance (a ROC-AUC score of ~0.66) reflects these data constraints.


***

## 💾 Data Description

The dataset (`Insurance claims data.csv`) contains 58,592 records with 41 distinct features for each policy.

**Key Feature Categories:**
* **Policyholder Information**: `customer_age`, `segment`, `subscription_length`.
* **Vehicle Information**: `vehicle_age`, `model`, `fuel_type`, `max_torque`, `max_power`, `engine_type`, `ncap_rating`, and various binary features for vehicle equipment (e.g., `is_esc`, `is_parking_camera`).
* **Geographic Information**: `region_code`, `region_density`.
* **Target Variable**: `claim_status` (1 for a claim, 0 for no claim).

***

## ⚙️ Approach & Methodology

The project was executed through a series of structured steps, from data cleaning to model training and evaluation.

1.  **Data Cleaning and Preprocessing**:
    * The `policy_id` was dropped as it's an identifier with no predictive value.
    * Text-based columns like `max_torque` and `max_power` were cleaned using regular expressions to extract numerical values.
    * Numerous binary features stored as 'Yes'/'No' strings were converted to 1/0 integers.

2.  **Exploratory Data Analysis (EDA)**:
    * Visual analysis was conducted to understand the distributions of key features like `customer_age` and `vehicle_age`.
    * The relationship between customer segments, vehicle age, and claim status was explored to uncover initial insights.

3.  **Feature Engineering & Scaling**:
    * Categorical features with multiple unique values (e.g., `segment`, `fuel_type`, `engine_type`) were one-hot encoded to convert them into a numerical format.
    * Numerical features were standardized using `StandardScaler` to ensure that all features contributed equally to the model's training process.

4.  **Handling Class Imbalance**:
    * To address the severe class imbalance, a weighting strategy was employed. For the XGBoost model, `scale_pos_weight` was calculated to give more importance to the minority class (claims). For Logistic Regression and Random Forest, the `class_weight='balanced'` parameter was used.

5.  **Model Selection & Training**:
    * Three different classification models were trained and compared:
        * **Logistic Regression**: A reliable and interpretable baseline model.
        * **Random Forest**: An ensemble model known for its robustness.
        * **XGBoost**: A powerful gradient-boosting algorithm, often a top performer in competitions.
    * The models were trained on 80% of the data and evaluated on the remaining 20%.

6.  **Model Evaluation & Interpretation**:
    * The primary metric was the **ROC-AUC score**. The `classification_report` was also used to assess precision, recall, and F1-score for each class.
    * **SHAP (SHapley Additive exPlanations)** was used to interpret the best-performing model (XGBoost), providing insights into which features were most influential in its predictions.

***

## 📊 Results & Insights

The XGBoost model demonstrated the best performance, validating its reputation as a state-of-the-art classifier.

| Model | ROC-AUC Score |
| :--- | :--- |
| Logistic Regression | 0.6511 |
| Random Forest | 0.6133 |
| **XGBoost** | **0.6636** |

* **Best Model**: The **XGBoost Classifier** was the top-performing model with a ROC-AUC score of 0.6636, indicating a reasonable ability to distinguish between high-risk and low-risk customers.
* **Key Predictive Features (from SHAP analysis)**: The most important features driving the XGBoost model's predictions were `subscription_length` , `vehicle_age` , and `power_to_weight_ratio`. This highlights the critical role of vehicle safety features in predicting claim risk.
* **Business Impact**: A simulation showed that by targeting the top 20% of policyholders identified as high-risk, the company could potentially prevent a significant number of claims and reduce overall financial loss, demonstrating a clear business case for deploying the model.

***

## ✅ Conclusion

This project successfully developed an XGBoost model that provides a moderate predictive lift in identifying insurance claim risk, achieving the best possible result of ~0.66 ROC-AUC with the given data. The model effectively handles the significant class imbalance and provides interpretable results, highlighting that vehicle safety features are the most critical predictors.

While the model offers tangible business value, its performance is ultimately constrained by the limited predictive signal in the source data.

***

## 🚀 Future Improvements

* **Acquire More Predictive Data**: The most critical next step is to enrich the dataset with higher-impact features. This could include customer driving history, telematics data (from in-car devices), credit scores, or more detailed claim histories.
* **Hyperparameter Tuning**: Implement a more exhaustive hyperparameter search (e.g., GridSearchCV, Bayesian Optimization) to further fine-tune the XGBoost model for potentially higher performance.
* **Advanced Feature Engineering**: Explore interaction terms between features (e.g., `vehicle_age` * `ncap_rating`) to capture more complex, non-linear relationships.
* **Alternative Imbalance Techniques**: Experiment with other methods for handling class imbalance, such as SMOTE (Synthetic Minority Over-sampling Technique), to see if they can improve model performance.
