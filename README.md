# House Price Prediction and Classification Analysis

## Assginment Overview
This assignment provides a comprehensive ML pipeline to analyze, predict, and classify house prices using a dataset of property sales. The assignment is divided into four main sections: Exploratory Data Analysis & Preprocessing, Continuous Price Prediction (Regression), Binary Price Classification, and Multiclass Price Categorization.

## 📊 Dataset
*   **Source:** [Kaggle - House Data](https://www.kaggle.com/datasets/shree1992/housedata?select=data.csv)
*   **Features:** Includes structural features (`bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `condition`), and location/contextual data (`city`, `yr_built`).
*   **Target Variable:** `price` (Continuous for Regression, Discretized for Classification).

## ⚙️ Project Workflow

### 1. Data Preprocessing & EDA
*   **Data Cleaning:** Identified and removed anomalous rows where `price = 0`.
*   **Feature Engineering & Scaling:** Excluded non-predictive columns (`date`, `street`, `statezip`, `country`). Applied `StandardScaler` to numerical features and `OneHotEncoder` to categorical features (`city`) using a `ColumnTransformer`.
*   **Data Splitting:** Data was split into $80\%$ training and $20\%$ testing sets. Stratified splitting was strictly enforced for classification tasks to maintain class balances.

### 2. Part 1: Regression Analysis
**Objective:** Predict the exact continuous house price.
*   **Models Evaluated:** Linear Regression, Ridge ($L2$), LASSO ($L1$), and Kernel Ridge Regression (RBF).
*   **Key Insight:** Standard Linear Regression experienced catastrophic failure (perfect multicollinearity/dummy variable trap). This was successfully resolved using regularization techniques.
*   **Results:** **Ridge** and **LASSO** emerged as the best performers, handling multicollinearity effectively and achieving an $R^2 \approx 0.68$ and an Mean Absolute Error (MAE) of $\approx \$130,500$.

### 3. Part 2: Binary Classification
**Objective:** Classify houses as "Expensive" ($1$) or "Not Expensive" ($0$) based on a median price split.
*   **Models Evaluated:** Logistic Regression, Linear SVM, Kernel SVM (RBF), K-Nearest Neighbors ($K=5$), Decision Trees, and Random Forest.
*   **Results:** Linear models and ensemble methods excelled, indicating linear separability in the high-dimensional encoded space. **Logistic Regression** yielded the best $AUC$ ($0.9350$), while **Random Forest** achieved the highest Accuracy ($87.05\%$) and $F1$-Score ($0.8706$).

### 4. Part 3: Multiclass Classification
**Objective:** Categorize houses into four balanced price brackets using quartiles: Budget ($0$), Average ($1$), Expensive ($2$), and Luxury ($3$).
*   **Models Evaluated:** Support Vector Machines (OVR), Logistic Regression (OVR & Multinomial), KNN, Decision Trees, XGBoost, LightGBM, AdaBoost, and CatBoost.
*   **Results:** Models easily differentiated between extreme classes (Budget vs. Luxury) but struggled slightly with adjacent middle classes due to the continuous nature of housing prices. **CatBoost** and **Logistic Regression (Multinomial)** were the top performers, achieving accuracies of $\approx 67.5\% - 67.7\%$.

### 5. Part 4: Theoretical Concepts
Includes explanations covering:
*   Bias-Variance Trade-off.
*   $L1$ vs. $L2$ Regularization dynamics and geometric interpretations.
*   The effects of outliers and class imbalances on various metrics.
*   Decision boundary comparisons across linear, tree-based, and distance-based algorithms.
*   Evaluation metric deep-dives ($MAPE$ limitations, Micro vs. Macro $F1$, $ROC$ vs. Precision-Recall curves).

## 🛠️ Technologies Used
*   **Language:** Python $3.x$
*   **Data Manipulation:** `pandas`, `numpy`
*   **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
*   **Data Visualization:** `matplotlib`, `seaborn`

## 🚀 How to Run the Notebook
1. Clone this repository to your local machine.
2. Ensure the `data.csv` file is placed in the same directory as the notebook.
3. Install the required dependencies:
4. Open and execute `main.ipynb` sequentially from top to bottom.

