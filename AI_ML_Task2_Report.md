# Enhanced House Price Prediction System
## AI ML Task 2: Model Comparison Report

**Student Name:** Venkata Durga Sai D

---

## 1. Title/Header

**Project Title:** Enhanced House Price Prediction System  
**Objective:** Feature Engineering, Model Optimization & Performance Comparison  
**Student Name:** Venkata Durga Sai D

---

## 2. Methodology

### Task Objective
The overall objective of this task was to perform Feature Engineering, Model Optimization, and Performance Comparison using the California Housing Dataset. The goal was to build and compare multiple machine learning models to predict house prices accurately.

### Dataset Used
- **California Housing Dataset**: A classic regression dataset containing 20,640 samples with 8 features (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude) to predict house prices.

### Feature Scaling
A critical preprocessing step was applied using `StandardScaler` to standardize features. This transformation ensures that all features have a mean of 0 and standard deviation of 1, bringing them to a common scale. This step is essential because:
- Features with different scales can negatively impact model learning
- It improves convergence speed for gradient-based algorithms
- It ensures fair comparison across different models

### Models Trained and Rationale
Three regression models were trained and compared:

1. **Linear Regression (Baseline Model)**
   - Rationale: Simple baseline model that assumes a linear relationship between features and target

2. **Ridge Regression (L2 Regularization)**
   - Rationale: Addresses potential overfitting by adding penalty term to prevent large coefficients

3. **Decision Tree Regressor**
   - Rationale: Captures non-linear relationships and complex patterns in data without requiring feature scaling

---

## 3. Results: Model Performance Comparison

### Performance Metrics

| Model Name        | RMSE (Root Mean Squared Error) | $R^2$ Score (Coefficient of Determination) |
| :---------------- | :------------------------------ | :------------------------------------------ |
| Linear Regression | 0.7456                         | 0.5758                                     |
| Ridge Regression  | 0.7456                         | 0.5758                                     |
| Decision Tree     | 0.7030                         | 0.6228                                     |

### Interpretation of Metrics

- **RMSE (Root Mean Squared Error)**: Measures the average magnitude of prediction errors. Lower RMSE indicates better prediction accuracy. The Decision Tree model achieved the lowest RMSE of 0.7030.

- **$R^2$ Score (Coefficient of Determination)**: Represents the proportion of variance in the target variable explained by the model. Higher $R^2$ score indicates better explanatory power. The Decision Tree model achieved the highest $R^2$ score of 0.6228.

---

## 4. Conclusion and Justification

### Best-Performing Model
**Decision Tree Regressor** is the best-performing model based on the comparison of numerical results:

| Metric | Decision Tree | Improvement over Linear/Ridge |
|--------|--------------|-------------------------------|
| RMSE   | 0.7030       | ~5.7% lower                   |
| $R^2$  | 0.6228       | ~8.2% higher                 |

### Justification
The Decision Tree Regressor was selected as the best model for the following reasons:

1. **Lowest RMSE (0.7030)**: This indicates that the model's predictions are closest to the actual house prices on average, demonstrating superior prediction accuracy.

2. **Highest $R^2$ Score (0.6228)**: This shows that the Decision Tree model explains approximately 62.28% of the variance in house prices, compared to only 57.58% for the linear models.

3. **Non-linear Pattern Capture**: The Decision Tree successfully captures non-linear relationships and complex interactions between features (such as location coordinates and income levels) that linear models cannot adequately represent.

4. **Feature Scaling Success**: The preprocessing step with `StandardScaler` ensured that all features contributed equally to the model training, allowing the Decision Tree to effectively learn from all input features without bias toward higher-magnitude features.

---

**Task Completed Successfully!**

