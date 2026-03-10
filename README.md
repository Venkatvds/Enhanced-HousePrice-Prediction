---

# 🏠 Enhanced House Price Prediction

A Machine Learning project that predicts housing prices using multiple regression algorithms and compares their performance through feature engineering and model optimization.

---

## 📌 Project Overview

Predicting real estate prices is complex due to multiple interacting factors such as location, population density, income levels, and housing attributes.

This project analyzes how **data preprocessing and algorithm selection affect predictive performance** using the California Housing Dataset. 

The goal is to identify the most accurate model for predicting housing prices.

---

## 📊 Dataset

The project uses the **California Housing Dataset** derived from the **1990 U.S Census**.

**Dataset Features**

* 20,640 data samples
* 8 numerical features
* Geographic and demographic information

Key features include:

* Median Income
* Housing Median Age
* Total Rooms
* Population
* Latitude & Longitude

---

## ⚙️ Machine Learning Pipeline

### 1️⃣ Data Preprocessing

**Feature Scaling**

Standardization is applied using **StandardScaler** so that:

* Mean = 0
* Variance = 1

This prevents features with large magnitudes from dominating model weights. 

**Train-Test Split**

The dataset is divided into training and testing sets to evaluate how well the model generalizes to unseen data. 

---

### 2️⃣ Models Implemented

#### Linear Regression

Baseline model assuming a linear relationship between features and house prices.

#### Ridge Regression

An extension of linear regression that uses **L2 regularization** to reduce overfitting.

#### Decision Tree Regressor

A non-parametric model that captures **non-linear relationships** by splitting the dataset into hierarchical branches. 

---

## 📈 Model Performance

| Model             | RMSE       | R² Score   |
| ----------------- | ---------- | ---------- |
| Linear Regression | 0.7456     | 0.5758     |
| Ridge Regression  | 0.7456     | 0.5758     |
| Decision Tree     | **0.7030** | **0.6228** |

The **Decision Tree Regressor achieved the best performance** with the lowest prediction error. 

---

## 📉 Visualization

The project includes an **Actual vs Predicted Price visualization**.

Insights:

* Points closer to the diagonal line represent better predictions
* Decision Tree predictions align closer to the ideal prediction line
* Linear and Ridge regression produce almost identical results due to low multicollinearity in the dataset. 

---

## 🧠 Conclusion

The **Decision Tree Regressor** was selected as the best performing model.

Reasons:

* Captures **non-linear relationships** in housing data
* Lower RMSE compared to linear models
* Better handling of interactions between geographic and economic features. 

---

## 🚀 Future Improvements

Possible improvements include:

* Random Forest Regressor
* XGBoost
* Hyperparameter tuning
* Feature importance analysis
* Model deployment as a web application

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## ▶️ How to Run the Project

Clone the repository

```bash
git clone https://github.com/Venkatvds/Enhanced-HousePrice-Prediction.git
cd Enhanced-HousePrice-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python main.py
```

or

```bash
jupyter notebook
```

---

## 👨‍💻 Author

**Venkata Durga Sai D**
Machine Learning Project
March 2026

---