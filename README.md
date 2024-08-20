# Project Overview
This project aims to predict customer churn for a telecommunications company using the Telco Customer Churn dataset. The primary objective is to develop a predictive model using Artificial Neural Networks (ANN) to identify customers who are likely to leave the service. Predicting churn accurately allows the company to take proactive measures to retain customers.

# Dataset
Telco Customer Churn Dataset
Source: Kaggle Telco Customer Churn Dataset
Description: The dataset contains information on 7,043 customers, including demographics, account information, the services they signed up for, and whether they churned (i.e., left the company) in the last month.

# Key Features:

customerID: Unique identifier for each customer
gender: Gender of the customer (Male, Female)
SeniorCitizen: Indicates if the customer is a senior citizen (1, 0)
Partner: Whether the customer has a partner (Yes, No)
Dependents: Whether the customer has dependents (Yes, No)
tenure: Number of months the customer has stayed with the company
PhoneService: Whether the customer has a phone service (Yes, No)
MultipleLines: Whether the customer has multiple lines (Yes, No, No phone service)
InternetService: Customer's internet service provider (DSL, Fiber optic, No)
OnlineSecurity: Whether the customer has online security (Yes, No, No internet service)
OnlineBackup: Whether the customer has online backup (Yes, No, No internet service)
DeviceProtection: Whether the customer has device protection (Yes, No, No internet service)
TechSupport: Whether the customer has tech support (Yes, No, No internet service)
StreamingTV: Whether the customer has streaming TV (Yes, No, No internet service)
StreamingMovies: Whether the customer has streaming movies (Yes, No, No internet service)
Contract: Type of contract (Month-to-month, One year, Two year)
PaperlessBilling: Whether the customer has paperless billing (Yes, No)
PaymentMethod: Payment method (Electronic check, Mailed check, Bank transfer, Credit card)
MonthlyCharges: The amount charged to the customer monthly
TotalCharges: The total amount charged to the customer
Churn: Whether the customer churned (Yes, No)

# Project Workflow
# 1. Data Preprocessing
Handling Missing Values: Imputed missing values in the TotalCharges column.
Encoding Categorical Variables: Converted categorical variables into numerical formats using label encoding and one-hot encoding.
Feature Scaling: Applied feature scaling to normalize the input features.
# 2. Model Training
Model Architecture: Built an Artificial Neural Network using TensorFlow/Keras. The network consists of:
Input layer with 20 features
One hidden layers with 20 neurons, using the ReLU activation function
Output layer with a single neuron using the sigmoid activation function for binary classification
Model Compilation: The model was compiled using the Adam optimizer, binary cross-entropy as the loss function, and accuracy as the evaluation metric.
Model Training: The model was trained on 80% of the data, with 20% reserved for validation. Early stopping was implemented to prevent overfitting.

# 3. Model Evaluation
Confusion Matrix: Evaluated the model's performance using a confusion matrix to determine true positives, true negatives, false positives, and false negatives.
Classification Report : To analyze precision, recall and f1 score.


# Results
The ANN model achieved an accuracy of 78% on the test set, with a ROC-AUC score of YY%. The model is effective in predicting customer churn and can be utilized by the company to implement retention strategies.


# Conclusion
This project demonstrates the application of Artificial Neural Networks in predicting customer churn. The model can be further refined by experimenting with different architectures, hyperparameters, and feature engineering techniques.

# License
This project is licensed under the MIT License.

# Acknowledgements
The Telco Customer Churn dataset is sourced from Kaggle.
TensorFlow/Keras for the neural network implementation.
