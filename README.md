# Walmart Sales Prediction

**Problem**

There are many seasons that sales are significantly higher or lower than averages. If the company does not know about these seasons, it can lose too much money. Predicting future sales is one of the most crucial plans for a company. Sales forecasting gives an idea to the company for arranging stocks, calculating revenue, and deciding to make a new investment. Another advantage of knowing future sales is that achieving predetermined targets from the beginning of the seasons can have a positive effect on stock prices and investors' perceptions. Also, not reaching the projected target could significantly damage stock prices, conversely. And, it will be a big problem especially for Walmart as a big company.

**Aim:**

My aim in this project is to build a model which predicts sales of the stores. With this model, Walmart authorities can decide their future plans which is very important for arranging stocks, calculating revenue and deciding to make new investment or not.

**Key Features**
- End-to-end ML pipeline architecture.
- Modular components for scalability and maintainability.
- Experiment tracking using MLflow.
- Remote tracking integration via DagsHub.
- Hyperparameter tuning using RandomizedSearchCV.
- Model artifact versioning.

**Artifacts folder structure**

Artifacts
|
|- data_ingestion
    |- final_data.csv
    |- train.csv
    |- test.csv
|- data_transformation
    |- transformed_data
        |- train.npy
        |- test.npy
    |- transformed_object
        |- preprocessor.skops
|- model_trainer
    |- trained_model.skops


Data Ingestion: This component where data is fetched from original source, split into train and test and saved in the artifacts folder under data_ingestion.

Data Transformation: This component where data is processed, by applying Feature Engineering, encoding techniques for categorical features, converting final data into array and saving it in transformed_data in data_transformation folder. Also to store object used for applying encoding techniques in transformed_object

Model Trainer: This component where model is trained, metrics are stored in dagshub and experiments are tracked using mlflow. Utilised RandomForestRegressor along with RandomizedSearchCV to train the model.

**Observation**: RandomForestRegressor is good enough for the model instead of complex ANN and RNN. Model was able to perform better with RandomForestRegressor which is much cost effective than complex Deep Learning models.

We use R2 score in combination with MSE, MAE. R2 score evaluates how model captures overall variance, MAE gives interpretable average error for business decisions, and MSE penalizes large errors to reduce financial risk. Together, they ensure the model is both accurate and reliable.
