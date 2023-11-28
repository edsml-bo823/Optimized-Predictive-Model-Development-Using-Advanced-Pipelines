# Regression Model Development Using Advanced Pipelines for Median Prices for Postcodes

## Project Overview

This project focused on predicting median house prices for different postcodes. The data posed challenges due to significant outliers and a lack of high correlation between main features and the target variable. The code primarily succeeded by imposing a strict threshold to remove outliers from the training and test datasets. The main notebook detailing the analysis can be found in the 'Barin Final Model.ipynb' file.


# Median House Price Prediction for Postcodes

This project focuses on predicting median house prices for different postcodes using machine learning techniques.

## Dataset Processing

The initial data processing involved reading the dataset files 'sector_data.csv', 'district_data.csv', and 'postcodes_labelled.csv'. The primary focus was on sorting and unifying the data based on the postcode.

The provided Python script involved several steps:
- Cleaning and formatting postcodes for uniformity across datasets.
- Merging dataframes 'dd' (District_Data) and 'df' (postcode_labelled) based on the 'postcode' column.
- Conducting data exploration and visualizations to understand relationships and distributions within the dataset.

## Exploratory Data Analysis (EDA)

- Explored null values and dropped them from the dataset.
- Observed histograms and boxplots to identify outliers and relationships between features.
- Identified the need for log-transforming 'medianPrice' due to extreme values affecting the data skewness.
- Visualized the relationship between 'northing' and 'easting' in relation to 'medianPrice'.

## Transformers and Pipelines

Implemented machine learning pipelines using libraries such as Pandas, NumPy, Seaborn, and Scikit-learn.
- Created pipelines for data preprocessing (numerical and categorical) using imputation, scaling, and one-hot encoding techniques.
- Utilized RandomForestRegressor, Support Vector Regression (SVR), and XGBoostRegressor models for prediction.

## Performance Metrics

- Computed various evaluation metrics including Mean Squared Error (MSE), R-squared, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Explained Variance Score, Median Absolute Error (MedAE), and Max Error for each model.
- Compared the performance of models such as RandomForest, Support Vector Regression (SVR), XGBoostRegressor, and a Dummy Classifier.

Please refer to the provided Python script for detailed code implementation and model evaluation.


## Evolution of the Project

### First Attempt
- Encoded categorical values via ordinal encoding, but it yielded unsatisfactory results.
- Worked with the 'postcode_unlabelled' dataset but struggled to derive correlated values.
- Explored various transformations, including formatting and utilizing full postcodes, without significant success.

### Second Attempt
- Extracted only the district part of the postcode, leading to 718 unique postcode districts.
- Attempted to link postcodes to easting and northing values, but it didn't enhance correlation, reducing the dataset significantly.

### Final Attempt: Merging Datasets
- Extracted district parts of postcodes for merging datasets.
- Merged 'district_data' to 'labelled_postcode' datasets, enlarging the dataset and adding 'catsPerHousehold' and 'dogsPerHousehold' variables.
- Tried incorporating a third dataset, resulting in 80% NaN values, leading to the idea's abandonment.

## Exploratory Data Analysis (EDA)

- Identified non-normally distributed features, necessitating normalization via log transformation.
- Addressed outliers in the 'medianPrice' column by log transformation to handle extreme values.
- Faced 10% missing data in the target variable; the decision to drop these values was made.
- Emphasized the need to scale, impute, and encode features.

## Transformers and Pipelines

- Implemented three pipelines and a ColumnTransformer:
  - Num_pipe: Transformed numeric features by imputing, log-transforming, and MinMaxScaling.
  - Cat_pipe: Handled categorical features by imputing and one-hot-encoding.
  - Preprocessor: Combined the two pipelines and fitted on X_train.
  - Rfr_pipeline: Utilized the Preprocessor transformer with the RandomForestRegressor model pipeline.

## Results and Performance Metrics

- Achieved improved results using a Function Transformer to log scale the data, reducing outlier impact.
- Merged datasets on postcode sectors, leading to successful OneHotEncoding, utilized in XGBoost and RandomForest models.
- Notable performance metrics:
  - Mean Squared Error DUMMY CLASSIFIER: 0.407
  - Mean Squared Error FOREST: 0.164
  - Mean Squared Error XGBOOST: 0.189
  - RandomForestRegressor model pipeline results:
    - R-squared (R2): 0.550
    - Mean Squared Error (MSE): 0.165
    - Mean Absolute Error (MAE): 0.248
    - Root Mean Squared Error (RMSE): 0.406
    - Explained Variance Score: 0.551
    - Median Absolute Error (MedAE): 0.143
    - Max Error: 4.768

### Note:
For detailed analysis and code implementation, please refer to the 'Barin Final Model.ipynb' notebook in the notebook folder.
