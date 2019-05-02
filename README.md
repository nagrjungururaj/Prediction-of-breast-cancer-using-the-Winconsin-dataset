# Prediction-of-breast-cancer-using-the-Winconsin-dataset
Analysis and prediction of breast cancer in patients 

# Problem Statement : 
To classify the type of tumour as 'Malign' or 'Benign' for the breast cancer dataset (https://www.kaggle.com/uciml/breast-cancer-wisconsin-data). For more information on the dataset, please look at this link : https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

# Approach : 
There are 4 parts of solutions,

1. Data Analyis : Histogram, Correlation matrix of features, mean and variance are calculated
2. Data cleaning : Find any missing values, NaNs, Null or empty values and columns in the dataset
3. Feature selection : Solution uses a decision trees from sklearn to rank features and drop the least 50% important features.
4. Model for classification : The solution suggest a 5 layered feed-forward neural network for classifying the type of tumour.

Model is trained and tested for 2 cases: 

1. Input : Cleaned data without feature selection ; Output : a single number denoting probability of 2 classes
2. Input : Cleaned data with feature selection ; Output : a single number denoting probability of 2 classes

# Results : 
1st case : With feature selected data as input to FFNN
Test accuracy: 0.8571428543045407
F1-score: 0.8983050847457628

2nd case: Without feature selected data as input to FFNN
Test accuracy: 0.8809523809523809
F1-score: 0.9152542372881356

# Libraries and Dependencies : 
Numpy, sklearn,keras(with tensorflow backend), seaborn, matplotlib, pandas

# How to :
