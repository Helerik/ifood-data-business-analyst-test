# !/usr/bin/env Python3
# Author: Erik Davino Vincent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.stats import ttest_ind

# SkLearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

# Utilities
import winsound
import warnings

def preprocess(data):

    # Ignores user ID column, it does not serve any purpose in this analysis,
    # aswell as Z_CostContact and Z_Revenue.

    data.pop('CODE_ID')
    data.pop('Z_CostContact')
    data.pop('Z_Revenue')

    # Date data from Dt_Customer was converted to integer manually (via Excell editor)

    # Preprocess income - fill in blanks with mean yearly income of the dataset.
    # Considering not many people have this value as blank, there should be little
    # problem.
    data['Income'] = data['Income'].replace(np.nan, data['Income'].mean())

    # Year_Birth is clipped at 1920
    warnings.filterwarnings('ignore')
    data['Year_Birth'][data['Year_Birth'] <= 1920] = 1920
    warnings.filterwarnings('default')

    # Removes YOLO (!?) marital status from table (substitutes it with single)
    # Substitutes Alone marital status with single
    # Substitutes Absurd marital status with married
    # Substitutes Together marital status with married
    data['Marital_Status'] = data['Marital_Status'].replace('YOLO', 'Single')
    data['Marital_Status'] = data['Marital_Status'].replace('Alone', 'Single')
    data['Marital_Status'] = data['Marital_Status'].replace('Absurd', 'Married')
    data['Marital_Status'] = data['Marital_Status'].replace('Together', 'Married')

    print('Preprocessed data: ')
    print(data, '\n')
    input('Continue: ')
    print()

    return data

def encode_scale(data):

    # Separates target labels from data
    target_labels = data.pop('Response')

    # Preprocess scale - performs scalling
    scaler = StandardScaler()
    data[['Year_Birth', 'Income', 'Dt_Customer', 'Recency', 'MntWines',
                'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                'NumWebVisitsMonth']] = scaler.fit_transform(data[['Year_Birth', 'Income', 'Dt_Customer', 'Recency', 'MntWines',
                'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                'NumWebVisitsMonth']])

    # Performs One Hot Encoding on categorical data values
    encoded = pd.get_dummies(data['Education'], prefix = 'Education', prefix_sep = '_')
    data = pd.concat([data, encoded], axis = 1)
    data.pop('Education')
    encoded = pd.get_dummies(data['Marital_Status'], prefix = 'Marital_Status', prefix_sep = '_')
    data = pd.concat([data, encoded], axis = 1)
    data.pop('Marital_Status')

    print('Scalled and encoded data: ')
    print(data, '\n')
    input('Continue: ')
    print()

    return (data, target_labels)

def initial_analysis(data):
    
    # Analyses categorical data
    print(cat_info := data[['Education', 'Marital_Status']].describe(), '\n')

    # Analyses binary data
    print(bin_info := data[['Complain', 'AcceptedCmp1',
                'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4',
                'AcceptedCmp5']].astype(bool).describe(), '\n')

    # Analyses numeric data
    print(num_info := data[['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines',
                'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                'NumWebVisitsMonth']].describe(), '\n')

    # About the target
    print(target_info := data[['Response']].astype(bool).describe(), '\n')

    # Saves information on data
    cat_info.to_csv('Categorical_Info.csv')
    bin_info.to_csv('Binary_Info.csv')
    num_info.to_csv('Numeric_Info.csv')
    target_info.to_csv('Target_Info.csv')
    input('Continue: ')
    print()

    return 1

def secondary_analysis(data):

    # See correlation matrix
    correlation_analysis(data)

    # Information on people who have accepted the offer and the ones who didn't

    # Analyses categorical data
    print(cat_info := data[['Education', 'Marital_Status', 'Response']].groupby('Response').describe(), '\n')

    # Analyses binary data
    print(bin_info := data[['Complain', 'AcceptedCmp1',
                'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4',
                'AcceptedCmp5', 'Response']].astype(bool).groupby('Response').describe(), '\n')

    # Analyses numeric data
    print(num_info := data[['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines',
                'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                'NumWebVisitsMonth', 'Response']].groupby('Response').describe(), '\n')

    # Performs t-tests assuming equal variance (they come from the same distribution)
    # Using a significance of 0.05
    # H0: equal means
    # H1: means are different
    # If p-value > significance: Can't reject H0
    # Else: Rejects H0 => H1
    t_test_results = []
    data0 = data[data['Response'] == 0]
    data1 = data[data['Response'] == 1]
    for indx in ['Year_Birth', 'Income', 'Recency', 'MntWines',
                'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                'NumWebVisitsMonth', 'Response']:

        t_test_info = ttest_ind(data0[indx], data1[indx])
        t_test_results.append((t_test_info[1] < 0.05)*': Different means' + (t_test_info[1] >= 0.05)*': Equal means')
    print(np.core.defchararray.add(np.array(['Year_Birth', 'Income', 'Recency', 'MntWines',
                'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                'NumWebVisitsMonth', 'Response']), np.array(t_test_results)))
    
    # Saves information on data
    cat_info.to_csv('Categorical_Info_Response.csv')
    bin_info.to_csv('Binary_Info_Response.csv')
    num_info.to_csv('Numeric_Info_Response.csv')
    input('Continue: ')
    print()

    return 1

def prediction_model_1(X, y):

    # Predicts whether a user, given their data, is more or less likely to
    # buy/respond to the offer.
    # This can be done with a simple support vector machine algorithm,
    # with Stochastic Gradient Descent, which classifies binary data lineary.

    clf = SGDClassifier(max_iter = 1000, alpha = 0.01, class_weight = 'balanced',
                        tol = 1e-5)
    clf.fit(X,y)

    # Show results of training on X and y
    y_pred = clf.predict(X)
    print(classification_report(y, y_pred, zero_division = 0), '\n')

    feature_importance_analysis(clf, X)

    return clf

def prediction_model_2(X, y):

    # Predicts whether a user, given their data, is more or less likely to
    # buy/respond to the offer.
    # This can be done with a slightly more complex algorithm,
    # a Multilayer Perceptron with Adam optimizer.

    clf = MLPClassifier(hidden_layer_sizes = (100,), activation = 'relu',
                        max_iter = 1000, learning_rate_init = 0.01, tol = 1e-4)
    clf.fit(X,y)

    # Show results of training on X and y
    y_pred = clf.predict(X)
    print(classification_report(y, y_pred, zero_division = 0), '\n')

    return clf

def pca_vizualization(data, labels):

    pca = PCA(n_components = 2)
    reduced_data = pca.fit_transform(data)

    plt.figure('PCA')
    plt.scatter(reduced_data[:,0], reduced_data[:,1], c = labels)
    plt.title('Vizualization of reduced data with PCA\nPurple = 0, Yellow = 1')
    plt.show()

    return 1

def correlation_analysis(data):

    corr_matx = data.corr()
    plt.figure('Correlation Matrix')
    sn.heatmap(corr_matx, annot =  False, cmap = 'RdBu_r', center = 0)
    plt.title('Correlation Matrix')
    plt.show()

def feature_importance_analysis(predictor, data):

    importance = predictor.coef_[0]
    for i, v in enumerate(importance):
        print(f'Feature: {list(data.head())[i]}, Score: {v}')

    plt.figure('Feature Importances')
    plt.bar([list(data.head())[i] for i in range(len(importance))], importance)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation = 'vertical')
    plt.title('Feature Importance Bar Chart')
    plt.show()
    print()

def simulation_campaign(predictor, train_data, dev_data, train_labels, dev_labels, avg_cost, avg_gain):

    # In this simulation, the predictor is put to test on the train_data
    # and dev_data. Since there is not much data on the dev set, the test
    # will be done in the training aswell, but it is to be noted that the
    # predictions made on the training are unreliable due to overfitting.

    # Simulation on training data
    pred_labels = predictor.predict(train_data)

    pred_train_data = train_data[pred_labels == 1]
    true_train_labels = train_labels[pred_labels == 1]

    print('Estimations based on training data')
    total_cost = avg_cost * pred_train_data.shape[0]
    print(f'Estimated cost of campaign: {total_cost} MU')
    total_gains = avg_gain * true_train_labels[true_train_labels == 1].shape[0]
    print(f'Estimated gains of campaign: {total_gains} MU')
    print(f'Estimated profits of campaign: {total_gains - total_cost} MU')
    print(f'Estimated profit percentage: {(total_gains - total_cost)/total_cost:.2%}\n')

    # Simulation on dev data
    pred_labels = predictor.predict(dev_data)

    pred_dev_data = dev_data[pred_labels == 1]
    true_dev_labels = dev_labels[pred_labels == 1]

    print('Estimations based on dev data')
    total_cost = avg_cost * pred_dev_data.shape[0]
    print(f'Estimated cost of campaign: {total_cost} MU')
    total_gains = avg_gain * true_dev_labels[true_dev_labels == 1].shape[0]
    print(f'Estimated gains of campaign: {total_gains} MU')
    print(f'Estimated profits of campaign: {total_gains - total_cost} MU')
    print(f'Estimated profit percentage: {(total_gains - total_cost)/total_cost:.2%}\n')
    
    input('Continue: ')
    print()

    return 1

def main():

    # Average cost of campaign per customer (2240 total customers). Should be 3.0
    avg_cost = 6720/2240

    # Average gain from customers who accepted the offer (334 accepted). Should be 11.0
    avg_gain = 3674/334

    # Import dataset:
    data = pd.read_csv('ml_project1_data.csv')
    
    # Preprocess data
    data = preprocess(data)

    inp = input('See initial analysis? [y/n]\n>> ')
    print()
    if inp == 'y':
        initial_analysis(data)

    inp = input('See secondary analysis? [y/n]\n>> ')
    print()
    if inp == 'y':
        secondary_analysis(data)

    # Scale and encode data
    data, target_labels = encode_scale(data)
    
    pca_vizualization(data, target_labels)

    # Separate data into train and dev sets
    train_data, dev_data, train_labels, dev_labels = train_test_split(
        data, target_labels, test_size = 0.2)

    prediction_models = [prediction_model_1, prediction_model_2]

    inp = int(input('Choose model [1, 2]\n>> '))
    print()
    
    predictor = prediction_models[inp-1](train_data, train_labels)
    
    # Show results of prediction on dev set
    pred_labels = predictor.predict(dev_data)
    print(classification_report(dev_labels, pred_labels, zero_division = 0), '\n')

    # Test the predictor and estimate profits for a better directed campaign
    simulation_campaign(predictor, train_data, dev_data, train_labels, dev_labels, avg_cost, avg_gain)
    
main()










