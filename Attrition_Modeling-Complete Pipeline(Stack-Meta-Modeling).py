# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:09:05 2018

@author: mkundu
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 13:16:50 2018

@author: mkundu
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os 
# Import necessary modules
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer
from sklearn.preprocessing import Imputer
# Import FeatureUnion
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import pandas as pd 
import numpy as np
from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report
#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.model_selection import train_test_split

os.chdir('C:\Users\mkundu\Desktop\Employee_Engagement\SALES ATTRITION_NEW\USABLE_DATA_ATTRITION\NON_SALES\Attition Model_data\NEW DATA BY KUMUD')

train_file_path = 'C:\Users\mkundu\Desktop\Employee_Engagement\SALES ATTRITION_NEW\USABLE_DATA_ATTRITION\NON_SALES\Attition Model_data\NEW DATA BY KUMUD\Train.xlsx' ## excel file to be provided 
test_file_path = 'C:\Users\mkundu\Desktop\Employee_Engagement\SALES ATTRITION_NEW\USABLE_DATA_ATTRITION\NON_SALES\Attition Model_data\NEW DATA BY KUMUD\Test.xlsx'

train = pd.read_excel(train_file_path)
test = pd.read_excel(test_file_path)

print '--List of Columns for Analysis--'
USABLE_COLUMNS = ['Employee ID', 'Age (Bucket)', 'Attrition', 'PerformanceRating',
                  'YearsWithCurrManager (Bucket)','Comp Ratio (Bucket)',
                  'HIPO','IC/Manager','Net_Score_Encode']


print'--USABLE COLUMNS---'
train = train[USABLE_COLUMNS]
test = test[USABLE_COLUMNS]
print train.shape
print test.shape





print '-----Train & Test File Uploaded ---'
print train.shape
print test.shape
print train['Attrition'].value_counts()
print test['Attrition'].value_counts()

print '----Shuffle before doing anything---'
train = train.sample(frac=1) ### Shuffle 
test = test.sample(frac=1)

print '----Ordinal & Dummy Encoding---'
## Age ##
train['Age (Bucket)'] = train['Age (Bucket)'].astype('category')
train['Age (Bucket)'] = train['Age (Bucket)'].cat.reorder_categories(['Early Career', 'Gen X', 'Baby Boomers'])
train['Age (Bucket)']  = train['Age (Bucket)'].cat.codes
train['Age (Bucket)'].value_counts() ## -1 missing values 

test['Age (Bucket)'] = test['Age (Bucket)'].astype('category')
test['Age (Bucket)'] = test['Age (Bucket)'].cat.reorder_categories(['Early Career', 'Gen X', 'Baby Boomers'])
test['Age (Bucket)']  = test['Age (Bucket)'].cat.codes
test['Age (Bucket)'].value_counts() ## -1 missing values 

## PerformanceRating ##
train['PerformanceRating'] = train['PerformanceRating'].astype('category')
train['PerformanceRating'] = train['PerformanceRating'].cat.reorder_categories(['Needs Improvement', 'Strong', 'Outstanding'])
train['PerformanceRating']  = train['PerformanceRating'].cat.codes
train['PerformanceRating'].value_counts() ## -1 missing values 

test['PerformanceRating'] = test['PerformanceRating'].astype('category')
test['PerformanceRating'] = test['PerformanceRating'].cat.reorder_categories(['Needs Improvement', 'Strong', 'Outstanding'])
test['PerformanceRating']  = test['PerformanceRating'].cat.codes
test['PerformanceRating'].value_counts() ## -1 missing values


##Years AT Company ##(not using)
train['YearsAtCompany (Bucket)'] = train['YearsAtCompany (Bucket)'].astype('category')
train['YearsAtCompany (Bucket)'] = train['YearsAtCompany (Bucket)'].cat.reorder_categories(['Less than 1', 'Between 1 to3', 'Between 3 to 5','Between 5 to 7','Between 7 to 10','Greater than 10'])
train['YearsAtCompany (Bucket)']  = train['YearsAtCompany (Bucket)'].cat.codes
train['YearsAtCompany (Bucket)'].value_counts()

test['YearsAtCompany (Bucket)'] = test['YearsAtCompany (Bucket)'].astype('category')
test['YearsAtCompany (Bucket)'] = test['YearsAtCompany (Bucket)'].cat.reorder_categories(['Less than 1', 'Between 1 to3', 'Between 3 to 5','Between 5 to 7','Between 7 to 10','Greater than 10'])
test['YearsAtCompany (Bucket)']  = test['YearsAtCompany (Bucket)'].cat.codes
test['YearsAtCompany (Bucket)'].value_counts()

## Years SInce Last Promotion ##(not using)
train['YearsSinceLastPromotion (Bucket)'] = train['YearsSinceLastPromotion (Bucket)'].astype('category')
train['YearsSinceLastPromotion (Bucket)'] = train['YearsSinceLastPromotion (Bucket)'].cat.reorder_categories(['Less than 1', 'Between 1 to3', 'Between 3 to 5','Between 5 to 7','Between 7 to 10','Greater than 10'])
train['YearsSinceLastPromotion (Bucket)']  = train['YearsSinceLastPromotion (Bucket)'].cat.codes
train['YearsSinceLastPromotion (Bucket)'].value_counts()

test['YearsSinceLastPromotion (Bucket)'] = test['YearsSinceLastPromotion (Bucket)'].astype('category')
test['YearsSinceLastPromotion (Bucket)'] = test['YearsSinceLastPromotion (Bucket)'].cat.reorder_categories(['Less than 1', 'Between 1 to3', 'Between 3 to 5','Between 5 to 7','Between 7 to 10','Greater than 10'])
test['YearsSinceLastPromotion (Bucket)']  = test['YearsSinceLastPromotion (Bucket)'].cat.codes
test['YearsSinceLastPromotion (Bucket)'].value_counts()

## Years with Current Manager ##
train['YearsWithCurrManager (Bucket)'] = train['YearsWithCurrManager (Bucket)'].astype('category')
train['YearsWithCurrManager (Bucket)'] = train['YearsWithCurrManager (Bucket)'].cat.reorder_categories(['Less than 1', 'Between 1 to3', 'Between 3 to 5','Between 5 to 7','Between 7 to 10','Greater than 10'])
train['YearsWithCurrManager (Bucket)']  = train['YearsWithCurrManager (Bucket)'].cat.codes
train['YearsWithCurrManager (Bucket)'].value_counts()

test['YearsWithCurrManager (Bucket)'] = test['YearsWithCurrManager (Bucket)'].astype('category')
test['YearsWithCurrManager (Bucket)'] = test['YearsWithCurrManager (Bucket)'].cat.reorder_categories(['Less than 1', 'Between 1 to3', 'Between 3 to 5','Between 5 to 7','Between 7 to 10','Greater than 10'])
test['YearsWithCurrManager (Bucket)']  = test['YearsWithCurrManager (Bucket)'].cat.codes
test['YearsWithCurrManager (Bucket)'].value_counts()

## CompRatio Bucket ##
train['Comp Ratio (Bucket)'] = train['Comp Ratio (Bucket)'].astype('category')
train['Comp Ratio (Bucket)'] = train['Comp Ratio (Bucket)'].cat.reorder_categories(['Less than .4', 'Between .4 to .6', 'Between .6 to .8','Between .8 to 1','Greater than 1'])
train['Comp Ratio (Bucket)']  = train['Comp Ratio (Bucket)'].cat.codes
train['Comp Ratio (Bucket)'].value_counts()

test['Comp Ratio (Bucket)'] = test['Comp Ratio (Bucket)'].astype('category')
test['Comp Ratio (Bucket)'] = test['Comp Ratio (Bucket)'].cat.reorder_categories(['Less than .4', 'Between .4 to .6', 'Between .6 to .8','Between .8 to 1','Greater than 1'])
test['Comp Ratio (Bucket)']  = test['Comp Ratio (Bucket)'].cat.codes
test['Comp Ratio (Bucket)'].value_counts()

## Net_Score_Encode ##
train['Net_Score_Encode'] = train['Net_Score_Encode'].astype('category')
train['Net_Score_Encode'] = train['Net_Score_Encode'].cat.reorder_categories(['Low','Neutral','High'])
train['Net_Score_Encode']  = train['Net_Score_Encode'].cat.codes
train['Net_Score_Encode'].value_counts()

test['Net_Score_Encode'] = test['Net_Score_Encode'].astype('category')
test['Net_Score_Encode'] = test['Net_Score_Encode'].cat.reorder_categories(['Low','Neutral','High'])
test['Net_Score_Encode']  = test['Net_Score_Encode'].cat.codes
test['Net_Score_Encode'].value_counts()

print '===== Ordinal Encoding Done==='
print '----Dummy Encoding Start--------'

hipo_train = pd.get_dummies(train['HIPO'],drop_first=True)
hipo_train.rename(columns = {'Yes':'hipo_yes'}, inplace = True)
hipo_test = pd.get_dummies(test['HIPO'],drop_first=True)
hipo_test.rename(columns = {'Yes':'hipo_yes'}, inplace = True)

mngr_train = pd.get_dummies(train['IC/Manager'],drop_first=True)
mngr_train.rename(columns = {'Yes':'manager_yes'}, inplace = True)
mngr_test = pd.get_dummies(test['IC/Manager'],drop_first=True)
mngr_test.rename(columns = {'Yes':'manager_yes'}, inplace = True)

train = train.join(hipo_train)
train = train.join(mngr_train)
test = test.join(hipo_test)
test  = test.join(mngr_test)

del train['HIPO']
del test['HIPO']
del train['IC/Manager']
del test['IC/Manager']

churn_dict = {'Yes':1,'No':0}
train['Attrition'] = train['Attrition'].replace(churn_dict)
test['Attrition'] = test['Attrition'].replace(churn_dict)

train.to_csv('Processed_Train.csv')
test.to_csv('Processed_Test.csv')

print '============= Data Preprocessing Done======================'




print'-------Check Number of Missing Values in Columns---'
# Check number of nulls in each feature column
nulls_per_column_train = train.isnull().sum()
print(nulls_per_column_train)
nulls_per_column_test = test.isnull().sum()
print(nulls_per_column_test)

'''
##### DO WHATEVER U WANT TO MEAN IMPUTATION(NEED TO BE TRAIN TEST) ########
print '----- Missing Values Treatment---------'
train['Age'].fillna(train['Age'].mean(),inplace=True)
train['YearsSinceLastPromotion'].fillna(train['YearsSinceLastPromotion'].mean(),inplace=True)
train['PercentSalaryHike'].fillna(train['PercentSalaryHike'].mean(),inplace=True)
###############################################################################
'''
print '--Type Casting--'
train.Age = train.Age.astype(int)
#train.OTE = train.OTE.astype(int)
train.YearsAtCompany = train.YearsAtCompany.astype(int)
train.YearsSinceLastPromotion = train.YearsSinceLastPromotion.astype(int)
train['Comp Ratio'] = train['Comp Ratio'].astype(int)

test.Age = test.Age.astype(int)
#test.OTE = test.OTE.astype(int)
test.YearsAtCompany = test.YearsAtCompany.astype(int)
test.YearsSinceLastPromotion = test.YearsSinceLastPromotion.astype(int)
test['Comp Ratio'] = test['Comp Ratio'].astype(int)

###############################################################################



print '----No of Train & Test Columns-----'
print train.shape
print test.shape
print train.dtypes
print test.dtypes



print '--Train Test & Validation Spliting ----'
X = train.drop(['Attrition'],axis=1)
y = train['Attrition']
print X.head(2)
print y.value_counts()
print '--Seperation for Train & Validation---'
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3)
print X_train.shape
print X_val.shape
print y_train.value_counts()
print y_val.value_counts()

X_test = test.drop(['Attrition'],axis=1)
y_test = test['Attrition']


print'--Takeout Employe ID for Test set ---'
del X_train['Employee ID']
del X_val['Employee ID']
test_ids = test['Employee ID']
del X_test['Employee ID']
print X_train.shape
print X_val.shape
print X_test.shape
print y_train.shape
print y_test.shape
print y_val.shape

print '--Start Tunning Xgboost Model---'

del dtrain,dtest
dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test)
dval = xgb.DMatrix(X_val)
train_labels = dtrain.get_label()
#test_labels = dtest.get_label()
params = {'objective':'binary:logistic',
          'n_estimators':1000,
         'max_depth':6,
         'eta':0.1}

num_rounds = 200 
ratio = float(np.sum(train_labels== 0))/np.sum(train_labels == 1)
params['scale_pos_weight'] = ratio
#params['scale_pos_weight'] = 8.25
print params
del X['Employee ID']
bst = xgb.train(params,dtrain,num_rounds)
y_test_prd = (bst.predict(dtest) > 0.5).astype(int)
y_val_prd = (bst.predict(dval) > 0.5).astype(int)
y_test_pred_prob = bst.predict(dtest) # Extract probs 

print '--- Check the Valdation Status---'
print confusion_matrix(y_val,y_val_prd)
print roc_auc_score(y_val,y_val_prd)
print classification_report(y_val,y_val_prd)

print'---Check the Test Status----'
print confusion_matrix(y_test,y_test_prd)
print roc_auc_score(y_test,y_test_prd)
print classification_report(y_test,y_test_prd)

print'---Complete now export the probabilities---'

result = pd.DataFrame()
result['test_id'] = test_ids
result['Probabilities'] = y_test_pred_prob
result.to_csv('NonSales_Attrition_probabilities.csv')

print '-------Test Ids Probabilities Export------'

##############################    FEATURE EXTRACTION  ##########################################
from sklearn.feature_selection import RFECV
# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=LogisticRegression(class_weight='balanced',penalty='l1',max_iter=1000), step=1, cv=10, scoring='accuracy')
rfecv.fit(X_train, y_train)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X_train.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
#---------------------------------------------------------------------------------------------------
# STACKING MODEL 
#---------------------------------------------------------------------------------------------------
### BASE LEARNERS ###

model1 = RandomForestClassifier(class_weight='balanced')
model2 = LogisticRegression()

model1.fit(X_train,y_train)
model2.fit(X_train,y_train)

preds1 = model1.predict(X_val)
preds2 = model2.predict(X_val)

test_preds1 = model1.predict(X_test)
test_preds2 = model2.predict(X_test)

stack_pred_val = np.column_stack((preds1,preds2))
stack_pred_val

stack_pred_test = np.column_stack((test_preds1,test_preds2))

#---Meta Model 

meta_model =xgb.XGBClassifier(
    n_estimators=1000, 
    max_depth=5,  
    scale_pos_weight=9.85,
    #min_weight_fraction_leaf=2
    ) 

meta_model.fit(stack_pred_val,y_val)

final_prediction = meta_model.predict(stack_pred_test)
final_prediction

print classification_report(y_test,final_prediction)
print roc_auc_score(y_test,final_prediction)
print confusion_matrix(y_test,final_prediction)

#------------------- Done --------------------------------------------------








































