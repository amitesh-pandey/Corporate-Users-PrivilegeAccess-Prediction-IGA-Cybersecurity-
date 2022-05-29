# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:38:57 2020

@author: Rishi
"""

import xgboost as xgb
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_curve
from sklearn. linear_model import LogisticRegression 
import numpy as np
from sklearn.utils import shuffle
from scipy import stats
import pickle

dataHR = pd.read_csv(r'D:\Hackathon\Model2\data\train\HR.csv')
dataINFRA = pd.read_csv(r'D:\Hackathon\Model2\data\train\Infra.csv')
dataMalware = pd.read_csv(r'D:\Hackathon\Model2\data\train\malware.csv')

X = dataHR.drop(['userName','EmployeID','Department','Label'], axis = 1)
y = dataHR['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 47)

#df_mal_train = df_mal.iloc[:1500,:]
#df_mal_test = df_mal.iloc[1500:,:]
#df_clean_train = df_clean.iloc[:250,:]
#df_clean_test = df_clean.iloc[250:,:]
#df_train = pd.concat([df_mal_train, df_clean_train])
#df_test = pd.concat([df_mal_test, df_clean_test])
#X_numeric = df_train.drop(['MD5', 'function_names', 'label', ', df_mal3vbaEntropy','IntegerRatio',
#                     'ConversionFunctions', 'EncFunctions', 'StringFunctions',
#                     'TrigFunctions','FileSize'], axis = 1).reset_index(drop = True)
#function_names = df_train['function_names']
#
#n_features = 20
#fh = FeatureHasher(n_features = n_features, input_type = 'string', non_negative = True)
#f_hashed = fh.fit_transform(function_names.tolist())
#f_hashed = pd.DataFrame(f_hashed.toarray(), columns = ['f'+str(i) for i in range(n_features)]).reset_index(drop = True)
#X_train = pd.concat([X_numeric, f_hashed], axis = 1)
#y_train = df_train['label']
#
#
#X_numeric = df_test.drop(['MD5', 'function_names', 'label', 'vbaEntropy','IntegerRatio',
#                     'ConversionFunctions', 'EncFunctions', 'StringFunctions',
#                     'TrigFunctions','FileSize'], axis = 1).reset_index(drop = True)
#function_names = df_test['function_names']
#
#fh = FeatureHasher(n_features = n_features, input_type = 'string', non_negative = True)
#f_hashed = fh.fit_transform(function_names.tolist())
#f_hashed = pd.DataFrame(f_hashed.toarray(), columns = ['f'+str(i) for i in range(n_features)]).reset_index(drop = True)
#X_test = pd.concat([X_numeric, f_hashed], axis = 1)
#y_test = df_test['label']

#                                                    
#

clf = LogisticRegression()
clf.fit(X_train, y_train)


# clf = xgb.XGBClassifier(objective = 'binary:logistic')
# param_dist = {'n_estimators': stats.randint(50, 1000),
#               'learning_rate': stats.uniform(0.01, 0.6),
#               'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#               'max_depth': [3, 4, 5, 6, 7, 8, 9],
#               'colsample_bytree': [0.4,0.5, 0.6, 0.7, 0.8, 0.9],#stats.uniform(0.5, 0.9),
#               'min_child_weight': [1, 2, 3, 4,5]
#              }
# kfold_5 = KFold(shuffle = True, n_splits = 5)
# grid = RandomizedSearchCV(clf,
#                           param_dist,
#                           n_jobs = -1, 
#                           random_state = 47, 
#                           scoring = 'roc_auc',
#                           cv = kfold_5,
#                           n_iter=100,
#                           verbose=1)
# grid.fit(X,y)
# print(grid.best_params_)


# model_HR = xgb.XGBClassifier(learning_rate=0.1,
#                           n_estimators=100,
#                           max_depth=4,
#                           subsample=0.7,
#                           colsample_bytree = 0.5,
#                           min_child_weight = 1,
#                           scale_pos_weight = (X.shape[0]-sum(y))/sum(y)
#                           )
# model_HR.fit(X,y)

# model = xgb.XGBClassifier(learning_rate=0.4730,
#                           n_estimators=746,
#                           max_depth=7,
#                           subsample=0.4,
#                           colsample_bytree = 0.6,
#                           min_child_weight = 5,
#                           scale_pos_weight = (X.shape[0]-sum(y))/sum(y)
#                           )
# model.fit(X_train,y_train)

#scoring = "accuracy"
#model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#       colsample_bynode=1, colsample_bytree=0.4, gamma=0, gpu_id=-1,
#       importance_type='gain', interaction_constraints='',
#       learning_rate=0.33050881863099396, max_delta_step=0, max_depth=3,
#       min_child_weight=1, monotone_constraints='()',
#       n_estimators=930, n_jobs=0, num_parallel_tree=1,
#       objective='binary:logistic', random_state=0, reg_alpha=0,
#       reg_lambda=1, scale_pos_weight=1, subsample=0.8,
#       tree_method='exact', validate_parameters=1, verbosity=None)
#model.fit(X_train,y_train)

#scoring = "roc_auc"
#model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#       colsample_bynode=1, colsample_bytree=0.7, gamma=0, gpu_id=-1,
#       importance_type='gain', interaction_constraints='',
#       learning_rate=0.2921710859040528, max_delta_step=0, max_depth=7,
#       min_child_weight=3, monotone_constraints='()',
#       n_estimators=347, n_jobs=0, num_parallel_tree=1,
#       objective='binary:logistic', random_state=0, reg_alpha=0,
#       reg_lambda=1, scale_pos_weight=1, subsample=0.9,
#       tree_method='exact', validate_parameters=1, verbosity=None)
#
#model.fit(X_train,y_train)      
#impdf = pd.DataFrame({'col':X_train.columns, 'imp':model.feature_importances_}).sort_values('imp', ascending = False)


#model_HR.save_model('D:\\Malware_xgb.model')
pickle.dump(clf, open(r'C:\Users\LENOVO\Desktop\Hackathonmodel2.pkl', 'wb'))
#X=[1,1,0,0,0,0,0,0]

# y_predict = model_HR.predict(X)
# y_predict_prob = model_HR.predict_proba(X)
# false_positive_rate, true_positive_rate, thresholds = roc_curve(y, y_predict_prob[:,1])
# roc_auc = auc(false_positive_rate, true_positive_rate)
# optimal_idx = np.argmax(true_positive_rate - false_positive_rate)
# optimal_threshold = thresholds[optimal_idx]
# print('optimal threshold', optimal_threshold)
# print('accuracy         ', accuracy_score(y, np.round(y_predict)))
# print(confusion_matrix(y, np.round(y_predict)))

# dataHR = pd.read_csv('D:\\Hackathon\\malware_test_correct1.csv')
# X = dataHR.drop(['userName','EmployeID','Department','Label'], axis = 1)
# y = dataHR['Label']
# hr = [{'username': 'abc1','access': ["redmine","doc portal"]}]



# ['internal file sharing', 'FTP server', 'redmine', 'doc portal', 'server', 'firewall', 'con file']
# features = [macro_kw_count, var_casing_ratio, Randomness_var,fun_casing_ratio, Randomness_fun,string_casing_ratio, Varience,
#             stringVarience,split_count] + string_counts + char_encoding_list
#     threshold = 0.57
#     features1 = xgb.DMatrix(np.matrix(features))
#     confidance_score = MacroModel.predict(features1).item()   
#     if confidance_score > threshold:
#         result = 1
#     else:
#         result = 0
        

# df['Pred'] = list(y_predict)
# fn = df[(df["label"]==1)&(df["Pred"]==0)]

