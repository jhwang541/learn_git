import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

rcParams['figure.figsize'] = 20, 8
from pylab import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import make_scorer
from nltk import precision

def parameter_tuning(x_dev, y_dev, mld_builder, initial_params, param_grid_list,cv=3, scoring='roc_auc', drop_threshold=0.05, feature_name='auto', categorical_feature='auto'):
    ite = 1
    results_summary = pd.DataFrame()
    result_params = initial_params

    for param_grid in param_grid_list:
        estimator_grid = GridSearchCV(estimator=mld_builder(**result_params),
                                      param_grid=param_grid,
                                      cv=cv,
                                      scoring=scoring,
                                      n_jobs=5)
        estimator_grid.fit(X=x_dev, y=y_dev, feature_name=feature_name, categorical_feature=categorical_feature)
        cv_results = estimator_grid.cv_results_
#         print cv_results
        results = pd.DataFrame({'params': cv_results['params'],
                                'mean_train_score': cv_results['mean_train_score'],
                                'mean_test_score': cv_results['mean_test_score']})
        results['interation'] = ite
        print(ite)
        #        print param_grid
        ite = ite + 1
        results['drop%'] = 1 - (results['mean_test_score'] / results['mean_train_score'])
        min_threshold = max(drop_threshold, results[['drop%']].min()[0])
        results_selected = results[(results['drop%'] < min_threshold)]
        best_stable_param = results_selected.params[results_selected['mean_test_score'].argmax()]
        results['selected'] = ''
        results.selected[results_selected['mean_test_score'].argmax()] = '<-----'
        for var_name in best_stable_param.keys():
            result_params[var_name] = best_stable_param[var_name]
        # print result_params
        results_summary = results_summary.append(results)
    return {'results_summary': results_summary, 'best_params': result_params}

def modelfit(x_dev, y_dev, x_oot, y_oot, mld_builder, predictors, performCV=True, printFeatureImportance=True,
             cv_folds=5, feature_name='auto', categorical_feature='auto'):
    # Fit the algorithm on the data
    mld = mld_builder.fit(X=x_dev[predictors], y=y_dev, feature_name=feature_name, categorical_feature=categorical_feature)

    # Predict training set:
    dev_pred = mld.predict(x_dev[predictors])
    dev_prob = mld.predict_proba(x_dev[predictors])

    # Predict oot set:
    oot_pred = mld.predict(x_oot[predictors])
    oot_prob = mld.predict_proba(x_oot[predictors])

    # Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(mld, x_dev[predictors], y_dev, cv=cv_folds, scoring='roc_auc')

    # Perform KS:
    dev_perf = pd.DataFrame({'bad': y_dev, 'prob': dev_prob[:, 1]})
    oot_perf = pd.DataFrame({'bad': y_oot, 'prob': oot_prob[:, 1]})

    ks_dev = ks_group_equal(dev_perf, 'bad', 'prob', 20, reverse=True).ks.max()
    ks_oot = ks_group_equal(oot_perf, 'bad', 'prob', 20, reverse=True).ks.max()

    # Print model report:
    print("\nModel Report")
    print( "Training Accuracy : %.4g | Testing Accuracy : %.4g" % (
    accuracy_score(y_dev.values, dev_pred), accuracy_score(y_oot.values, oot_pred)))
    print("Training AUC Score : %f | Testing AUC Score : %f" % (
    roc_auc_score(y_dev, dev_prob[:, 1]), roc_auc_score(y_oot, oot_prob[:, 1])))
    print("Training KS : %f | Testing KS : %f" % (ks_dev, ks_oot))

    #    print "Testing Accuracy : %.4g" % accuracy_score(y_oot.values, oot_pred)
    #    print "Testing AUC Score : %f" % roc_auc_score(y_oot, oot_prob[:,1])
    #    print "Testing KS : %f" % ks_oot

    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (
        np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(mld.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

























