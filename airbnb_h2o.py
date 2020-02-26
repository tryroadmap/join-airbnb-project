
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import confusion_matrix, make_scorer, classification_report, cohen_kappa_score, accuracy_score, average_precision_score, roc_auc_score


import warnings

import itertools


import h2o
from h2o.estimators import H2OGradientBoostingEstimator


from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch
#from __future__ import print_function
import datetime

h2o.init(ip="127.0.0.1" , port = 5151, nthreads= -1, max_mem_size="25g")

h2o.remove_all


# Import a sample binary outcome train/test set into H2O
train_ka = h2o.import_file("data_train_c0.csv")
test_ka = h2o.import_file("data_valid_c0.csv")
test_ka_ = h2o.import_file("data_test_c0.csv") #for the final evaluation
# features and target
target = "dim_is_requested"

train_ka[target] = train_ka[target].asfactor()
test_ka[target] = test_ka[target].asfactor()
test_ka_[target] = test_ka_[target].asfactor()


train_columns_x1=[
'dim_is_requested',
'ds_night',
'id_listing_anon',
'ds_night_day_of_year', #bool
'm_effective_daily_price',
'm_pricing_cleaning_fee',
'dim_person_capacity', #num
'dim_is_instant_bookable', #bool
'm_checkouts',  #num
'ds_checkin_gap', #num 0-6
'days_since_last_booking', #num
'cancel_policy', #cat 3-9
'image_quality_score', #num
'occ_occupancy_plus_minus_14_ds_night', #num
'm_total_overall_rating', #nu m
'dim_has_wireless_internet',
'general_market_m_unique_searchers_0_6_ds_night',#num
'general_market_m_contacts_0_6_ds_night',#num
'p2_p3_click_through_score', #num
'general_market_m_is_booked_0_6_ds_night',
'kdt_score',
'r_kdt_listing_views_0_6_avg_n100',
'r_kdt_n_available_n100',
] #bool


#create feature for X3, X4
# create features for model 1
train_columns_x2 = [
'dim_is_requested',
'ds_night_day_of_week', #bool
'id_listing_anon',
'm_reviews', #num
'ds_checkout_gap', #num 0-6
'occ_occupancy_plus_minus_7_ds_night', #num
'occ_occupancy_trailing_90_ds', #num
'm_minimum_nights', #num
'price_booked_most_recent', #num
'p3_inquiry_score',  #num
'm_professional_pictures', #num
'listing_m_listing_views_2_6_ds_night_decay', #num
'general_market_m_reservation_requests_0_6_ds_night',#num
'm_available_listings_ds_night',
'r_kdt_n_active_n100',
'r_kdt_m_effective_daily_price_n100_p50',
'r_kdt_m_effective_daily_price_booked_n100_p50'
]



nfolds = 10 # Number of CV folds (to generate level-one data for stacking) 10, 100
n_trees_ = 120

x = train_columns_x1
y = target
x.remove(y)


def KAXGBoostEstimator(feats = x, label = y , train_ka = train_ka, n_trees= 25 , nfolds = 5):
    # Train and cross-validate a GBM
    encoding = "one_hot_explicit"
    my_gbm = H2OGradientBoostingEstimator(distribution="bernoulli",
                                          ntrees=n_trees,
                                          max_depth=3,
                                          min_rows=2,
                                          learn_rate=0.2,
                                          nfolds=nfolds,
                                          fold_assignment="Modulo",
                                          keep_cross_validation_predictions=True,
                                          categorical_encoding = encoding,
                                          seed=1234)
    my_gbm.train(x=x, y=y, training_frame=train_ka)
    print(n_trees)
    return(my_gbm)

def KARandomForestEstimator(feats = x, label = y , train = train_ka, n_trees= 25 , nfolds = 5):
    """
    # notes@ n_trees, #from basemodel vs ntree plot recommend running for 250
    # encodoing on rf -overfitting
    """
    encoding = "one_hot_explicit"
    my_rf = H2ORandomForestEstimator(ntrees = n_trees,
                                     nfolds=nfolds,
                                     fold_assignment="Modulo",
                                     keep_cross_validation_predictions=True,
                                     categorical_encoding = encoding,
                                     seed=1234)
    my_rf.train(x=feats, y=label, training_frame=train)
    print(n_trees)
    return(my_rf)


def KAStackdEsnsumbleEstimator(modellist, nfolds = nfolds, stack_id="ka", train = train_ka, ensemble_tag ="ka"):
    print("starting time:")
    ensemble = H2OStackedEnsembleEstimator(model_id=("airbnb_ensemble_binomial_" + ensemble_tag),
                                           base_models = modellist)
    ensemble.train(x=x, y=y, training_frame=train)
    return(ensemble)


my_gbm_x1 = KAXGBoostEstimator(n_trees = n_trees_ ,nfolds = nfolds )
model_path = h2o.save_model(model = my_gbm_x1, path="/home/scv/airbnb_challenge/balancedx1x2ntree200", force=True)

#my_rf_x1 = KARandomForestEstimator(n_trees = n_trees_, nfolds = nfolds)
#model_path = h2o.save_model(model = my_rf_x1, path="/home/scv/airbnb_challenge/balancedx1x2ntree200", force=True)



x = train_columns_x2
y = target
x.remove(y)

my_gbm_x2 = KAXGBoostEstimator(n_trees = n_trees_, nfolds = nfolds)
model_path = h2o.save_model(model = my_gbm_x2, path="/home/scv/airbnb_challenge/balancedx1x2ntree200", force=True)

my_rf_x2 = KARandomForestEstimator(n_trees = n_trees_ , nfolds = nfolds)
model_path = h2o.save_model(model = my_rf_x2, path="/home/scv/airbnb_challenge/balancedx1x2ntree200", force=True)



modellist = [my_gbm_x1, my_rf_x1, my_gbm_x2, my_rf_x2]
my_ensemble = KAStackdEsnsumbleEstimator(modellist)

def KAPerformanceEvaluator(ensemble =  my_ensemble , test = test_ka, test_= test_ka):
    # Eval ensemble performance on the test data
    perf_stack_test = ensemble.model_performance(test)

    # Compare to base learner performance on the test set
    perf_gbm_test = my_gbm.model_performance(test)
    perf_rf_test = my_rf.model_performance(test)
    baselearner_best_auc_test = max(perf_gbm_test.auc(), perf_rf_test.auc())
    stack_auc_test = perf_stack_test.auc()
    print("Best Base-learner Test AUC:  {0}".format(baselearner_best_auc_test))
    print("Ensemble Test AUC:  {0}".format(stack_auc_test))
    # Generate predictions on the final test set
    pred = ensemble.predict(test_)
    ensemble
    return(ensemble,pred)



#ensemble

KAPerformanceEvaluator(ensemble = my_ensemble)
