'''
=============================================================================================
|        A general utility function to select the best model and tune parameters            |
|        for various classification and regression ML algorithms                            |
|                   By Huanwang Henry Yang  (2016-08-15)                                    |
=============================================================================================

How to use it?
1. import this module to your python code, for example: 

tune_path='C:/Users/hyang978/data-science/Pyth/ML-tune/'
sys.path.append(tune_path)
import ml_tune as tune

2. Tune a perticular model : 
2a. For classification,  use tune.tune_classifier(X_train, y_train, X_test, y_test)
2b. For regression,      use tune.tune_regressor(X_train, y_train, X_test, y_test)

3. Tune all the models : 
3a. For classification,   use tune.tune_classifier_all(X_train, y_train, X_test, y_test)
3b. For regression,       use tune.tune_regressor_all(X_train, y_train, X_test, y_test)

'''
#===================================================================================
#
# -------------below for regression-------------
from  sklearn.ensemble import RandomForestRegressor
from  sklearn.ensemble import ExtraTreesRegressor
#from  sklearn.ensemble.weight_boosting import AdaBoostRegressor
from  sklearn.ensemble import AdaBoostRegressor
#from  sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from  sklearn.ensemble import GradientBoostingRegressor
from  sklearn.neighbors import KNeighborsRegressor
from  sklearn.neighbors import RadiusNeighborsRegressor #
from  sklearn.linear_model import LinearRegression
from  sklearn.linear_model import SGDRegressor
from  sklearn.linear_model import Ridge
from  sklearn.linear_model import Lasso
from  sklearn.linear_model import ElasticNet
from  sklearn.linear_model import BayesianRidge
from  sklearn.tree import DecisionTreeRegressor
from  sklearn.neural_network import MLPRegressor
from  sklearn.svm import SVR
from  sklearn.svm import LinearSVR
from  xgboost import XGBRegressor

# -------------below for classification-----------------
from  sklearn.ensemble import RandomForestClassifier
from  sklearn.ensemble import AdaBoostClassifier
from  sklearn.ensemble import GradientBoostingClassifier
from  sklearn.ensemble import ExtraTreesClassifier
from  xgboost import XGBClassifier
from  sklearn.tree import DecisionTreeClassifier
from  sklearn.neighbors import KNeighborsClassifier #often used
from  sklearn.neighbors import RadiusNeighborsClassifier  # if data not uniformly sampled
from  sklearn.linear_model import LogisticRegression
from  sklearn.linear_model import Perceptron
from  sklearn.linear_model import SGDClassifier
from  sklearn.svm import LinearSVC
from  sklearn.svm import SVC
from  sklearn.naive_bayes import GaussianNB
from  sklearn.naive_bayes import BernoulliNB  #for binary/boolean features
from  sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from  sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from  sklearn.gaussian_process import GaussianProcessClassifier
from  sklearn.neural_network import MLPClassifier
#from imblearn.ensemble import BalancedRandomForestClassifier  #for imbalance data

# library below is used for generating the results and validations ...
from sklearn.metrics import roc_auc_score   #for classifier
from sklearn.metrics import classification_report   #for classifier
from sklearn.metrics import confusion_matrix   #for classifier
from sklearn.metrics import accuracy_score   #for classifier
from sklearn.metrics import mean_squared_error, r2_score  #for regressor

from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import time

from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced


import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#===================================================================================
####----------------------------------------
def tune_classifier_all(X_train, y_train, X_test, y_test,tune):
    ''' Tuning ML algorithms for categorical data  (classification)
    X_* , y_* : the training and test data set. return result as list.
    tune: a key to indicate tune (=1) or not tune (=0) the hyper-parameters
    '''

    classifier=[
     'RandomForestClassifier'
 #   ,'BalancedRandomForestClassifier'
    ,'AdaBoostClassifier'
    ,'GradientBoostingClassifier'
    ,'ExtraTreesClassifier'
    ,'XGBClassifier'
    ,'DecisionTreeClassifier'
    ,'KNeighborsClassifier'
    ,'LogisticRegression'
    ,'GaussianNB'
    ,'BernoulliNB'
    ,'LinearDiscriminantAnalysis'
    ,'MLPClassifier'
    ,'LinearSVC'   
#    ,'QuadraticDiscriminantAnalysis'
#    ,'SVC'   #takes too much memory for large files   
#    ,'Perceptron'
#    ,'GaussianProcessClassifier'  #take all the memory for large data set! remove it
#    ,'RadiusNeighborsClassifier'  #
     ]
    
#    classifier=['KNeighborsClassifier','ExtraTreesClassifier']
    
    all_result=[]
    for model in classifier: 
        result=tune_classifier(model, X_train, y_train, X_test, y_test,tune)
        all_result.append(result)
        
    best=sorted(all_result, key=lambda x: x[2], reverse=True)
    columns=['model_inp', 'train_score','test_score','cpu','para','model']
    df_results = pd.DataFrame(best, columns=columns)
  
    print('==================================================================')
    print('\nThe table for each model\n', df_results.iloc[:,0:4], '\n')
    print('\nThe best model=', df_results.iloc[0,5])

#--------------------------------------------------
def tune_regressor_all(X_train, y_train, X_test, y_test,tune):
    ''' Tuning ML algorithms for numerical data 
    X_* , y_* : the training and test data set. return result as list.
    tune: a key to indicate tune (=1) or not tune (=0) the hyper-parameters

    '''

    regressor=[
    'RandomForestRegressor'
   ,'AdaBoostRegressor'
   ,'GradientBoostingRegressor'
   ,'ExtraTreesRegressor'
   ,'XGBRegressor'
   ,'DecisionTreeRegressor'
   ,'KNeighborsRegressor'
   ,'MLPRegressor'
   ,'LinearRegression'
   ,'Ridge'
   ,'Lasso'
   ,'ElasticNet'
   ,'BayesianRidge'
#   ,'SVR'
#   ,'LinearSVR'
#   ,'SGDRegressor'

    ]
    
    all_result=[]
    for model in regressor: 
        result=tune_regressor(model, X_train, y_train, X_test, y_test,tune)
        all_result.append(result)
        
    best=sorted(all_result, key=lambda x: x[2], reverse=True)
    columns=['model_inp', 'train_score','test_score','cpu','para','model']
    df_results = pd.DataFrame(best, columns=columns)
  
    print('==================================================================')
    print('\nThe table for each model\n', df_results.iloc[:,0:4], '\n')
    print('\nThe best model:\n model=', df_results.iloc[0,5])

####-------------------------------------------------
def tune_regressor(model, X_train, y_train, X_test, y_test, tune=1):
    '''Using *_train, tuning various popular ML algorithm for regression
    X_* , y_* : the training and test data set. return result as list.
    tune: a key to indicate tune (=1) or not tune (=0) the hyper-parameters
    '''
    
    time_start = time.time()
    print('----------------------------------------------------------------------------\n')   
    print( '\nTuning hyperparameters for ', model)
    
    if model=='RandomForestRegressor':
       hyper_para=dict(criterion=['mse', 'mae'], max_depth=[8,6,None], n_estimators=[150],
                  max_features=['auto','sqrt'])

       mod = RandomForestRegressor()

    elif model=='ExtraTreesRegressor':
       hyper_para=dict(criterion=[ 'mse','mae'], max_depth=[8,6, None], n_estimators=[150],
                  max_features=['auto','sqrt'])

       mod = ExtraTreesRegressor()

    elif model=='AdaBoostRegressor':
       hyper_para=dict(n_estimators=[150],learning_rate=[1.0,0.7], loss=['linear', 'square', 'exponential'])

       mod = AdaBoostRegressor()

    elif model=='GradientBoostingRegressor':
       hyper_para=dict(n_estimators=[150],learning_rate=[0.1,0.05], loss=['ls', 'lad', 'huber'],
                  max_features= [ 'auto','sqrt'])

       mod = GradientBoostingRegressor()

    elif model=='DecisionTreeRegressor':
       hyper_para=dict(splitter=['best', 'random'], max_depth=[5,4,3,2,None], max_features=
                  ['auto','sqrt'])
       mod = DecisionTreeRegressor()

    elif model=='KNeighborsRegressor':
       hyper_para=dict(n_neighbors=list(range(1, 30)), weights=['uniform', 'distance'])
       mod = KNeighborsRegressor()  #by default
   
    elif model=='MLPRegressor':
       hyper_para=dict(solver=['lbfgs', 'sgd', 'adam'])
       mod = MLPRegressor()

    elif model=='SGDRegressor':
       hyper_para=dict(loss=['squared_loss', 'huber'])
       mod = SGDRegressor()

    elif model=='LinearRegression':
       hyper_para=dict()
       mod = LinearRegression()  #by default

    elif model=='Ridge':
       hyper_para=dict(solver=['auto'], alpha=[1.0, 2.0])
       mod = Ridge()

    elif model=='Lasso':
       hyper_para=dict(alpha=[1.0])
       mod = Lasso()
    
    elif model=='ElasticNet':
       hyper_para=dict( l1_ratio=[0, 0.3, 0.5, 0.7, 1.0])
       mod = ElasticNet()    

    elif model=='BayesianRidge':
       hyper_para=dict()
       mod = BayesianRidge()    

    elif model=='SVR':
       hyper_para=dict()
       mod = SVR()
    
    elif model=='LinearSVR':
       hyper_para=dict()
       mod = LinearSVR()
 
    elif model=='XGBRegressor':
       hyper_para=dict()
       mod = XGBRegressor()
        
    if tune==0 : hyper_para=dict()
    grid = do_GridSearchCV(mod, X_train, y_train, X_test, y_test, hyper_para, 'reg')

#    plot_learning_curve(grid, "{}".format(model), X_train, y_train, ylim=(0.75,1.0), cv=5)   
        
    time_end = time.time()
    time_dif = time_end - time_start
    
    best_train_score = grid.score(X_train, y_train)
    best_test_score = grid.score(X_test, y_test)
        
    print('\nbest_train_score={tr:.3f}: best_test_score={tt:.3f} : CPU time= {t:.2f} s'.format(tr=grid.best_score_,tt= best_test_score, t=time_dif))
    print('best_params=', grid.best_params_)
    print('model=', grid.best_estimator_)
    
    result=[model, best_train_score, best_test_score, time_dif, grid.best_params_, grid.best_estimator_]
    return result
 
####------------------------------------------------
def tune_classifier(model, X_train, y_train, X_test, y_test, tune=1):
    ''' Using *_train, tuning various popular ML algorithm for classification 
    X_* , y_* : the training and test data set. return result as list
    tune: a key to indicate tune (=1) or not tune (=0) the hyper-parameters    
    '''
    print('\n------------------------------------------------------------------')           
    print( '\nTuning hyperparameters for ', model)
    
    time_start = time.time()
    if model=='RandomForestClassifier':
       h_para=dict(criterion=['gini', 'entropy'], max_depth=[6,4,2, None], n_estimators=[200],
              max_features=[ 'auto',None])
       mod=RandomForestClassifier()
       
    elif model=='BalancedRandomForestClassifier':
       h_para=dict(criterion=['gini', 'entropy'], max_depth=[6,4,2, None], n_estimators=[200],
              max_features=[ 'auto',None])
       mod=BalancedRandomForestClassifier()

    elif model=='ExtraTreesClassifier':
       h_para=dict(criterion=['gini', 'entropy'], max_depth=[6,4,None], n_estimators=[200],
                  max_features=[ 'auto',None])
       mod=ExtraTreesClassifier()

    elif model=='XGBClassifier':
       h_para=dict(max_depth=[2,5,7, 9],subsample=[1],n_estimators=[200],
                  colsample_bytree=[ 1])
       mod = XGBClassifier( )
        
    elif model=='AdaBoostClassifier':
       h_para=dict(n_estimators=[200], algorithm=['SAMME', 'SAMME.R'])
       mod = AdaBoostClassifier()

    elif model=='GradientBoostingClassifier':
       h_para=dict(n_estimators=[200],learning_rate=[1.0], loss=['deviance'],
                  max_features= ['auto',None], max_depth=[5,4,3])
       mod = GradientBoostingClassifier()

    elif model=='DecisionTreeClassifier':
       h_para=dict(splitter=['best', 'random'], max_depth=[5,4,3,None], max_features=
                  ['auto', None])
       mod = DecisionTreeClassifier()

    elif model=='KNeighborsClassifier':
       h_para = dict(n_neighbors=[3,5,7,9,11,13,15], weights=['uniform', 'distance'])
       mod=KNeighborsClassifier()  #by default

    elif model=='RadiusNeighborsClassifier':
       h_para = dict(radius=[0.5, 1, 2, 5], weights=[ 'distance'])
       mod=RadiusNeighborsClassifier()  #by default

    elif model=='LogisticRegression':
       h_para=dict(penalty=['l2'], class_weight=[None, 'balanced'], 
                  solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
       mod = LogisticRegression()

    elif model=='LinearDiscriminantAnalysis':
       h_para=dict(solver=['svd', 'lsqr', 'eigen'])
       mod = LinearDiscriminantAnalysis()

    elif model=='SVC':
       h_para=dict(kernel=['linear', 'poly', 'rbf'], decision_function_shape=[ 'ovo', 'ovr'],
                      Cs = [0.001, 0.01, 0.1, 1, 10],gammas = [0.001, 0.01, 0.1, 1],
                        param_grid = {'C': Cs, 'gamma' : gammas})
       mod = SVC()

    elif model=='LinearSVC':
       h_para=dict(penalty=[ 'l2'], loss=['hinge', 'squared_hinge'])
       mod = LinearSVC()

    elif model=='BernoulliNB':
       h_para=dict()
       mod = BernoulliNB()

    elif model=='GaussianNB':
       h_para=dict()
       mod = GaussianNB()

    elif model=='Perceptron':
       h_para=dict()
       mod = Perceptron()

    elif model=='SGDClassifier':
       h_para=dict(loss=['hinge', 'log', 'modified_huber'], penalty=['l2', 'l1', 'elasticnet'])
       mod = SGDClassifier()

    elif model=='QuadraticDiscriminantAnalysis':
       h_para=dict()
       mod = QuadraticDiscriminantAnalysis()
 
    elif model=='GaussianProcessClassifier':
       h_para=dict()
       mod = GaussianProcessClassifier()

    elif model=='MLPClassifier':
       h_para=dict(solver=['lbfgs', 'sgd', 'adam'])
       mod = MLPClassifier()
        
    if tune==0 : h_para=dict()
    grid = do_GridSearchCV(mod, X_train, y_train, X_test, y_test, h_para,'class')

#    plot_learning_curve(grid, "{}".format(model), X_train, y_train, ylim=(0.75,1.0), cv=10)   
        
    time_end = time.time()
    time_dif = time_end - time_start
    
    best_train_score = grid.score(X_train, y_train)
    best_test_score = grid.score(X_test, y_test)
        
    print('\nbest_train_score={tr:.3f}: best_test_score={tt:.3f} : CPU time= {t:.2f} s'.format(tr=grid.best_score_,tt= best_test_score, t=time_dif))
    print('best_params=', grid.best_params_)
    print('model=', grid.best_estimator_)
    
    result=[model, best_train_score, best_test_score, time_dif, grid.best_params_, grid.best_estimator_]
    return result

####-------------------------------------------------
def do_GridSearchCV(model, X_train, y_train, X_test, y_test, param_grid, type):
    '''model: the given model (regressor or clasifier)
    X_: a data frame containing all the features (excep target) (train | test)
    y_: the target (or class) (train | test)
    param_grid: a dictionary containing each hyper-parameter
    type: reg (for regressor) or class (classification)
    '''
    
    run_param=param_grid 
    if len(param_grid)==0 : run_param=dict()  #include default
 
    print('input_params=', run_param)
    if type == 'reg' :  #regression
        grid = GridSearchCV(model, run_param, cv=4,  n_jobs=-1) #use all cpu
    elif type =='class': #classification
        grid = GridSearchCV(model, run_param, cv=4, scoring='accuracy', n_jobs=-1) 
    else:
        print( 'Error: please indicate "reg" or "class" for the GridSearchCV.')
        return 
    estimator=grid.fit(X_train, y_train)

    return estimator

####-------------------------------------------------
 
####-------------------------------------------------
def generate_RF_Hpram():
# Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
    max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
# Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
    bootstrap = [True, False]
# Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    pprint(random_grid)
    
####-------------------------------------------------
def run_RandomizedSearchCV(model, X, Y, param_grid, type):
    '''model: the given model (regressor such as RandomForestRegressor() or clasifier)
    X: a data frame containing all the features (excep target)
    Y: the target
    param_grid: a dictionary containing each hyper-parameter
    type: reg (for regressor) or class (classification)
    '''
    from sklearn.model_selection import RandomizedSearchCV
#The most important arguments in RandomizedSearchCV are n_iter, which controls the number of different #combinations to try, and cv which is the number of folds to use for cross validation    
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = model, param_distributions = param_grid,  
                                   verbose=2, n_iter = 100, cv = 3, random_state=42, n_jobs = -1)
# Fit the random search model
    rf_random.fit(X, Y)
#get the best parameters
#  rf_random.best_estimator_
#====================================    
def evaluate(model, test_features, test_labels):
    #test_features ; test_X; (The features)
    #test_labels ; test_Y   (The target)
    
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

    
    
##==========================================================================
##==========================================================================


dict_classifiers = {
    "Logistic Regression": 
            {'classifier': LogisticRegression(),
                'params' : [
                            {
                             'penalty': ['l1','l2'],
                             'C': [0.001,0.01,0.1,1,10,100,1000]
                            }
                           ]
            },
    "Nearest Neighbors": 
            {'classifier': KNeighborsClassifier(),
                 'params': [
                            {
                            'n_neighbors': [1, 3, 5, 10],
                            'leaf_size': [3, 30]
                            }
                           ]
            },
             
    "Linear SVM": 
            {'classifier': SVC(),
                 'params': [
                            {
                             'C': [1, 10, 100, 1000],
                             'gamma': [0.001, 0.0001],
                             'kernel': ['linear']
                            }
                           ]
            },
    "Gradient Boosting Classifier": 
            {'classifier': GradientBoostingClassifier(),
                 'params': [
                            {
                             'learning_rate': [0.05, 0.1],
                             'n_estimators' :[50, 100, 200],
                             'max_depth':[3,None]
                            }
                           ]
            },
    "Decision Tree":
            {'classifier': DecisionTreeClassifier(),
                 'params': [
                            {
                             'max_depth':[3,None]
                            }
                             ]
            },
    "Random Forest": 
            {'classifier': RandomForestClassifier(),
                 'params': {}
            },
    "Naive Bayes": 
            {'classifier': GaussianNB(),
                 'params': {}
            }
}

#============================================================
from sklearn.model_selection import learning_curve 
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.6, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

#=============================================
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score
num_classifiers = len(dict_classifiers.keys())

def batch_classify(X_train, Y_train, X_test, Y_test, verbose = True):
    df_results = pd.DataFrame(
        data=np.zeros(shape=(num_classifiers,4)),
        columns = ['classifier',
                   'train_score', 
                   'test_score',
                   'training_time'])
    count = 0
    for key, classifier in dict_classifiers.items():
        t_start = time.time()
        grid = GridSearchCV(classifier['classifier'], 
                      classifier['params'],
                      refit=True,
                        cv = 10, # 9+1
                        scoring = 'accuracy', # scoring metric
                        n_jobs = -1
                        )
        estimator = grid.fit(X_train,
                             Y_train)
        t_end = time.time()
        t_diff = t_end - t_start
        train_score = estimator.score(X_train,
                                      Y_train)
        test_score = estimator.score(X_test,
                                     Y_test)
        df_results.loc[count,'classifier'] = key
        df_results.loc[count,'train_score'] = train_score
        df_results.loc[count,'test_score'] = test_score
        df_results.loc[count,'training_time'] = t_diff
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=key,
                                                    f=t_diff))
        count+=1
        plot_learning_curve(estimator, 
                              "{}".format(key),
                              X_train,
                              Y_train,
                              ylim=(0.75,1.0),
                              cv=10)
    return df_results

