'''
----------------------------------------------------------------------------------------------------
This is a general utility function to perform various tasks for using ML algorithms
----------------------------------------------------------------------------------------------------
By Huanwang Henry Yang  (2017-06-12)
'''

import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy import stats

from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import  confusion_matrix, classification_report
from sklearn.metrics import  roc_auc_score, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

#=============================================================================
# ============== below are functions for data exploration  ===================
#=============================================================================
data_missiing_info= '''
  1. df.fillna(df.mode().iloc[0]) for categorical data (all columns)
  2. df.fillna(df.mean()) for numerical data (all columns)
  3. remove the rows with NaN. Could be a biased if too many missing.
  4. remove the columns with many NaN (e.g. >50%). Could be an important feature.
  5. use some features to predict the missing values (better if randomly missing)
     a). I used RF for prediction (see functions below)
     b). use kNNmputer for prediction (see functions below)
  6. give a unique name for the missing values (if categorical)
     ex.   df.loc[df.country.isna(), 'country']='contry_na'
  
  '''
#-----------------------------------------------------------------------
def df_info(df):
    '''df_info(df): check df for basic info: missing value, type, unique value
    '''
    variable_name = []
    total_value = []
    total_missing_value = []
    missing_value_rate = []
    unique_value_list = []
    total_unique_value = []
    df_type = []

    for col in df.columns:
        dt=df[col].dtype
        
        variable_name.append(col)
        df_type.append(dt)
        total_value.append(df[col].shape[0])
        total_missing_value.append(df[col].isnull().sum())
        missing_value_rate.append(round(df[col].isnull().sum()/df[col].shape[0],4))
        unique_value_list.append(df[col].unique())
        total_unique_value.append(len(df[col].unique()))
        
    dic={"Variable":variable_name, "#_Total":total_value, "#_Missing":total_missing_value,\
         "%_Missing":missing_value_rate, "Unique_Value":unique_value_list, "#_Unique_Value":total_unique_value, \
         "df_Type":df_type}   
    
    missing_df=pd.DataFrame(dic)
    
    missing_df = missing_df.set_index("Variable")
    return missing_df.sort_values("#_Missing",ascending=False)

#-----------------------------------------------------------------------
def df_info_all(df):
    '''df_info(df): check df for basic info: missing value, type, unique value
    '''
    variable_name = []
    total_value = []
    total_missing_value = []
    missing_value_rate = []
    unique_value_list = []
    total_unique_value = []
    df_type = []
    min1=[]  # last one if it is object
    max1=[]  # top one if it is object
    mean1=[] # mode if it is object
    std1=[]
    
    df_des=df.describe(include='all')
    for col in df.columns:
        dt=df[col].dtype
        
        variable_name.append(col)
        df_type.append(dt)
        total_value.append(df[col].shape[0])
        total_missing_value.append(df[col].isnull().sum())
        missing_value_rate.append(round(df[col].isnull().sum()/df[col].shape[0],4))
        unique_value_list.append(df[col].unique())
        total_unique_value.append(len(df[col].unique()))
        
        if  dt=='int64' or dt=='float64': #numerical
            min1.append(df_des.loc['min', col])
            max1.append(df_des.loc['max', col])
            mean1.append(df_des.loc['mean', col])
            std1.append(df_des.loc['std', col])
        elif dt=='O':  #object
            vc=df[col].value_counts()
            min1.append(vc.index[-1])
            max1.append(vc.index[0])
            mean1.append(df[col].mode().values[0])
            std1.append(vc.values.std())  
        elif dt=='<M8[ns]':  #date
            vc=df[col]
            min1.append(df[col].min())
            max1.append(df[col].max())
            mean1.append(df[col].mean())
            std1.append('NA')
            
        else:
            min1.append('NA')
            max1.append('NA')
            mean1.append('NA')
            std1.append('NA')
        
    dic={"Variable":variable_name, "#_Total":total_value, "#_Missing":total_missing_value,\
         "%_Missing":missing_value_rate, "Unique_Value":unique_value_list, "#_Unique_Value":total_unique_value, \
         "df_Type":df_type, "min/last":min1,"max/top":max1,"mean/mode":mean1,"std":std1 }   
    
    missing_df=pd.DataFrame(dic)
    
    missing_df = missing_df.set_index("Variable")
    return missing_df.sort_values("#_Missing",ascending=False)

##-----------------------------------------------------------------------
def split_data(df_in, size=0.25, target='', stratify='yes'):
    ''' randomly split the data into training and testing part
    
    '''
    
    df_final=df_in.copy()
    if len(target)==0: 
        print("Error: please specify the target (class) name.")
        return 

    X=df_final.drop([target], axis=1)
    y=df_final[target]
    if stratify=='yes':
        X_train0, X_test0, y_train0, y_test0 = train_test_split(X, y, test_size=size, random_state=42,stratify=y)
    else:
        X_train0, X_test0, y_train0, y_test0 = train_test_split(X, y, test_size=size, random_state=42)
        

    print ('shape of X={}, X_train={},  X_test={}'.format(X.shape, X_train0.shape, X_test0.shape) )
    
    return X_train0, X_test0, y_train0, y_test0

#-----------------------------------------------------------------------
def df_check(df, outlier_cutoff=3, corr_cutoff=0.9, var_cutoff=0.001):
    ''' A function to report details about the dataframe (df)
    '''
  
  #check different type of data types
    dtype=[df[x].dtypes for x in df.columns ]
    print('The unique dtype in DF=',set(dtype))

  #split the DF into various types
    df_num=df.select_dtypes(include='number')
    df_obj=df.select_dtypes(include='object')
    df_cat=df.select_dtypes(include='category')
    df_time=df.select_dtypes(include=[np.datetime64]) #or (include=np.datetime64))
    
 # df_dtime=df.select_dtypes(include= [np.timedelta])  #,np.timedelta64]
    print('\nshape of df_num=', df_num.shape)
    print('shape of df_obj=', df_obj.shape)
    print('shape of df_cat=', df_cat.shape)
    print('shape of df_time=', df_time.shape)

  #check missing values (all), return a DF
    dd1=check_missing_values(df)
    print('\nNumber of columns with missing values=', dd1.shape[0])
    if (len(dd1)>0) : print(dd1)
    print("\n")

  #check outliers (return a DF) 
    dd2a=find_outliers(df_num, method='iqr', cutoff=outlier_cutoff)
    if dd2a.shape[0]>0:
      print(f'The outliers {outlier_cutoff}*IQR (InterQuartile Range)')  
      print(dd2a,"\n")  

    dd2b=find_outliers(df_num, method='std', cutoff=outlier_cutoff)
    if dd2b.shape[0]>0:
      print(f'The outliers {outlier_cutoff}*std (standard deviation)')  
      print( dd2b)  

  #check related columns using the correation
    corr_col=find_corr_columns(df_num, corr_cutoff)
    if len(corr_col)>0:
      d=df_num.corr()[corr_col]
      dd3=d[d>corr_cutoff].dropna(how='all')
      print(f'The corrlation (threshold>{corr_cutoff}) table:', '\n', dd3)
      
  #check columns with constant values or very small variances
    const_col=find_const_columns(df, var_threshold=var_cutoff)

  #check if any duplicated rows 
    print('\n', f'Number of duplicated rows={len(df[df.duplicated()])}')
    
#-----------------------------------------------------------------------
def check_target(df, col='', target=''):
  '''check how the target is distributed in each level
  '''
  d1=df[df[target]==0][col].value_counts().to_frame()
  d2=df[df[target]==1][col].value_counts().to_frame()

#pd.merge(d1, d2, left_index=True, right_index=True)
  dd=pd.concat([d1,d2], axis=1)
  dd.columns=[col+'_no', col+'_yes']
  dd['ratio_yes/no'] = dd[col+'_yes']/dd[col+'_no']

  return dd
  
#-----------------------------------------------------------------------
def assign_other(df1, col='', val=0.01, action='keep', name='other'):
    ''' assign new name (or remove) categorical items with low value level. 
    '''
    
    df=df1.copy()
    for x in col:   
        d=df[x].value_counts()
        ind=d.index.to_list()
        v=list(d.values)
        s=sum(v)
        ddf=pd.DataFrame({'item':ind, 'val':v, 'p':v/s})
        if action=='drop':
            lls=ddf[ddf.p>val]['item'].values
            df=df[df[x].isin(lls)]
            print(f'{x}: Number of levels removed={len(ddf)-len(lls)}')
        else:
            lls=ddf[ddf.p<=val]['item'].values
            df[x]=df[x].apply(lambda y : name if y in lls  else y)
            print(f'{x}: Number of levels renamed={len(ddf)-len(lls)}')
            
    return df
#-----------------------------------------------------------------------
def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
#-----------------------------------------------------------------------
def find_non_numeric(ss, keep_na=1):
  '''find_non_numeric(ss,keep_na=0) : find rows excluding the numbers (2. 2.3, 4)
  ss: an array like or a series
  keep_na: =1, keep the np.nap in the list, =0 exclude nap
  '''
  if keep_na==1 :
    return ss[~ss.apply(lambda x: str(x).replace('.','',1).isdigit())]
  else:
    return ss[~ss.apply(lambda x: is_number(x))] #include np.nan
    
#-----------------------------------------------------------------------
def impute_by_knn(df, target=False, n_neighbor=5):
  '''impute missing data by kNN
  df is either a array or a df.  (prefered)
  '''

  from sklearn.impute import KNNImputer

  imputer = KNNImputer(n_neighbors=n_neighbor, weights="uniform")
  if not target :   #warning: may be biased
    print('Imputing data based on all columns of DF')
    d=imputer.fit_transform(df)   #as numpy  
    return pd.DataFrame(d, columns=df.columns) #back to DF
  print('Imputing data without target')
  X=df.drop(target, axis=1)
  y=df[target]
  #d=imputer.fit_transform(np.array(df))
  d=imputer.fit_transform(X)   #as numpy
  
  df1=pd.DataFrame(d, columns=X.columns) #back to DF
  dfn=pd.concat([df1, y], axis=1)

  return dfn

#-----------------------------------------------------------------------
def impute_by_knn_validate(df, target):
  '''using the pipeline to see how impute perform.
  '''

  from sklearn.pipeline import Pipeline
  from sklearn.model_selection import RepeatedStratifiedKFold
  from sklearn.model_selection import cross_val_score
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.impute import KNNImputer
  import numpy as np

  X=df.drop(target, axis=1)
  y=df[target]
# define modeling pipeline
  model = RandomForestClassifier()
  imputer = KNNImputer()
  pipeline = Pipeline(steps=[('i', imputer), ('m', model)])
# define model evaluation
  cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# evaluate model
  scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
  print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

#-----------------------------------------------------------------------
def impute_num_RF(df, feature):
  '''  A function to predict the missing value by RF
  
  df: the dataframe containing features for classification
  feature:  The feature to be predicted
  '''
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.metrics import r2_score, mean_squared_error

  ddf=df.loc[~df[feature].isna()]  #a DF with feature exist values
  ddfNA=df.loc[df[feature].isna()]  #a DF with feature exist NaN

#get training  and testing DF
  Xtrain=ddf.drop([feature], axis=1) 
  ytrain=ddf[feature]
  Xtest=ddfNA.drop([feature], axis=1) 
  ytest=ddfNA[feature]

  model=RandomForestRegressor()  #use RF for continue
  model.fit(Xtrain, ytrain)  #fit the model 
  y_pred = model.predict(Xtest)

  ddfNA[feature]=y_pred
#  print("Value_counts for the predicted=", y_pred)

  print('---------Score by the training data set----------')
  pred = model.predict(Xtrain)
  write_result_regr(pred, ytrain, 'RandomForestRegressor')

  dfn=ddf.append(ddfNA)
  return dfn
  
#-----------------------------------------------------------------------
def impute_cate_xgb(df, feature):
  '''A function to predict the categorical value by xgb. Better than the 
  simple method df.fillna(df.mode().iloc[0]). 

  df: the dataframe containing features for classification
  feature:  The feature to be predicted
  '''
  from  xgboost import XGBClassifier

  ddf=df.loc[~df[feature].isna()]  #a DF with feature exist values
  ddfNA=df.loc[df[feature].isna()]  #a DF with feature exist NaN

#get training  and testing DF
  Xtrain=ddf.drop([feature], axis=1) 
  ytrain=ddf[feature]
  Xtest=ddfNA.drop([feature], axis=1) 
  ytest=ddfNA[feature]

  mod=XGBClassifier()  #use XGB, no need to do label encode
  mod.fit(Xtrain, ytrain)  #fit the model 
  y_pred = mod.predict(Xtest)

  return y_pred
  
#---------------------------------------------------------------------------
#this is another impute test
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

class imputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean',filler='NA'):
       self.strategy = strategy
       self.fill = filler

    def fit(self, X, y=None):
       if self.strategy in ['mean','median']:
           if not all(X.dtypes == np.number):
               raise ValueError('dtypes mismatch np.number dtype is \
                                 required for '+ self.strategy)
       if self.strategy == 'mean':
           self.fill = X.mean()
       elif self.strategy == 'median':
           self.fill = X.median()
       elif self.strategy == 'mode':
           self.fill = X.mode().iloc[0]
       elif self.strategy == 'fill':
           if type(self.fill) is list and type(X) is pd.DataFrame:
               self.fill = dict([(cname, v) for cname,v in zip(X.columns, self.fill)])
       return self

    def transform(self, X, y=None):
       return X.fillna(self.fill)
       
#----------------------------------------------------------
def find_outliers(df_in, method='std', cutoff=3):
  '''find_outliers(df_in, method='std', cutoff=3)
  list the number of outliers using boxplot and Zscore. 
  method='std' or 'iqr' (interquartile range)
  '''

  df_num=df_in.select_dtypes(include='number') #only for numerical

  val_all=[]
  for col_name in df_num.columns:  
    ddf=df_num[col_name].dropna()
    q1 = ddf.quantile(0.25)
    q3 = ddf.quantile(0.75)
    iqr = q3-q1 #Interquartile range
    
    if method=='std':
      fence_low = ddf.mean() - cutoff*ddf.std()
      fence_high = ddf.mean() + cutoff*ddf.std()
    elif method=='iqr':
      fence_low  = q1-cutoff*iqr
      fence_high = q3+cutoff*iqr

    n=0
    for val in ddf:
      if val>fence_high or val<fence_low : n=n+1
    if n >0 :
      val_all.append([col_name, fence_low, fence_high,ddf.min(), ddf.max(), n])
#      print('{}:   value beyond ({:.2f}  {:.2f})   outliers ={}: ' \
#      .format(col_name, fence_low, fence_high, n))

  df_col= ['col_name', 'fence_low', 'fence_high', 'min_val', 'max_val','num_outlier']  
  if len(val_all)>0 :
    return pd.DataFrame(val_all, columns=df_col).sort_values(by=['num_outlier'], ascending=False )
  else:
    print('Not found outliers under cutoff=', cutoff) 
    return ''

#find_outliers(dt, method='std', cutoff=3)

#----------------------------------------------------------

def remove_outliers_z(df_in, method='std', col=[], action='drop', cutoff=3):
    
    '''handle_outliers_std(df_in, col=[], std=3, action='drop')
    action=drop or keep
    cutoff: the cutoff value for the zscore or IQR.
    method = 'std' or 'iqr' 
    
    For the Zscore method:
    1. For each column, first it computes the Z-score of each value in the column, 
    relative to the column mean and standard deviation Z=(X-mean)/std.
    2. Take the abs value below the threshold.
    3. all(axis=1) ensures that for each row, all column satisfy the constraint.
    
    Warning: output may not be as desired, if over 90% zeros. To avoid the problem, it's 
    neccessary to drop/keep outlies by group. For example, do group 1 (col), give 
    zcutoff=3, then do group 2 (new col) and give zcutoff=4 ...
    
    '''
    from scipy import stats

    df=df_in.select_dtypes(include='number') #must be num
#    df_cat=df_in.select_dtypes(exclude='number') 
    
    if action=='drop' and method=='std':
        print("Removing outliers if Zscore >",cutoff)
        if len(col) > 0: df=df[col]
            
        dfn=df[(np.abs(stats.zscore(df)) < cutoff).all(axis=1)] #remove outliers using Zscore
        dfm=pd.concat([df_in, dfn], axis=1, join='inner')
        dfm = dfm.loc[:,~dfm.columns.duplicated()]
        return dfm
#----------------------------------------------------------
def handle_outliers(df_in, method='std', col=[], action='drop', cutoff=3):    
    '''handle_outliers_std(df_in, col=[], std=3, action='drop')
        action=drop or keep
        cutoff: the cutoff value for the zscore or IQR.
        method = 'std' or 'iqr' 
    
    For the Zscore method:
    1. For each column, first it computes the Z-score of each value in the column, 
    relative to the column mean and standard deviation Z=(X-mean)/std.
    2. Take the abs value below the threshold.
    3. all(axis=1) ensures that for each row, all column satisfy the constraint.
    
    Warning: 
    1. IQR will fail, if IQR=0 (over 80% const value) for both drop/keep outlier!!
    2. If too many const, use STD with high cutoff.
    Output may not be as desired, if over 90% zeros. To avoid the problem, it's 
    neccessary to drop/keep outlies by group. For example, do group 1 (col), give 
    zcutoff=3, then do group 2 (new col) and give zcutoff=4 ...
    
    '''
    from scipy import stats

    df=df_in.select_dtypes(include='number') #must be num
    df_cat=df_in.select_dtypes(exclude='number') 
    
    if len(col) > 0: 
        cols=col
        print("Handling outliers for the given columns.  Action=", action, ': cutoff=',cutoff )
    else:  
        cols=list(df.columns)
        print("Handling outliers for all the columns.  Action=", action, ': cutoff=',cutoff)
                
    #remvoe binary columns. #do not check (0, 1) flags.     
    col_flag =[]   
    for x in cols:
        ss=list(df[x].unique())
        if len(ss)==2 and (ss[0]==0 or ss[1]==0) and (ss[0]==1 or ss[1]==1):
            col_flag.append(x)  
#    if len(col_flag) >0 :    df.drop(col_flag, axis=1, inplace=True)
        
    print("Number of binary flags {:2d}: categorical {:2d}: Numerical {:2d} ".
          format(len(col_flag), df_cat.shape[1], df.shape[1]))
    
    for x in cols:
        iqr=df[x].quantile(.75) - df[x].quantile(.25)
        fence_low  = df[x].quantile(.25) - cutoff*iqr
        fence_high = df[x].quantile(.75) + cutoff*iqr
                        
        if action=='drop':
            if method=='std':
                df=df[(np.abs(stats.zscore(df[x])) < cutoff)]
            elif method=='iqr':
                df.drop(df[df[x]>fence_high].index, inplace=True)
                df.drop(df[df[x]<fence_low].index, inplace=True)
                
            print('shape after removing outlier for ', x, df.shape)
            
        elif action=='keep':  #use clip 
            if method=='std':
                fence_low = df[x].mean() - cutoff*df[x].std()
                fence_high = df[x].mean() + cutoff*df[x].std()
            
#            print(x, df.shape, 'before {:.4f} {:.4f}  {:.4f} {:.4f}  {:.4f} ' 
#                  .format(df[x].mean(),df[x].min(), df[x].max(), fence_low, fence_high))
            df[x].clip(lower=fence_low,upper=fence_high, inplace=True)
            
            print(x, df.shape, 'mean={:.2f}, min, max ({:.2f}, {:.2f}), fence_low & hign ({:.2f},{:.2f}) ' 
                  .format(df[x].mean(),df[x].min(), df[x].max(), fence_low, fence_high))
    '''
    if df_cat.shape[1]>0:  #
        df=pd.concat([df,df_cat], axis=1)  #make the input& output the same columns
    if len(col_flag) >0 :
        df=pd.concat([df,df_in[col_flag]], axis=1)  
    '''

        
    return df
#  dfn=df[(np.abs(stats.zscore(df)) < zcutoff).all(axis=1)]
# df=df[(np.abs(stats.zscore(df[x])) < zcutoff)]

#dd=handle_outliers(dt,  method='std', col=['admitscost', 'Admits','Beddays'], action='drop', cutoff=7)

#----------------------------------------------------------
def column_clean(dfin, old=[], new=[]):
    '''Clean the column names: column_clean(dfin, old=[], new=[]): 
       example:  old=['[', ')', '/', '>'], new= ['', '', '_', '_']
    '''
    df=dfin.copy()
    if len(old) ==0 or len(old) !=len(new):
      print('Zero length of input list: old=[..], new=[..]')

    all_cols = []
    for col in df.columns:
        for i in range(len(old)):
            col=col.replace(old[i], new[i])
        all_cols.append(col)
    df.columns = all_cols
    return(df)  
#-----------------------------------------------------------------------
def column_remove(dfin, cut=0.5):
    '''remove_column(df, cut): remove if missing value >= cut, or a const.
    df: the input dataframe;  dfn, the output dataframe
    '''
    df=dfin.copy()
    col_const=df.columns[~(df != df.iloc[0]).any()].values.tolist()
    print("\nColumns with constant variance: ", len(col_const),'\n', col_const)
    
    col_ok=df.columns[(df != df.iloc[0]).any()].values.tolist()
    df0=df[col_ok]
    
    d=df0.isna().mean()  #get a series
    df1=df0[d[d<cut].index.tolist()]  #new df after cut
    
    dd=d[d>=cut].index.tolist()
    print("\nColumns with missing values >", 100*cut,'%  :', len(dd), "\n", dd)
  
    return df1
#-----------------------------------------------
def check_missing_values(df):
  ''' A function to check any missing values, return a DF
  '''

  d=df.isna().any() #only check missing part
  d1=df[d[d].index].isna().sum()
  d2=df[d[d].index].isna().mean()
  dd=pd.DataFrame({'number_missing':d1.values, 'percent':d2*100})
  dd.sort_values(by=['percent'], ascending=False, inplace=True)

  return dd
    
#-----------------------------------------------------------------------
    
def find_corr_columns(df, threshold):
  '''check related columns using the correation. 
  df: a numerical dataframe or np
  threshold: the given threshold. Normally 0.9. if >0.9 then pick the columns
  '''

# Create correlation matrix
  corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than  threshold
  to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
  print('\nfeatures with correlation > threshold=',to_drop)
  
  return to_drop
    
#---------------------------------------------------------------------------
def find_const_columns(df, var_threshold=0.00001):
  '''remove columns if the column has constant variance (or <var_threshold)
  df: a numerical DF 
  var_threshold : =0 for constant columns;  
  '''
  from sklearn.feature_selection import VarianceThreshold
  
  col_const=df.columns[~(df != df.iloc[0]).any()].values.tolist()
  print("\nColumns with constant variance=", col_const)
  
  #further check if any column has variance < var_threshold
  df1=df.loc[:, (df != df.iloc[0]).any()]  #get DF with non-const columns
  df_num=df1.select_dtypes(include='number')
  select=VarianceThreshold(var_threshold)
  df2=select.fit_transform(df_num)

  feature_idx = select.get_support()
  feature_name = df_num.columns[feature_idx]
  to_drop=[x for x in df_num.columns.values if x not in feature_name.values]
 # to_drop = set(df_num.columns.values) - set(feature_name.values) 
#  print("shape for original DF, df_num", df.shape, df_num.shape)
  print("Columns of variance <", var_threshold,':', to_drop)
  col_all=to_drop + col_const
  return col_all

#-----------------------------------------------------------------------
def colum_length(df,  col_name):
  #count the document length. return a df
  d=df[col_name].apply(lambda x: len(x)).values
  return pd.DataFrame({'column_len':d}).column_len.value_counts()

#--------------------------------------------------------------------
def get_common_df(df1, df2) :
    '''get_common_df(df1, df2)
    combine two DF (by index), return the common columns 
    '''    
    
    col_len=len(list(set(list(df1.columns)) & set(list(df2.columns)))) 
    idx_len=len(list(set(list(df1.index)) & set(list(df2.index)))) 
    
    dd=pd.DataFrame({'column_length':[len(df1.columns),len(df2.columns), col_len], 
                 'index_length':[len(df1.index),len(df2.index), idx_len]}
                 , index=['DF1', 'DF2', 'DF1&DF2'])
    print(dd)
    dfm=pd.concat([df1, df2], axis=1, join='inner')
    dfm = dfm.loc[:,~dfm.columns.duplicated()]
    return dfm
#    list1=list(set(df1.columns) - set(df2.columns))

#=============================================================================
# ============== below are functions for modeling  ===================
#=============================================================================
modeling_info='''

'''

#-----------------------------------------------------------------
def data_scalor(df_train_x, df_test_x, data_type='df', scale_type='Standard'):
    ''' A general function to use only the training set to scale/normalize data, 
    then blindly apply the same transform to the test set. (no 'leaking' info.)
    
    df_train_x: a traing dataframe or a numpy
    df_test_x: a testing dataframe or a numpy
    data_type: a return data type. df: a dataframe (default), np: a numpy
    
    Stochastic Gradient Descent and any distance-related calculations are sensitive to  
    feature scaling, therefore, it is highly recommended to scale the data before modeling. 
    Note: Here I used to scale data (to handle variance)
    
    Standardize the training set using the training set means & standard deviations.
    Standardize any test set using the training set means & standard deviations.
    
    1. Use MinMaxScaler (X-Xmin/(Xmax-Xmin)) as the default if you transform a feature. 
        It's non-distorting. sensitive to outliers, value typically, between [0,1] or [-1,1].
    2. Use RobustScaler (IQR) if you have outliers and want to reduce their influence. 
       However, you might be better off removing the outliers, instead.
    3. Use StandardScaler (use (X-mean)/std ) if you need a relatively normal distribution.
    '''
    
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    if(not (isinstance(df_train_x, pd.DataFrame) or isinstance(df_train_x, np.ndarray))):
        print("Error: input data type is not DF or Numpy.")
        return df_train_x, df_test_x
        
    #Standard: mean=0, std=1; good for clustering and PCA
    if (scale_type=='Standard') :  
        scaler = StandardScaler()
    #MinMax :scale to (0-1): good for image processing (0-255) & neural network
    elif (scale_type=='MinMax') : 
        scaler = MinMaxScaler()
    elif (scale_type=='Robust') : 
        scaler = RobustScaler()
    else:
        print("Error: Given wrong scalor. Use one of (Standard, MinMax, Robust) " )
        return df_train_x, df_test_x
        
    scaler.fit(df_train_x)  # fit only on training data!!! (get mean and std)
    df_train_x1 = scaler.transform(df_train_x) #expect sum()=0 for each column
    df_test_x1 = scaler.transform(df_test_x)  #use mean & std from fitting
    
    if(isinstance(df_train_x, pd.DataFrame) and data_type=='df'):
        df_train_x1=pd.DataFrame(df_train_x1, columns=df_train_x.columns.values)
        df_test_x1=pd.DataFrame(df_test_x1, columns=df_test_x.columns.values)
        
    return df_train_x1, df_test_x1

#--------------------------------------------------------
def data_scale(X, data_type='df', scale_type='Standard'):
    ''' A function similar to data_scalor (less used)
    X is a dataframe or a numpy
    dtype: a return data type. df: a dataframe (defaut), np: a numpy
    '''

    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    if(not (isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray))):
        print("Error: input data type is not DF or Numpy.")
        return X
    #Standard: mean=0, std=1; good for clustering and PCA
    if (scale_type=='Standard') :  
        scaler = StandardScaler()
    #MinMax :scale to (0-1): good for image processing (0-255) & neural network
    elif (scale_type=='MinMax') : 
        scaler = MinMaxScaler()
    elif (scale_type=='Robust') : 
        scaler = RobustScaler()
    else:
        print("Error: Given wrong scalor. Use one of (Standard, MinMax, Robust) " )
        return df_train_x, df_test_x
    X1 = scaler.fit_transform(X)
    if(isinstance(X, pd.DataFrame) and data_type=='df'):
        X1=pd.DataFrame(X1, columns=X.columns.values)
    return X1
    
#----------------------------------------------------------------------
def add_dummy_scale(X, dtype='df'):
  '''A function to scale the features & assign dummy variables to categorical.
  X: a dataframe. Note: not include target (not need to be scaled)
  dtype: if 'df', return a DF;  if 'np' , return a num array 
  '''

  #1. separate the categorical and numerical
  X_cat=X.select_dtypes(include=['object'])
  X_num=X.select_dtypes(include=['number'])

  if(X_cat.shape[1]!=0): # categorical exist
    Xn_num=data_scale(X_num, 'df')  #standerize the num data
    df_dummy=pd.get_dummies(X_cat, drop_first=True)
    Xn=df_dummy.join(Xn_num)
    if(isinstance(X, pd.DataFrame) and dtype=='df'): #return a DF
        X1=pd.DataFrame(Xn, columns=Xn.columns.values)
        return X1
    return np.array(Xn)
  else:
    Xn_num=ut.data_scale(X_num, dtype)
    return Xn_num

#----------------------------------------------------------------------
def resample_imbalanced_data(X, y, types, ratio, target):
    ''' A function to handle imbalanced data for classification
    X: the training data with more than one features.
    y: the training data for the class only. It is a vector
    types: the model name
    target: the class 0 or 1
    ratio: the ration between class 0 or 1
    '''    
    prog=['NearMiss','RandomUnderSampler','RandomOverSampler','SMOTE','ADASYN','SMOTEENN','SMOTETomek']
    from imblearn.under_sampling import NearMiss
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.over_sampling import SMOTE
    from imblearn.over_sampling import ADASYN 
    from imblearn.combine import SMOTEENN
    from imblearn.combine import SMOTETomek
    random_state=0
    
    if(types.upper()=='OVER'):
        #upper-sample the majority class(es) by picking samples at random with replacement.
        rs = RandomOverSampler(sampling_strategy=ratio, random_state=42)
        X_rs, y_rs = rs.fit_resample(X, y)
    elif (types.upper()=='UNDER'):
        #under-sample the majority class(es) by picking samples at random with replacement.
        rs = RandomUnderSampler(sampling_strategy=ratio, random_state=42)
        X_rs, y_rs = rs.fit_sample(X, y)
    elif (types.upper()=='NEARMISS'):
        # define the undersampling method
        #https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/
        undersample = NearMiss(version=1, n_neighbors=3)
        X_rs, y_rs = undersample.fit_resample(X, y)   # transform the dataset
                
    elif (types.upper()=='ADASYN'):
        rs = ADASYN( sampling_strategy=ratio, random_state=42)
        X_rs, y_rs = rs.fit_sample(X, y)
    elif (types.upper() =='SMOTE'):
        #Synthetic Minority Oversampling Technique
        rs = SMOTE(sampling_strategy=ratio, random_state=42)
        X_rs, y_rs = rs.fit_sample(X, y)
    elif (types.upper() =='SMOTEENN'):
        rs = SMOTEENN(sampling_strategy=ratio, random_state=42)
        X_rs, y_rs = rs.fit_sample(X, y)
    elif (types.upper() =='SMOTETOMEK'):
        rs = SMOTETomek(sampling_strategy=ratio, random_state=42)
        X_rs, y_rs = rs.fit_sample(X, y)
    else:
        print("Error: Giving wrong argument!")
     
    
    XX=pd.DataFrame(X_rs, columns=X.columns.values)
    yy=pd.DataFrame(y_rs, columns=[target])

    perc=100.* len(X_rs)/len(X)
    print("Prog: {};  # of original X={}; # of resampled X={}; perc={:4.2}%".
           format(types, len(X),len(X_rs),perc))
    print('After Sampling, the shape of X: {}'.format(X_rs.shape))
    print("After Sampling, counts of label '1': {}".format(sum(y_rs==1)))
    print("After Sampling, counts of label '0': {}".format(sum(y_rs==0)))
        
    return XX, yy  #as dataframe    

#--------------------------------------------------------
def smote_pipeline(model, X_train, y_train):
  import warnings
  # ignore all caught warnings
  with warnings.catch_warnings():warnings.filterwarnings("ignore")

  weights=np.linspace(0.005, 0.8,10)

  pipe=make_pipeline(SMOTE(),model)
  gscv=GridSearchCV(estimator=pipe,param_grid={'smote__ratio': weights},
                    scoring='f1', cv=3)
  gs_res=gscv.fit(X_train, y_train)
  print("Grid serach best_params_:", gs_res.best_params_)
  ddf=pd.DataFrame({'score' : gs_res.cv_results_['mean_test_score'],
                   'weight': weights})
  ddf.plot(x='weight')
  return gs_res.best_params_['smote__ratio']

#model=XGBClassifier()
#model=RandomForestClassifier()
#best_ratio=smote_pipeline(model, X_train_sm, y_train_sm)


#--------------------------------------------------------

def get_feature_importance(model, feature_name):
    df = pd.DataFrame({'feature': list(feature_name),
                       'importance': model.feature_importances_}).\
                        sort_values('importance', ascending = False)
    print(df.head(len(feature_name)))
#--------------------------------------------------------
def transf(select, X_train, y_train, X_test, y_test):
    '''
    '''
    select.fit(X_train, y_train)
    X_train_s=select.transform(X_train)
    X_test_s=select.transform(X_test)
    print( 'The shape of X_train = ', X_train.shape)
    print( 'The shape of X_train_select = ', X_train_s.shape)

    return X_train_s, X_test_s
    
#--------------------------------------------------------
def feature_selection(X_train, X_test, y_train, y_test, ftype, nfeature):
    '''select the best feature to reduce noise, overfit
        X & y are the train, test for feature and target
        ftype: feature type ("model", "RFE");  
        nfeature: number of features (like 20)

Recursive Feature Elimination (RFE) is based on the idea to repeatedly construct 
a model and choose either the best or worst performing feature, setting the feature 
aside and then repeating the process with the rest of the features. This process 
is applied until all features in the dataset are exhausted. The goal of RFE is to 
select features by recursively considering smaller and smaller sets of features.

It enables the machine learning algorithm to train faster.
It reduces the complexity of a model and makes it easier to interpret.
It improves the accuracy of a model if the right subset is chosen.
It reduces overfitting.

refer to https://scikit-learn.org/stable/modules/feature_selection.html

    '''

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import RFE
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestRegressor
    from  xgboost import XGBClassifier
    from sklearn.feature_selection import VarianceThreshold

#Univariate feature selection works by selecting the best features 
# based on univariate statistical tests
    print("Feature selection:  using ftype=", ftype, "nfeature=", nfeature)
    if ftype=='VAR': 
        p=nfeature  #p =0.01 , remove the column having 99% same value 
        select = VarianceThreshold(threshold=p)
        
    elif ftype=='CHI2': 
        select = SelectKBest(score_func=chi2, k=nfeature)

    elif ftype=='RFE_RF': #  Recursive Feature Elimination (RFE)
        model = RandomForestRegressor(n_jobs=-1)
        select = RFE(model, nfeature)      
    elif ftype=='RFE_XGB': #  Recursive Feature Elimination (RFE)
        model = XGBClassifier(n_jobs=-1)
        select = RFE(model, nfeature)

    elif ftype=='model_RF':  #model RF based method, threshold=None
        model=RandomForestRegressor(n_jobs=-1)  #use all cores
        if nfeature==0:
            select=SelectFromModel(model, threshold=None)
        else:
            select=SelectFromModel(model, threshold=-np.inf, max_features= nfeature)
    elif ftype=='model_XGB':  #model XGB based method
        model=XGBClassifier(n_jobs=-1)
        if nfeature==0:
            select=SelectFromModel(model, threshold=None)
        else:
            select=SelectFromModel(model, threshold=-np.inf, max_features= nfeature)

    X_train_s, X_test_s = transf(select, X_train, y_train, X_test, y_test)

#the transf gives a array. I want it to be a dataframe (keep selected features)
    feature_idx = select.get_support()
    feature_name = X_train.columns[feature_idx]
    X_train_s=pd.DataFrame(X_train_s, columns=feature_name)  #mkae a df
    X_test_s=pd.DataFrame(X_test_s, columns=feature_name)  #mkae a df
    model.fit(X_train_s, y_train)
    get_feature_importance(model, feature_name)
 #   model.fit(X_train, y_train)
 #   get_feature_importance(model, list(X_train.columns))
    
    return X_train_s, X_test_s    
#--------------------------------------------------------
def get_dataset_as_df(sk_dataset):
    ''' 
    https://scikit-learn.org/stable/datasets/index.html
    There are 6 toy datasets and 9 real word datasets. USE import sklearn.datasets as data
    load_(boston, iris, diabetes, digits, linnerud, wine, breast_cancer)
    Note: digits, linnerud must be treated differently
    sk_dataset : an obj for the data: example datasets.load_boston()
    '''    
    df = pd.DataFrame(sk_dataset.data, columns=sk_dataset.feature_names)
    df['target'] = pd.Series(sk_dataset.target)
    return df

#df = get_dataset_as_df(data.load_diabetes())

#--------------------------------------------------------

def evaluate_threshold(model, X_test, y_test, threshold=0.5):
    '''
    define a function that accepts a threshold and prints sensitivity and specificity
    IMPORTANT: first argument is true values, second argument is predicted probabilities
    store the predicted probabilities for class 1
   '''

    from sklearn import metrics

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

    print ('\nSensitivity:', tpr[thresholds > threshold][-1])
    print ('Specificity:', 1 - fpr[thresholds > threshold][-1])

#evaluate_threshold(model,X_test, 0.5)

#----------------------------------------------------------------------------
def compare_test_pred_class(model, X_test):
    #compare the predicted with the testing data
    #the predicted class is the one with highest mean probability estimate

    prob=model.predict_proba(X_test)
    y_pred_prob = np.argmax(prob, axis=1)  #get the index for max prob
    print(prob.shape)

    probn=[]
    for i in range(len(prob)):  
       probn.append(prob[i][y_pred_prob[i]])
    dd=pd.DataFrame({"y_test":y_test, "y_pred": y_pred,  
                     "diff":(y_pred-y_test), "probability":probn}).reset_index()

    return(dd)


#---------------------------------------------------------------------------
#Classification results
def write_result_class(X_test, y_test, y_pred, model):
    '''X_test: test set containing features only
       y_test: test set containing target only
       y_pred: the predicted values corresponding to the y_test
       model:  the model used to train the data (X_train)
       
       Note: By default, the threshold is 0.5. (If prob>0.5, y_pred=1). 
       y_pred1 = (model.predict_proba(X_test_s)[:,1] >= 0.5).astype(int)
    '''


    #for i in range(len(y_pred)): print( 'predicted, target=', y_pred[i],y_test.values[i])

    print( '\nConfusion_matrix=\n', confusion_matrix(y_test, y_pred))
    print( 'Classification_report=\n', classification_report(y_test, y_pred))

    if len(np.unique(y_test))>2 : return

    prob=model.predict_proba(X_test)[:, 1]
#    model_roc_auc = roc_auc_score(y_test, y_pred)
    model_roc_auc = roc_auc_score(y_test, prob)
    print( 'Classification accuracy=', model.score(X_test, y_test))
    print( 'Classification AUC_ROC= ', model_roc_auc)
    

    fpr, tpr, thresholds = roc_curve(y_test, prob)
    plt.figure()
    plt.plot(fpr, tpr, label='AUC_ROC (area = %0.2f)' % model_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
   # plt.savefig('AUC_ROC')
    plt.show()

##---------------------------------------------------------
def write_result_regr(pred, Y_test, model):
    
    from sklearn.metrics import r2_score, mean_squared_error
    print( '\nUsing %s for prediction' %model)
    print( '\nr2_score=', r2_score(Y_test, pred))
    print( '\nmean_squared_error=', mean_squared_error(Y_test, pred))
    print( '\nroot_mean_squared_error=', np.sqrt(mean_squared_error(Y_test, pred)))

#----------------------------------------------------------
#Classification results
def write_cnn_result(X_test, y_test, model):
    '''X_test: test set containing features only
       y_test: test set containing target only
       y_pred: the predicted values corresponding to the y_test
       model:  the model used to train the data (X_train)
    '''

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve

# different from the regular ML. For CNN, make it 1D series to match y_test
    y_pred = model.predict(X_test)[:,0]  
    y_pred_prob=model.predict_proba(X_test)[:,0]
    # make the pred as int, ready for confusion matrix
    yy_pred=(y_pred+0.5).astype(int)

    print( '\nConfusion_matrix=\n', confusion_matrix(y_test, yy_pred))
    print( 'Classification_report=\n', classification_report(y_test, yy_pred))

    #get the model accuracy
    cnn_accuracy=model.evaluate(X_test, y_test)
    print( 'Loss & accuracy=', cnn_accuracy)

#    model_roc_auc = roc_auc_score(y_test, yy_pred)
    model_roc_auc = roc_auc_score(y_test, y_pred_prob)
    print( 'Classification AUC_ROC= ', model_roc_auc)

    #plot the AUC_ROC curve
    plot_roc(y_test, y_pred_prob, model_roc_auc)
    
#----------------------------------------------------------

def one_hot_encode_topN(df, feature_name, topN):
  '''A function to handle a categorical feature with many levels. 
   (Note: if use get_dummies, you would create too many dummy features).
   This function will create the given number (topN) levels (features).

   df: a data frame containing the categorical data
   feature_name: name of the feature to be one_encoded
   topN: the top N levels to be selected. 
  '''
  #get the topN level (a list)
  val_count=df[feature_name].value_counts()
  if(topN>=val_count.shape[0]):
    topN=val_count.shape[0]-1
  top_col=val_count.sort_values(ascending=False).head(topN).index.tolist()

  for lab in top_col :
    x=feature_name+'_'+lab
    df[x]=np.where(df[feature_name]==lab, 1,0)

  return df
  
#------------------------------------
def cross_validation(model, X, y, model_name='',kfold=10):
    '''do general cross validation based on the trained model.
    model: the trained model (such as RF, kNN, ....)
    X,  y are the data for training or testing or validation.
    kfold: for giving number of K fold
    
    cross_val_score: calculate score for each CV split
    cross_validate: calculate one or more scores and timings for each CV split
    '''
    
    from sklearn.model_selection import cross_validate
    scoring = ['precision', 'recall', 'f1', 'roc_auc', 'accuracy']

    scores =  cross_validate(model, X, y, n_jobs=-1, cv=kfold, scoring=scoring, return_train_score=False)
#   scores = cross_val_score(model, X, y, n_jobs=-1, cv=kfold, scoring='roc_auc',error_score='raise')
    for x in scores.keys():
        print("Model={} : {:15s}  mean={:.2f};  std={:.4f}".format(model_name,x, scores[x].mean(), scores[x].std() ))
    
#-------------------------------------
def create_dataset_ts(X, y, time_steps=1):
    '''A generic function to generate many lists of length (time_steps)
    The shape of the returned data is [n, m, l], where n is number of the list
    ; m is the length of the list; l is the number of features
    by using the history (n time steps from it). It works with univariate 
    (single feature) and multivariate (multiple features) Time Series data.

    '''
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)
#-----------------------------------------------
         
#=============================================================================
# ============== below are functions for NLP ============================
#=============================================================================
NLP_info='''

'''
#---------------------------------------------------
def text_vectorizer(X_train, X_test, type='tf'):
  '''
  CountVectorizer(): (BOW: bag of word) tokenize the input text data to get a matrix.
  the row is the # of documents (or messages), and the column is the unique words. 
  The value in the matrix is the TF (term frequency) for the document at the row.
  TfidfVectorizer(): BOW for term frequency and inverse document frequency. The
  matrix is scaled by IDF. TfIdf=Tf X Idf  where IDF=log((1+n)/(1+m)); 
  n is # of docs; m is # of docs with the term.

  TfidfTransformer:  scales down the values of words that occur among all texts. 
  X_train: list of text for training
  X_test: list of text for testing
  type: the type of vectorization. 
        if ='tf', use CountVectorizer(), is='tfidf' use TfidfVectorizer()
  '''
  if (type=='tf'):
    vect = CountVectorizer(decode_error='ignore')
    x_train_v = vect.fit_transform(X_train)
    x_test_v = vect.transform(X_test)
  else:
    vect = TfidfVectorizer()
    x_train_v = vect.fit_transform(X_train)
    x_test_v = vect.transform(X_test)

  return x_train_v, x_test_v, vect

        
#=============================================================================
# ============== below are functions for ploting  ============================
#=============================================================================
plot_info='''

'''
#---------------------------------------------------
def plot_hist_class(df, features, nrow=3, ncol=3, figsize=(12,8), target=False):
    '''A general function to plot the numerical feature histogram with each class
    df: a dataframe.
    features: the given columns for ploting (one string or list)
    target: the class (one string or list)
    '''
    fig=plt.figure(figsize=figsize)
    for i, var_name in enumerate(features):
        ax=fig.add_subplot(nrow,ncol,i+1)
        if (not target) :
            df[var_name].hist(bins=10,ax=ax)
        else:
            df.groupby(target)[var_name].hist(bins=15,ax=ax)
        ax.set_title(var_name+"")
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()
    
#---------------------------------------------------
def plot_hist_class_one(df,  target='', feature=[], figsize=(8,4)):
    '''df: the data frame;  target: the class, feature: the features
    '''
    for col in feature:
        pd.crosstab(df[col], df[target]).plot(kind="bar",figsize=figsize)

        #plt.title('Frequency for {}'.format(col))
        #plt.xlabel(col)
        plt.ylabel('Frequency')
    #plt.savefig('heartDiseaseAndAges.png')
        plt.show()

#---------------------------------------------------
def plot_density_class(df, features, nrow=3, ncol=3, figsize=(12,8), target=False):
    '''A general function to plot the numerical features with each class
    df: a dataframe.
    features: the given columns for ploting (one string or list)
    target: the class (one string or list)
    '''
    import itertools
    palette = itertools.cycle(sns.color_palette())
    
    fig=plt.figure(figsize=figsize)

   # if target: targets=df[target].unique()
    for i, feature_name in enumerate(features):
        ax=fig.add_subplot(nrow,ncol,i+1)
        if (not target) :
            df[feature_name].hist(bins=10,ax=ax)
        else:
            for j in df[target].unique():
                sns.distplot(df[feature_name][df[target]==j],kde=1,label='{}'.format(j))
#                sns.histplot(df[feature_name][df[target]==j],kde=1,label='{}'.format(j), color=c)
                plt.legend()
           # df.groupby(target)[feature_name].hist(bins=10,ax=ax)
        #ax.set_title(feature_name + "")
    fig.tight_layout()  # Improves appearance a bit.
    
    plt.show()

#---------------------------------------------------
def plot_density_class_one(df,  target='', feature=[], figsize=(8,4)):
    '''df: the data frame;  target: class_name,  feature: the column name 
    '''
    
    import itertools
    palette = itertools.cycle(sns.color_palette())
    
    for col in feature:
        plt.figure(figsize=figsize)
        
        for i in df[target].unique():
            sns.distplot(df[col][df[target]==i],kde=1,label='{}'.format(i))
        plt.legend()

        plt.show()
    

#---------------------------------------------------
def plot_dist_boxplot_class(df, target):
    ''' A general function to plot the numerical features distribution and 
    boxplot with each class
    df: the data frame;  target: the column name for the target (class)
    '''
    import matplotlib.gridspec as gridspec
    
    data = df.select_dtypes(include=[np.number])  #only for numerical data

    data['class_plot']='class_' + data[target].astype(str)
    print('The class values=', data[target].unique())
    for feature in data.columns:
        if feature == target or feature == 'class_plot' : continue
        print('ploting ', feature)
        gs1 = gridspec.GridSpec(3,1)
        ax1 = plt.subplot(gs1[:-1])
        ax2 = plt.subplot(gs1[-1])
        gs1.update(right=1.00)
        sns.boxplot(x=feature,y='class_plot',data=data,ax=ax2)
        for i in data[target].unique():
            sns.kdeplot(data[feature][data[target]==i],ax=ax1,label=str(i))

        ax2.yaxis.label.set_visible(False)
        ax1.xaxis.set_visible(False)
        plt.show()
        
#---------------------------------------------------
def plot_feature(select):
    '''plot feature as index
    '''
    mask=select.get_support()
    plt.matshow(mask.reshape(1,-1), cmap='gray_r')
    plt.xlabel('Index of Features')
    
#---------------------------------------------------
def ploting(list1, list2, grid, xlab, ylab):
    grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
    print(grid_mean_scores)
    plt.plot(list1, list2)
    plt.xlabel('%s'%xlab)
    plt.ylabel('%s'%ylab)
    plt.show()

#---------------------------------------------------
def plot_target_feature_corr(df, target="class"):
  '''
  '''
  plt.figure(figsize=(12,5))
  df.corr()[target].sort_values(ascending = False).plot(kind='bar')

#---------------------------------------------------
def plot_value_counts_cate(df, columns):
    '''plot 3X3 graphs
    '''
    
    fig, axes = plt.subplots(nrows = 3,ncols = 3,figsize = (15,12))
    
    for i, item in enumerate(columns):
        if i < 3:
            ax = df_cat[item].value_counts().plot(kind = 'bar',ax=axes[i,0],rot = 0)       
        elif i >=3 and i < 6:
            ax = df_cat[item].value_counts().plot(kind = 'bar',ax=axes[i-3,1],rot = 0)
        
        elif i < 9:
            ax = df_cat[item].value_counts().plot(kind = 'bar',ax=axes[i-6,2],rot = 0)
        ax.set_title(item)

#---------------------------------------------------
def plot_feature_importance(X, y, model=RandomForestClassifier(),nfeature=30):
    '''plot important features
    X: the training data;  y the target for training data
    '''

    model=RandomForestClassifier()
    model.fit(X,y)
#    print(model.feature_importances_)  

    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat=feat_importances.nlargest(nfeature).sort_values(ascending=True)
    feat_importances.nlargest(nfeature).sort_values(ascending=True).plot(kind='barh')
    plt.show() 
    #feature=feat[feat>0.03].index
    return feat

#---------------------------------------------------
def plot_precision_and_recall(model,X_train, y_train, figsize=(10, 6)):
  # getting the probabilities of our predictions
  from sklearn.metrics import precision_recall_curve
  y_scores = model.predict_proba(X_train)[:,1]
  precision, recall, threshold = precision_recall_curve(y_train, y_scores)

  plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=3)
  plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=3)
  plt.xlabel("threshold", fontsize=12)
  plt.legend(loc="center right", fontsize=12)
  plt.ylim([0, 1])
  plt.grid()
  plt.figure(figsize=figsize)
  plt.show()
    
#------------------------------------------------------
def plot_precision_vs_recall(model,X_train, y_train, figsize=(10, 6)):
  # getting the probabilities of our predictions

  from sklearn.metrics import precision_recall_curve
  y_scores = model.predict_proba(X_train)[:,1]
  precision, recall, threshold = precision_recall_curve(y_train, y_scores)

  plt.plot(recall, precision, "g--", linewidth=2.5)
  plt.ylabel("recall", fontsize=14)
  plt.xlabel("precision", fontsize=14)
  plt.axis([0, 1.01, 0, 1.01])
  plt.grid()
  plt.figure(figsize=figsize)
  plt.show()
    
#-------------------------------------------------------------------------
def plot_deep_learning_curve(history):
    
    #plot the model accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Model accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    
    #plot the loss function
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()
    
#-------------------------------------------------------------------------

def plot_loss(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, historydf.values.max()))
    plt.title('Loss: %.3f' % history.history['loss'][-1])
#-------------------------------------------------------------------------

def plot_deep_learning_curve_reg(history):
    
    #plot the model accuracy
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title("Model mean_absolute_error")
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    
    #plot the loss function
    plt.plot(history.history['loss'])
    plt.plot(history.history['mean_absolute_error'])
    plt.title("Model loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()
    
#-------------------------------------------------------------------------

def plot_roc(y_test, y_pred_prob, model_roc_auc):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label='AUC_ROC (area = %0.2f)' % model_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('AUC_ROC')
    plt.show()
#-------------------------------------------------------------------------
def plot_decompose_ts(y):
  '''visulize distinct components: trend, seasonality, and noise.
  y : a series data with date as index (df['rgu'])
  '''
  from pylab import rcParams
  import statsmodels.api as sm

  rcParams['figure.figsize'] = 12, 8
  decomposition = sm.tsa.seasonal_decompose(y, model='additive')
  fig = decomposition.plot()
  plt.show()  
#plot_decompose_ts(d1['rgu'])
#-------------------------------------------------------------------------

def plot_df_ts(df):
  '''plot each column. The index should be time series
  '''
  fig, axes = plt.subplots(nrows=4, ncols=2, dpi=120, figsize=(10,8))
  for i, ax in enumerate(axes.flatten()):
    data = df[df.columns[i]]
    ax.plot(data, color='blue', linewidth=1)
    # Decorations
    ax.set_title(df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

  plt.tight_layout();
#plot_df_ts(df)

#-------------------------------------------------------------------------
def plot_validation_ARIMA(model, ts, start_date, ylab=' ', start_plot=''):
  '''A function to plot and forcast the future (from last date to some step)
  model: the ARIMA model built against the data
  ts: A series with date as index
  start_date: a date from here to the end
  ylab: the label for y
  '''

  if len(start_plot)==0: start_plot=start_date
  pred = model.get_prediction(start=pd.to_datetime(start_date), dynamic=False)
  pred_ci = pred.conf_int()
  ax = ts[start_plot :].plot(label='Observed') #original
  pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, 
                           figsize=(12, 6))
  ax.fill_between(pred_ci.index,
                  pred_ci.iloc[:, 0],
                  pred_ci.iloc[:, 1], color='k', alpha=.2)
  ax.set_xlabel('Date')
  ax.set_ylabel(ylab)
  plt.legend()
  plt.show()

  y_forecasted = pred.predicted_mean
  y_truth = ts[start_date :]
  mse = ((y_forecasted - y_truth) ** 2).mean()
  rmse=round(np.sqrt(mse), 2)
  print('The Mean Squared Error (MSE) is {}. RMSE={}'.format(round(mse, 2), rmse))

  pred_ci['forecast']=pred.predicted_mean
  pred_ci['true']=y_truth
  return pred_ci

#dd=plot_validation_ARIMA(model=results, ts=d1['rgu'], start_date='2020-03-09', 
#                         ylab=' ', start_plot='2018-03-09')

#-------------------------------------------------------------------------
def plot_conf_matrix(y_true, pred, size=6):
  CM = confusion_matrix(y_true, pred)
  fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(8, 8))
  plt.show()

#plot_conf_matrix(y_true, pred, size=6)
#-------------------------------------------------------------------------

def plot_conf_matrix0(model, class_label, y_test, y_pred, size=8):
  '''
  class_label: the labels for the class.  default size is 8
  '''
  from sklearn.metrics import confusion_matrix

  fig, ax = plt.subplots(figsize=(size,size))         # Sample figsize in inches
  cm=confusion_matrix(y_test, y_pred)
  sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=False, 
              xticklabels=class_label,  yticklabels=class_label, ax=ax)
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.show()
  
#-------------------------------------------------------------------------
  
def plot_corr(df, figsize=(18,14)):
  '''Plot the heat map:   plot_corr(df, figsize=(18,14))
  df: the Data frame
  '''
  
  print ('\nPlot heatmap to show correlations among the numerical features.')
  colormap = plt.cm.RdBu
  plt.figure(figsize=figsize)
  plt.title('Pearson Correlation of Features', y=1.05, size=10)
  pcorr=df.astype(float).corr()
  sns.heatmap(pcorr,linewidths=0.1,vmax=1.0,
         square=True, cmap=colormap, linecolor='white', annot=True)
  
#-------------------------------------------------------------------------
def plot_wordcloud(text, fig_size=(18,10), title=''):
  '''A function plot_wordcloud(text, fig_size=(18,10), title='')
  to plot WordCloud.  WordCloud representation of most used words in each doc.
  '''
  from wordcloud import WordCloud,STOPWORDS

  plt.figure(figsize=fig_size)

  cloud_toxic = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          collocations=False,
                          width=2500,
                          height=1800
                         ).generate(" ".join(text))

#plt.subplot(2, 3, 1)
  plt.axis('off')
  plt.title(title,fontsize=40)
  plt.imshow(cloud_toxic)

#-------------------------------------------------------------------------
def plot_bars(cate, value, fig_size=(12,6), title='', xlab='', ylab=''):
  '''A general function to plot bar giving bar names (cate) and the value (list)
  plot_bars(cate, value, fig_size=(12,6), title='', xlab='', ylab='')
  cate: A list for the target features.
  value: A list values for each target feature.
  '''
  sns.set(font_scale = 1.5)
  plt.figure(figsize=fig_size)
  ax= sns.barplot(cate, value)

  plt.title(title)
  plt.xlabel(xlab,fontsize=12)
  plt.ylabel(ylab)

#adding the text labels on the top of bar
  
  rects = ax.patches
  labels = value
  for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', 
            va='bottom')
  
plt.show()
#-------------------------------------------------------------------------
def plot_tree(model, max_depth=3, feature_name=[], class_name=''):
    '''
    model: should be the decisen tree model.
    '''

    from sklearn import tree
    import graphviz
    
# DOT data  'test-tree.png'
    dot_data = tree.export_graphviz(model, max_depth=max_depth, label='all', filled=True,
                                    feature_names=feature_name, class_names=class_name )

# Draw graph
    graph = graphviz.Source(dot_data, format="png") 
    return graph

#-------------------------------------------------------------------------
#========================================================================================
def optimize_SARIMAX(ts, num):
  ''' A function to use a grid search to find the optimal set of parameters  
  that yields the best performance for our model.
  ts: A series with date as index (work when: d1=d1.resample('W').mean() )
  num: a given number for various orders

  After running, check the DF and use the order to the model
  '''

  import itertools
  from statsmodels.tsa.arima_model import ARIMA
  from statsmodels.tsa.statespace.sarimax import SARIMAX

  p = d = q = range(0, num)
  pdq = list(itertools.product(p, d, q))  #combinations for Seasonal ARIMA.
  seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

  par=[]
  for param in pdq:
    for param_seasonal in seasonal_pdq:
      try:
        mod = SARIMAX(ts,
                      order=param,
                      seasonal_order=param_seasonal,
                      enforce_stationarity=False,
                      enforce_invertibility=False)
        results = mod.fit()
        par.append([param, param_seasonal, results.aic])
      except:
#        print('Error occurred')
        continue
  df_par=pd.DataFrame(data=par, columns=['param', 'param_seasonal', 'results.aic'])
  return df_par.sort_values(by=['results.aic'])

#dd= optimize_SARIMAX(d1['rgu'], num=2) 

#-------------------------------------------------------------------------
def forecast_future_ARIMA(model, ts, steps, ylab=' ', plot=False):
  '''A function to plot and forcast the future (from last date to some step)
  model: the ARIMA model built against the data
  ts: A series with date as index
  steps: a given number of steps forward (e.g 10 weeks, 10 month)
  ylab: the label for y
  '''
  pred_uc = model.get_forecast(steps)
  pred_ci = pred_uc.conf_int()  # a DF
  pred_mean=pred_uc.predicted_mean
  pred_ci['avg']=pred_mean.tolist()
  if not plot: return pred_ci

  ax = ts.plot(label='Observed', figsize=(12, 6))
  pred_mean.plot(ax=ax, label='Forecast')
  ax.fill_between(pred_ci.index,
                  pred_ci.iloc[:, 0],
                  pred_ci.iloc[:, 1], color='k', alpha=.25)
  ax.set_xlabel('Date')
  ax.set_ylabel(ylab)
  plt.legend()
  plt.show()
  return pred_ci

#dd=forecast_future_ARIMA(model=results, ylab='RGU', ts=d1['rgu'], steps=20, plot=True)

#---------------------------------------------------
def arima_forcast(df, target, train_size):
  '''A function to use ARIMA for 1D forcasting
  Autoregressive Integrated Moving Average (ARIMA)
  It combines both Autoregression (AR) and Moving Average (MA) models as well 
  as a differencing pre-processing step of the sequence to make the sequence 
  stationary, called integration (I).
  order=(3,1,1)  AR(p), I(d), and MA(q)

  df: a dataframe containing target, timestamp as index
  target: a column for prediction
  train_size: a percentage of df used for training
  '''

  from statsmodels.tsa.arima_model import ARIMA
  from statsmodels.tsa.statespace.sarimax import SARIMAX
  from sklearn.metrics import mean_squared_error
  import math

  X = df[target]
  size = int(len(X) * train_size)
  print('train size=', size)
  train, test = X[0:size], X[size:len(X)]
  history = [x for x in train]
  predictions = list()
  #predicting one at a time
  for i in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
#    model = SARIMAX(history, order=(1, 1, 1), seasonal_order=(2, 2, 2, 2))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[i]
    history.append(obs) #adding one value from test to train
#    print(t+1, 'predicted=%f, expected=%f' % (yhat, obs))
   
  error = mean_squared_error(test, predictions)
  print('Test RMSE: %.2f' % math.sqrt(error))

  dd=pd.DataFrame(data=test)
  if type(predictions[0])==list:
    dd['pred']=predictions
  else:
    dd['pred']=[x.tolist()[0] for x in predictions]
  dd.plot(figsize=(10,5))
  train.plot(figsize=(10,5))
  plt.show()
  return dd
#dd=arima_forcast(d1, target='rgu', train_size=0.95) 

#--------------------------------------------------------
def print_report(y_actual, y_pred, thresh):
    
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    fscore = f1_score(y_actual,(y_pred > thresh) )
    specificity = sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)
    print('AUC:%.3f'%auc)
    print('accuracy:%.3f'%accuracy)
    print('recall:%.3f'%recall)
    print('precision:%.3f'%precision)
    print('fscore:%.3f'%fscore)
    print('specificity:%.3f'%specificity)
    print(' ')
    return auc, accuracy, recall, precision,fscore, specificity 
#--------------------------------------------------------

def logit_prob(reg, x_test):
    '''logit_prob(reg, x_test): calc prob from x_test and return prob.
    '''
    import numpy as np
    logit=x_test.dot(reg.coef_.T) + reg.intercept_
    p=1./(1+np.exp(-logit))
    return p

#-----------------------------
import sys
import warnings
import math
import statsmodels
import numpy as np
from scipy import stats
import statsmodels.api as smf

def firth_likelihood(beta, logit):
    return -(logit.loglike(beta) + 0.5*np.log(np.linalg.det(-logit.hessian(beta))))

#-----------------------------
# Do firth regression
# Note information = -hessian, for some reason available but not implemented in statsmodels
def fit_firth(y, X, start_vec=None, step_limit=1000, convergence_limit=0.0001):

    logit_model = smf.Logit(y, X)
    
    if start_vec is None:
        start_vec = np.zeros(X.shape[1])
    
    beta_iterations = []
    beta_iterations.append(start_vec)
    for i in range(0, step_limit):
        pi = logit_model.predict(beta_iterations[i])
        W = np.diagflat(np.multiply(pi, 1-pi))
        var_covar_mat = np.linalg.pinv(-logit_model.hessian(beta_iterations[i]))

        # build hat matrix
        rootW = np.sqrt(W)
        H = np.dot(np.transpose(X), np.transpose(rootW))
        H = np.matmul(var_covar_mat, H)
        H = np.matmul(np.dot(rootW, X), H)

        # penalised score
        U = np.matmul(np.transpose(X), y - pi + np.multiply(np.diagonal(H), 0.5 - pi))
        new_beta = beta_iterations[i] + np.matmul(var_covar_mat, U)

        # step halving
        j = 0
        while firth_likelihood(new_beta, logit_model) > firth_likelihood(beta_iterations[i], logit_model):
            new_beta = beta_iterations[i] + 0.5*(new_beta - beta_iterations[i])
            j = j + 1
            if (j > step_limit):
                sys.stderr.write('Firth regression failed\n')
                return None

        beta_iterations.append(new_beta)
        if i > 0 and (np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) < convergence_limit):
            break

    return_fit = None
    if np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) >= convergence_limit:
        sys.stderr.write('Firth regression failed\n')
    else:
        # Calculate stats
        fitll = -firth_likelihood(beta_iterations[-1], logit_model)
        intercept = beta_iterations[-1][0]
        beta = beta_iterations[-1][1:].tolist()
        bse = np.sqrt(np.diagonal(np.linalg.pinv(-logit_model.hessian(beta_iterations[-1]))))
        
        return_fit = intercept, beta, bse, fitll

    return return_fit

#==================================================================
def logit_Firth(X, y):
    ''' logit_firth(X, y)
    Python implementation of Firth regression by John Lees
    See https://www.ncbi.nlm.nih.gov/pubmed/12758140
    '''

  
    # create X and y here. Make sure X has an intercept term (column of ones)
    # ...

    # How to call and calculate p-values
    (intercept, beta, bse, fitll) = fit_firth(y, X)
    beta = [intercept] + beta
    
    # Wald test
    waldp = []
    for beta_val, bse_val in zip(beta, bse):
        waldp.append(2 * (1 - stats.norm.cdf(abs(beta_val/bse_val))))

    # LRT
    lrtp = []
    for beta_idx, (beta_val, bse_val) in enumerate(zip(beta, bse)):
        null_X = np.delete(np.array(X), beta_idx, axis=1)
        (null_intercept, null_beta, null_bse, null_fitll) = fit_firth(y, null_X)
        lrstat = -2*(null_fitll - fitll)
        lrt_pvalue = 1
        if lrstat > 0: # non-convergence
            lrt_pvalue = stats.chi2.sf(lrstat, 1)
        lrtp.append(lrt_pvalue)
    
    df_result=pd.DataFrame({'features':X.columns.tolist(),'Beta':beta, 'Pvalue':lrtp, 'Wald_test':waldp })
    return(df_result, intercept)

# ------------------------------------------------------------------------------
def get_df_pca(df_in, loading=0.9, figsize=(8,6) ):

    from sklearn.decomposition import PCA
    #plot PCA
    df=df_in.copy()
    pca=PCA().fit(df)
    ratio=pca.explained_variance_ratio_
    plt.figure(figsize=figsize)
    plt.plot(range(0, len(ratio)), ratio.cumsum(), marker='o', linestyle='--')
    plt.grid()
    plt.show()

#get number of components if ask to explain variance of 0.95 
    npca=PCA(loading).fit(df).n_components_
    print('Number of components {:2d} corresponding to the loading factor of {:.2f}.'.format(npca,loading))

#final PCA and check clusters
    pca_f=PCA(n_components = npca).fit(df)
    df_tran=pca_f.transform(df)
    return df_tran

# ------------------------------------------------------------------------------
def groupby_target(df, col_inp=[], target=['target']):
    '''
    col_inp: the given columns for agg.
    target: the target or a list of INT variables
    '''
#    dfn=df.select_dtypes(include=['int64'])
    
    dfn=df.copy()
    print('Aggregating {} for the given (or all) features.'.format(target))
#    dfm=pd.DataFrame(columns=['feature','count', 'mean', 'sum', 'std', 'max', 'min'])
#  if specify columns, the DF will use it, but multiple index may be affected!!
    dfm=pd.DataFrame() 
    col=col_inp
    if len(col_inp) ==0 : col=dfn.columns.tolist()
    y=target
    for x in col:
        dft=dfn.groupby(y)[x].agg(['count', 'mean', 'sum', 'std', 'max', 'min'])
        dft['feature'] = x
        
        dfm=dfm.append(dft)
        
    dfm1= dfm.reset_index().sort_values(['feature']+target)
    dfm2= dfm1.reset_index(drop=True)
    return dfm2
        
#ddm1=groupby_target(df1, col_inp=[], target=['Participant', 'CKD_Flag'])