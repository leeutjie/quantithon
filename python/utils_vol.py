import numpy as np 
import pandas as pd 
import math

# scipy
from scipy.signal import lfilter, lfilter_zi

# Arch
from arch import arch_model

# Sklearn
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



############################# Actual Realised Variance #############################

def var_act(ret, h):
    """
    Realised variance.
    
    Args: 
      ret: Log returns.
      h: The horizon over which the returns are calculated. {1, 5, 21, 63, 126}.
    """
    return (ret**2).rolling(h).mean().shift(1-h)

############################# Variance Estimators #############################

def var_est_eq_wma(ret, k):
    """
    Variance estimates using equally weighted moving averages.
    Args: 
        ret: Log returns.
        k: Number of past observations.  
    """
    return (ret**2).rolling(k).mean().shift(1)

def var_est_expon_wma(ret, alpha):
    """
    Variance estimates using exponentially weighted moving averages. 
     Args: 
        ret: Log returns. 
        alpha: smoothing factor.
    """
    return (ret**2).ewm(alpha=alpha).mean().shift(1)

def var_est_expon_wma_z0(ret, alpha):
    """
    Variance estimates using exponentially weighted moving averages with an initial condition.
    
    Args: 
        ret: Log returns. 
        alpha: Smoothing factor. 
    """
    ret2 = ret**2
    a = [1, alpha-1]
    b = [alpha]
    zi = lfilter_zi(b,a)
    var  = lfilter(b=b, a=a, x=ret2,zi=zi*ret2[0] )
    var = var[0]
    return pd.Series(index=ret2.index, data=var).shift(1)


############################# VPE #############################

# Using logged varainces to measure the predictive efficacy.
def VPE(actual, predicted):
    return 1 - (abs(np.log(actual)-np.log(predicted))).mean(axis=0)/(abs(np.log(actual)-(np.log(actual)).mean())).mean(axis=0)


############################# Log-Likelihood #############################

def log_likelihood(func, ret, **kwargs):
    """
    Log-likelihood. 
    Args: 
        func: The variance estimator.
        ret: Log returns.
    """
 
    mu = 0
    sigma = 1
    
    var = func(ret, **kwargs)
   
    data_transform1 = (ret - ret.mean())/np.sqrt(var)
    data_transform2 = ( np.log(np.sqrt(var)) - ( np.log(np.sqrt(var)) ).mean() ) / ( np.log(np.sqrt(var)) ).std()

    log_likelihood = np.sum( np.log( 1/sigma*(np.sqrt(2*math.pi)) ) - 0.5*((data_transform1-mu)/sigma)**2 ) + np.sum(  np.log( 1/sigma*(np.sqrt(2*math.pi)) ) - 0.5*((data_transform2-mu)/sigma)**2  ) 

    return log_likelihood


############################# Parameter Optimisation #############################

def objective(trial, func, ret):
    if func == var_est_eq_wma:
        k = trial.suggest_int('k', 2, 100)
        log_like = log_likelihood(func, ret=ret, k=k)
    else:
        alpha = trial.suggest_float('alpha', 0.00001, 0.2)
        log_like = log_likelihood(func, ret=ret, alpha=alpha)
    return log_like


############################# Garman Klass #############################

def garman_klass(open, close, low, high):
     """
     Garman Klass estimates.
     Args:
          open: Log of the open price.
          close: Log of the close price.
          low: Log of the lowest price.
          high: Log of the highest price.
     """
     c = close - open
     h = high - open
     l = low - open
     #return 0.511*(h-l)**2-0.019*(c*(h+l)-2*h*l)-0.383*c**2 # Exact Garman Klass.
     return 0.5*(h-l)**2 - (2*np.log(2)-1)*c**2 # More practical.

def garman_klass_jumps(open, close, low, high):
     """
     Garman Klass estimates taking into account the price difference between the current and previous day.

     Args:
          open: Log of the open price.
          close: Log of the close price.
          low: Log of the lowest price.
          high: Log of the highest price.
     """
     c = close - open
     h = high - open
     l = low - open
     j = open - close.shift(1)
     return 0.5*(h-l)**2 - (2*np.log(2)-1)*c**2 + j**2


############################# ARCH #############################

def arch_h(ret, realised_var, n_train, n_test,  x, f_update, method='analytic', **kwargs):
    """
    Multiple horizon forecasts over a test period.

    Args: 
        ret: A series of returns.
        realised_var: Actual realised variance.
        n_train: Minimum or initial number of train samples.
        n_test: The number of test samples.
        x: Exogenous variables.
        f_update: The frequency at which the model will be fit.
        method: The forecast method. 'analytic', 'simulation', 'bootstrap'.

    Returns:
        forecasts_h: Forecasted volatilities.
        actual_h: Actual volatilities. 

    """
    h = [5, 21, 63, 126, 192, 252]

    if n_test==None: n_test=len(ret)-n_train
    
    if n_test<=252*2 or n_test>len(ret-n_train): raise Exception('Too little test data or too few test data points specified. Either decrease the minimum length of the training set or add more observations.') 

    forecasts_h = np.zeros([n_test-252, len(h)])
    actual_h = np.zeros([n_test-252, len(h)])

    ret_train = ret.iloc[:n_train]

    if x is None: x_train=None; x_forecast=None
    else: x_train = x.iloc[:n_train]; x_forecast = x.iloc[n_train:n_train+252]

    arch_train_res = arch_model(ret_train,x=x_train,**kwargs).fit(disp='off')
    
    for i in range(n_test-252): 
        ret_train = ret.iloc[:n_train+i]
        if x is None: x_train=None; x_forecast=None
        else:
            x_train = x.iloc[:n_train+i]
            x_forecast = x.iloc[n_train+i:n_train+i+252]

        if i%f_update==0:
            arch_train_res = arch_model(ret_train,x=x_train,**kwargs).fit(disp='off') 
        else:
            arch_train = arch_model(ret_train,x=x_train,**kwargs)
            arch_train_res =  arch_train.fix(arch_train_res.params.values)
            
        forecasts = arch_train_res.forecast(horizon=252, reindex=False, method=method, x=x_forecast) 
        forecasts_df = pd.DataFrame(data=forecasts.variance.values.T, index=realised_var.iloc[n_train+i:n_train+i+252].index, columns=['forecasts'])
        for j, w in enumerate(h):
            forecasts_h[i,j]=forecasts_df.iloc[0:w].mean()[-1]
            actual_h[i,j]=realised_var.iloc[n_train+i:n_train+i+w].mean()
    print(arch_train_res.summary())
    return forecasts_h, actual_h


############################# ML / Regression #############################

def est_h(features, ret, estimator, n_train, n_test, f_update, scale,  **kwargs):
    """
    
    Args:
        features: The feature set.
        ret: Log returns.
        estimator: The estimator function.
        n_train: The number of train samples.
        n_test: The number of test samples.
        f_update: The frequency at which the model gets updated.  
        scale: Standardize the data. A boolean - True or False.
    """
    h = [5, 21, 63, 126]
    
    if (n_test is not None and n_test < 130): raise Exception('Too few test points specified.')
    
    n_embargo = 130

    n_train_em = n_train - n_embargo
    
    predict_h = []
    actual_h = []
    model_h = []
    

    for j, w in enumerate(h):

        n_embargo = w+np.ceil(w*0.1)
        
        target = var_act(ret, w)/10000

        target = target.iloc[61:-w+1].dropna()

        idx = features.index.intersection(target.index)
        features_h = features.loc[idx]
        target_h = target.loc[idx]

        if n_test is None: n_test_h = len(features_h)-n_train

        
        X_tr, _, y_tr, _ = train_test_split(features_h, target_h, train_size=n_train_em, shuffle=False)

        scaler_x = StandardScaler().fit(X_tr)
        scaler_y = StandardScaler().fit(y_tr.values.reshape(-1, 1))

        if scale is True:
            features_scaled = scaler_x.transform(features_h)
            target_scaled = scaler_y.transform(target_h.values.reshape(-1, 1))
            features_h = pd.DataFrame(features_scaled, index=features_h.index, columns=features_h.columns)
            target_h = pd.Series(target_scaled.flatten(), index=target_h.index)


        split = np.arange(0,n_test_h,f_update)

        predict = []
        for i,s in enumerate(split):
            X_train = features_h.iloc[:n_train_em+s]; y_train = target_h.iloc[:n_train_em+s]
            
            if i==(len(split)-1): 
                X_test = features_h.iloc[n_train+s:n_train+n_test_h] 
 
            else:
                X_test = features_h.iloc[n_train+s:n_train+s+split[1]] 

            model = estimator(**kwargs).fit(X_train, y_train)
            predict = predict + list(model.predict(X_test))

        if scale is True:
            predict_h.append(pd.Series((scaler_y.inverse_transform(np.array(predict).reshape(-1,1))).flatten(), index=target_h.iloc[n_train:n_train+n_test_h].index))
            actual_h.append( pd.Series( (scaler_y.inverse_transform( target_h.iloc[n_train:n_train+n_test_h].values.reshape(-1,1) )).flatten() , index=target_h.iloc[n_train:n_train+n_test_h].index) )
        else:
            predict_h.append(pd.Series(predict, index=target_h.iloc[n_train:n_train+n_test_h].index))
            actual_h.append( target_h.iloc[n_train:n_train+n_test_h])



        model_h.append(model)
        
    return predict_h, actual_h, model_h


############################# Heading #############################

## Add your functions here
