import numpy as np
import pandas as pd
from itertools import combinations

# Sklearn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import math

# Optuna
import optuna 

# PyGAM
from pygam import LinearGAM

class GamCV:
    """
    Cross Validation for the Linear GAM.

    """
    def __init__(self, X:pd.DataFrame, y:pd.Series, cv:object, penalties:str='auto', constraints:str=None, lam:list|float=None, n_splines:int=None, spline_order:int=None, sample_weights:pd.Series=None, X_intercept:list=None, y_intercept:float|int=None):
        """
        Args:
            X: The full feature set.
            y: The target set.
            cv: Cross validation object.
            penalties: Smoothing penalty of terms. Choose from 'auto', 'l2', 'derivative' or None.
            constraints: Term constraints. Choose from 'convex', 'concave', 'monotonic_inc', 'monotonic_dec' or None. 
            lam: The lambda values of the features. If None then the best set of lambda values are found.
            n_splines: Number of splines.  If None then the best number of splines are found.
            spline_order: Spline order. If None then the best spline order is found.
            sample weights: Sample weights of the target set.
            X_intercept: A ficticious set of feature entries. 
            y_intercept: A ficticious target value.

        """
        self.X = X
        self.y = y
        self.cv = cv
        self.penalties = penalties
        self.constraints = constraints
        self.lam = lam 
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size=0.33, shuffle=False)
        self.sample_weights = sample_weights

        if sample_weights is None:
            self.sample_weights_train,  self.sample_weights_test = None, None
        else:
            self.sample_weights_train = sample_weights.loc[self.X_train.index]
            self.sample_weights_test = sample_weights.loc[self.X_test.index]
            
        self.X_intercept = X_intercept
        self.y_intercept = y_intercept


    def add_intercept_train(self)->tuple[pd.DataFrame,pd.Series,pd.Series]:
        """
        Includes a ficticious row of entries in the training sets of the feature, target and sample weights data.

        Returns:
            The augmented training sets.

        """
        data = {self.X_train.columns[i]: self.X_intercept[i] for i in range(len(self.X_train.columns))}

        if self.X_train.index.dtype==np.int64: 
            X_row = pd.DataFrame(index=[len(self.X_train)], data=data, columns=self.X_train.columns)
            y_row = pd.Series(index=[len(self.y_train)], data=self.y_intercept)
            s_row = pd.Series(index=[len(self.y_train)], data=4*len(self.X))

        elif self.X_train.index.dtype=='datetime64[ns]':
           X_row = pd.DataFrame(index=[self.X_train.index[-1]+pd.Timedelta(days=1)], data=data, columns=self.X_train.columns) 
           y_row = pd.Series(index=[self.X_train.index[-1]+pd.Timedelta(days=1)], data=self.y_intercept)
           s_row = pd.Series(index=[self.X_train.index[-1]+pd.Timedelta(days=1)], data=4*len(self.X))
        
        else: 
            raise Exception('Unsupported index type')
        
        X_train_augmented = pd.concat([self.X_train, X_row])
        y_train_augmented = self.y_train.append(y_row)
        weights_train_augmented = self.sample_weights_train.append(s_row)

        return X_train_augmented, y_train_augmented, weights_train_augmented


    def make_gam(self, lam_list:list, spline_order_list:list, n_splines_list:list)->object:
        """
        Creates a linear GAM according to specifications. Penalties and constraints are fixed.

        Args:
            lam_list: Lambda values.
            spline_order_list: Spline order values.
            n_splines: number of spline values.

        Returns:
            gam: A linear GAM.
        """
        gam = LinearGAM(penalties=self.penalties, 
                        constraints=self.constraints, 
                        lam=lam_list, 
                        spline_order=spline_order_list, 
                        n_splines=n_splines_list)
        
        return gam 
   
    def cv_score(self, X_train:pd.DataFrame, y_train:pd.Series, weights_train:pd.Series, lam_list:list, spline_order_list:list, n_splines_list:list)->float:
        """
        Scoring method for the cross validation object.

        Args:
            X_train: Training feature set.
            y_train: Training target set.
            weights_train: Training sample weights set. 
            lam_list: Lambda values.
            spline_order_list: Spline order values.
            n_splines: number of spline values.
        
        Returns:
            Mean cv score across the folds. 
        """

        score = []

        for train,test in self.cv.split(X_train):

            gam = self.make_gam(lam_list=lam_list, spline_order_list=spline_order_list, n_splines_list=n_splines_list)

            if weights_train is None:
                gam_fit = gam.fit(X_train.iloc[train], y_train.iloc[train])
            else:
                gam_fit = gam.fit(X_train.iloc[train], y_train.iloc[train], weights_train.iloc[train])
                
            y_hat = gam_fit.predict(X_train.iloc[test].values)

            if weights_train is None:
                r2 = r2_score(y_train.iloc[test], y_hat)
            else:
                r2 = r2_score(y_train.iloc[test], y_hat, sample_weight=weights_train.iloc[test])

            score.append(r2)

        return np.mean(score)
    

    def cv_score_matrix(self, X_train:pd.DataFrame, y_train:pd.Series, weights_train:pd.Series, lam_list:list, spline_order_list:list, n_splines_list:list)->float:

        # Test fold combinations.
        comb_list = self.cv.combination_list()
     
        score_matrix = np.zeros((self.cv.n_splits,len(comb_list)))

        # Generate the prediction scores for each test and train fold combination.
        for fold,(train,test) in enumerate(self.cv.split(X_train)):

            gam = self.make_gam(lam_list=lam_list, spline_order_list=spline_order_list, n_splines_list=n_splines_list)

            if weights_train is None:
                gam_fit = gam.fit(X_train.iloc[train], y_train.iloc[train])
            else:
                gam_fit = gam.fit(X_train.iloc[train], y_train.iloc[train], weights_train.iloc[train])
            
            for idx,test_batch in enumerate(test):
                y_hat = gam_fit.predict(X_train.iloc[list(test_batch)].values)
                if weights_train is None:
                    r2 = r2_score(y_train.iloc[list(test_batch)].values, y_hat)
                else:
                    r2 = r2_score(y_train.iloc[list(test_batch)].values, y_hat, sample_weight=weights_train.iloc[list(test_batch)].values)

                score_matrix[comb_list[fold][idx],fold] = r2

        score_matrix[score_matrix==0]=np.nan


        return np.nanmean(score_matrix)

    
    def objective(self, trial:object, X_train:pd.DataFrame, y_train:pd.Series, weights_train:pd.Series)->float:
        """
        Creates the Optuna objective function.

        Args:
            trial: Optuna trial.
            X_train: Training feature set.
            y_train: Training target set.
            weights_train: Training sample weights set. 

        Returns:
            Cross validated score.
        """

        n_feat = self.n_features(X_train)

        # lambdas
        if self.lam is None:
            lambda_list = []
            for i in range(n_feat):
                lambda_list.append(trial.suggest_float('lambda_{}'.format(i+1), 0, self.lambda_scale(X_train, i)))
        else:
            lambda_list = self.lam
        
        # spline_order and n_splines
        if self.spline_order is None and self.n_splines is None:
            spline_order_list = [trial.suggest_int('spline_order', 1, 5)] * self.n_features(X_train)
            n_splines_list = [trial.suggest_int('n_splines', spline_order_list[0]+1, 25)] * self.n_features(X_train)

        elif self.spline_order is not None and self.n_splines is None:
            spline_order_list = self.spline_order
            n_splines_list = [trial.suggest_int('n_splines', self.spline_order+1, 25)] * self.n_features(X_train)

        elif self.n_splines is not None and self.spline_order is None:
            n_splines_list = self.n_splines
            spline_order_list = [trial.suggest_int('spline_order', 1, self.n_splines-1)] * self.n_features(X_train)

        else:
            n_splines_list = self.n_splines
            spline_order_list = self.spline_order

        return self.cv_score(X_train, y_train, weights_train, lambda_list, spline_order_list, n_splines_list)
    
    def n_features(self, X:pd.DataFrame)->int:
        """
        Args: 
            X: Feature set.

        Returns:
            Number of features.
        """
        return len(X.columns)

    def lambda_scale(self, X:pd.DataFrame, i:int)->float:
        """
        Args:
            X: Feature set.
            i: index.

        Returns:
            The feature scaling.
        """
        return 10**(math.ceil(math.log10(abs(X.iloc[:,i].mean()))))

    def call(self)->tuple[float,object,object,object]:
        """
        Initiates parameter tuning for the Linear GAM using using cross validation. The final GAM is fit on the full training set.
        An out-of-sample r^2 is quoted for the test set. 

        Returns:
            test_r2: Out-of-sample test set r^2.
            study: The Optimisation study.
            gam_best: The tuned GAM.
            gam_fit: The tuned GAM fitted on the train set.
        """

        print('--------------------------------------------------------------------------------------')
        print('A Linear GAM will be created (with Normal distribution and Identity link)')
        if self.lam is not None:
            print('The lambda value/s will be {}.'.format(self.lam))
        if self.spline_order is not None:
            print('The spline order will be {}.'.format(self.spline_order))
        if self.n_splines is not None:
            print('The number of splines will be {}.'.format(self.n_splines))
        print('--------------------------------------------------------------------------------------')


        if self.X_intercept is not None and self.y_intercept is not None:
            X_train_augmented, y_train_augmented, weights_train_augmented = self.add_intercept_train()
            print('--------------------------------------------------------------------------------------')
            print('An augmented training set has been created based on the the supplied intercepts')
            print('--------------------------------------------------------------------------------------')
        else:
            X_train_augmented, y_train_augmented, weights_train_augmented = self.X_train, self.y_train, self.sample_weights_train  
           

        study = optuna.create_study(direction='maximize',study_name='gam_parameters', sampler=optuna.samplers.TPESampler())
        study.optimize(lambda trial: self.objective(trial, X_train=X_train_augmented, y_train=y_train_augmented, weights_train=weights_train_augmented), n_trials=100)
        print('--------------------------------------------------------------------------------------')
        print('Tuning is complete')
        print('Best trial: ', study.best_trial.number)
        print('Best cross-validated r^2: {:.2f}'.format(study.best_trial.value))
        print('Best hyperparameters: {}'.format(study.best_trial.params))
        print('--------------------------------------------------------------------------------------')

        best_params = list(study.best_trial.params.values())
        
        if self.n_splines is None and self.spline_order is None:
            lam, spline_order, n_splines = best_params[0:self.n_features(self.X)], best_params[self.n_features(self.X)], best_params[self.n_features(self.X)+1]

        if self.n_splines is not None and self.spline_order is None:
            lam, spline_order = best_params[0:self.n_features(self.X)], best_params[self.n_features(self.X)]
            n_splines = self.n_splines

        if self.spline_order is not None and self.n_splines is None:
            lam, n_splines = best_params[0:self.n_features(self.X)], best_params[self.n_features(self.X)]
            spline_order = self.spline_order

        if self.n_splines is not None and self.spline_order is not None:
            lam = best_params[0:self.n_features(self.X)]
            spline_order = self.spline_order
            n_splines = self.n_splines

        gam_best =  self.make_gam(lam_list=lam, spline_order_list=spline_order, n_splines_list=n_splines)
        gam_fit = gam_best.fit(X_train_augmented, y_train_augmented, weights_train_augmented) 
        pred = gam_fit.predict(self.X_test)
        test_r2 = r2_score(self.y_test, pred, sample_weight=self.sample_weights_test)
    
        print('--------------------------------------------------------------------------------------')
        print('Final remarks')
        print('A Linear GAM is created')
        print('Lambda value/s: {}.'.format(lam))
        print('Spline order: {}.'.format(spline_order))
        print('Number of splines: {}.'.format(n_splines))
        print('Test set r^2: {}'.format(test_r2))
        print('--------------------------------------------------------------------------------------')

        return test_r2, study, gam_best, gam_fit