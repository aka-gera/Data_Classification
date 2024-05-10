
from sklearn.model_selection import train_test_split , RepeatedKFold
from sklearn import preprocessing   
import pandas as pd
import numpy as np
import time 
from sklearn.model_selection import GridSearchCV



class aka_clean:
    def __init__(self) -> None:
        pass 

    def df_get(self,cvs_path):
        encodings_to_try = ['utf-8', 'latin-1', 'ISO-8859-1'] 
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(cvs_path, encoding=encoding)
                break  
            except UnicodeDecodeError:
                continue   
        return df
    

    def CleaningVar(self,dfT):

        cmLabel = [ '`'+str(elm) for elm in dfT[dfT.columns[-1]].dropna().unique()]
        ncols = dfT.shape[1]

        typOfVar = []
        for j in range(ncols):
            for i,elm in dfT[dfT.columns[j]].dropna().items():
                if isinstance(elm,str):
                    typOfVar.append(j)
                    break

        mapping = {}
        swapMapping = {}

        if typOfVar is not None:
            for j in typOfVar:
                mapping[dfT.columns[j]] = {}
                uniq = dfT[dfT.columns[j]].dropna().unique()
                for i in range(len(uniq)):
                    key = uniq[i]
                    mapping[dfT.columns[j]][key] = i

            if ncols in typOfVar:
                swapMapping = {v: k for k, v in mapping[dfT.columns[-1]].items()}

        return  cmLabel,typOfVar,mapping,swapMapping



    def CleaningDF(self,df,typOfVar,mapping):

        dfTemp = df.copy()
        ncols = df.shape[1]

        if typOfVar is not None:
            for j in typOfVar:
                dfTemp[dfTemp.columns[j]] = df[df.columns[j]].map(mapping[df.columns[j]]) 

        for j in range(ncols):
            mode1 = dfTemp[dfTemp.columns[j]].mode()
            dfTemp[df.columns[j]] = dfTemp[df.columns[j]].fillna(mode1[0])  

        return  dfTemp



class aka_filter:
    def __init__(self) -> None:
        pass


    def filter_std(self,df,cols,std_number=3):
        lower_limit = df.mean() + std_number[0]*df.std()
        upper_limit = df.mean() + std_number[1]*df.std() 
        for i in cols:
            df_i = df[df.columns[i]] 
            ind = df_i[~ ((df_i>lower_limit[i]) & (df_i<upper_limit[i]))].index 
            df = df.drop(ind)
        return df


    def filter_z_score(self,df,cols,std_number=[-3,3]):

        for i in cols:
            df_i = df[df.columns[i]] 
            df_z=(df_i - df_i.mean())/df_i.std()
            ind = df_i[~ ((df_z>std_number[0]) & (df_z<std_number[1]))].index 
            df = df.drop(ind)
        return df
    

    def filter_IQR(self,df,cols,IQR_number=[-1.5,1.5]):

        for i in cols:
            df_i = df[df.columns[i]]
            q25,q75 = np.percentile(a = df_i,q=[25,75])
            IQR = q75 - q25 
            lower_limit = q25 + IQR_number[0] * IQR
            upper_limit = q75 + IQR_number[1] * IQR 
            if 10**-10*round(10**10*np.abs(IQR)) > 0.0: 
                ind = df_i[~((df_i>lower_limit) & (df_i<upper_limit))].index
                df = df.drop(ind)
        return df
    
    def Columns_Correlated(self,df,percent_correlation):
        cm = df.corr()
        n = cm.shape[0]
        corr_tmp = []
        for i in range(n):
            for j in range(0,i):
                if np.abs(cm.iloc[i,j]) >= percent_correlation:
                    corr_tmp.append([i,j])
        return corr_tmp,cm
        
    def df_standardized(self,df):
        df_tmp = (df - df.mean()) / df.std()
        return df_tmp
  
class aka_learn:
    def __init__(self) -> None:
        pass 

    def MSE(self,y_pred, y_real):
      return (np.square(y_pred - y_real)).mean()
    
    def Learning_data(self,df,pre_proc=0): 
        transform = preprocessing.StandardScaler()
        try:
            X = df[df.columns[:-1]].values
            Y = df[df.columns[-1]].to_numpy()
            if pre_proc == 'XY': 
                X = transform.fit_transform(X)
                Y = transform.fit_transform(Y)
            elif pre_proc == 'X':
                X = transform.fit_transform(X)
            elif pre_proc == 'Y': 
                Y = transform.fit_transform(Y)
        
        except Exception as e: 
            print(f"Error: {e}") 
            
        X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=42)
        
        return X_train, X_test, y_train, y_test 
        
    def Choose_ML(self,ml):
        ml = ml.upper()

        if ml == 'LG' :
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(solver='lbfgs', max_iter=500)
        elif ml == 'DT':
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier()
        elif ml == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier()
        elif ml == 'SVC':
            from sklearn.svm import SVC 
            return SVC() 
        elif ml == 'NB':
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB()
        elif ml == 'SGD':
            from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline 
            return make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, penalty = 'elasticnet'))
        elif ml == 'RFC':
            from sklearn.ensemble import RandomForestClassifier
            return  RandomForestClassifier(objective="binary:logistic", random_state=42) 
        elif ml == 'ABC':
            from sklearn.ensemble import AdaBoostClassifier
            return  AdaBoostClassifier( random_state=42)
        else:
            from sklearn.linear_model import LinearRegression
            return LinearRegression() 


    def Choose_ML(self,ml,cv):
        ml = ml.upper() 

        if ml == 'LG' :
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
            parameters ={'C':[0.01,0.1,1],
                        'penalty':['l2'],
                        'solver':['lbfgs'],
                        'max_iter':[1000]
            }
        elif ml == 'DT':
            from sklearn.tree import DecisionTreeClassifier
            clf = DecisionTreeClassifier()
            parameters = {'criterion': ['gini', 'entropy'],
                            'splitter': ['best', 'random'],
                            'max_depth': [2*n for n in range(1,5)],
                            # 'max_features': ['auto', 'sqrt'],
                            'min_samples_leaf': [1, 2, 4],
                            'min_samples_split': [2, 5, 10]
            }
        elif ml == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier()
            parameters = {'n_neighbors': np.arange(1,5),
                            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                            'p': [1,2]
            }
        elif ml == 'SVC':
            from sklearn.svm import SVC 
            clf = SVC() 
            parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
                            'C': np.logspace(-3, 3, 5),
                            'gamma':np.logspace(-3, 3, 5)
            }
        elif ml == 'NB':
            from sklearn.naive_bayes import GaussianNB
            clf = GaussianNB()
            parameters = {
                'priors': [None, [0.2, 0.8], [0.5, 0.5]],  # Example class priors
                'var_smoothing': [1e-9, 1e-7, 1e-5]   
            }
        elif ml == 'SGD':
            from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline 
            clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=2500, tol=1e-3, penalty = 'elasticnet'))
            parameters = {
                'priors': [None, [0.2, 0.8], [0.5, 0.5]], 
                'var_smoothing': [1e-9, 1e-7, 1e-5]   
            }
        elif ml == 'RFC':
            from sklearn.ensemble import RandomForestClassifier
            clf =  RandomForestClassifier( random_state=42) 
            parameters = {
                # 'learning_rate': [0.1, 0.01, 0.05],
                # 'max_depth': [3, 5, 7],
                'n_estimators': [70, 100, 200,500]
            } 
        elif ml == 'ABC':
            from sklearn.ensemble import AdaBoostClassifier
            clf =  AdaBoostClassifier( random_state=42)
            parameters = {
                'learning_rate': [0.1, 0.01, 0.05],
                # 'max_depth': [3, 5, 7],
                'n_estimators': [70, 100, 200,500]
            }  
        
        # scoring = {
        #     "accuracy": "accuracy",
        #     "precision": "precision",
        #     "recall": "recall",
        #     "f1": "f1",
        # # }
        # clf_ = GridSearchCV(estimator=clf,param_grid=parameters, scoring=scoring, refit="recall",cv=cv) 
        clf_ = GridSearchCV(estimator=clf,param_grid=parameters, refit="recall",cv=cv) 
        return clf_
        

    def Train_ML(self,clf,X_train, X_test, y_train, y_test):
        try:
            clf.fit(X_train, y_train)
            try: 
                clf_params = clf.best_params_
                clf = clf.best_estimator_
                MSE_ = self.MSE(y_test, clf.predict(X_test))
                scre_ =  clf.score(X_test, y_test)
            except :
                MSE_ = self.MSE(y_test, clf.predict(X_test))
                scre_ =  clf.score(X_test, y_test)
        except ValueError:
            try:  
                scre_ = 0
                MSE_ = 1111
                clf.fit(X_train, y_train.reshape(-1))
                MSE_ = self.MSE(y_test.reshape(-1), clf.predict(X_test)) 
                scre_ = clf.score(X_test, y_test.reshape(-1))
            except Exception as e: 
                print(f"Error: {e}") 
 
        return clf,scre_, MSE_
        
 
     
 


    def ML(self,df,mls,mach,pre_proc,confidence_interval_limit,correlation_percentage_threshold,disp_dash):
        cols = range(df.shape[1])
        scre_max = -10
        print(f"conf_inter  corr_per   df_.shape   size_removed  ML   score      MSE    simul_time(min)")
        print("_______________________________________________________________________________________")
        for mm in confidence_interval_limit: 
            std_number = [-mm,mm]
            df_tmp = df.copy()
            df_tmp = aka_filter().filter_std(df_tmp,cols,std_number)
            for corr_per in correlation_percentage_threshold:
                df_ = df_tmp.copy()
                corr_tmp,cm = aka_filter().Columns_Correlated(df_,corr_per)
                if len(corr_tmp) != 0:
                    uniq_ = np.unique([item[1] for item in corr_tmp])
                    df_ = df_.drop( df_.columns[uniq_], axis=1)

                X_train, X_test, y_train, y_test = self.Learning_data(df_,pre_proc) 
                for ml in mls:
                    if mach == 'adv':
                        # n_splits, n_repeats = 5,3
                        cv = 5 #RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)  
                        clf = self.Choose_ML(ml,cv=cv)
                    else:
                        clf = self.Choose_ML(ml)
                    start_time = time.time()
                    clf,scre,MSE_ = self.Train_ML(clf,X_train, X_test, y_train, y_test)
                    total_time = (time.time() - start_time)/60
                    if disp_dash == 'all':  
                        print(f"  {mm:.1f}         {corr_per:.1f}    {df_.shape}       {len(corr_tmp)}        {ml}   {scre:.3f}     {MSE_:.3f}       {total_time:.2f} ")

                    if scre_max < scre:
                        if disp_dash == 'sup':
                            print(f"  {mm:.1f}         {corr_per:.1f}    {df_.shape}       {len(corr_tmp)}        {ml}   {scre:.3f}     {MSE_:.3f}       {total_time:.2f} ") 
                        scre_max = scre
                        corr_per_max = corr_per
                        corr_tmp_max = corr_tmp
                        std_number_max = std_number
                        ml_max = ml
        return clf, scre_max, corr_per_max,corr_tmp_max, std_number_max, ml_max


        



    def Search_ML(self,df,mls,mach,pre_proc,confidence_interval_limit,correlation_percentage_threshold,disp_dash,file_name,file_name_scre):
        cols = range(df.shape[1])
        scre_max = -10
                
        output_filename = f"{file_name}_Output.txt"
        with open(output_filename, "w") as output_file:
            f_0 = f"conf_inter  corr_per   df_.shape   size_removed  ML   score      MSE    simul_time(min)"
            f_1 =f"_______________________________________________________________________________________"
            print(f_0)
            print(f_1)
            
            output_file.write( f"conf_inter  corr_per   df_.shape   size_removed  ML   score      MSE    simul_time(min) \n")
            for mm in confidence_interval_limit: 
                std_number = [-mm,mm]
                df_tmp = df.copy()
                df_tmp = aka_filter().filter_std(df_tmp,cols,std_number)
                for corr_per in correlation_percentage_threshold:
                    df_ = df_tmp.copy()
                    corr_tmp,_ = aka_filter().Columns_Correlated(df_,corr_per)
                    if len(corr_tmp) != 0:
                        uniq_ = np.unique([item[1] for item in corr_tmp])
                        df_ = df_.drop( df_.columns[uniq_], axis=1)

                    X_train, X_test, y_train, y_test = self.Learning_data(df_,pre_proc) 
                    for ml in mls:
                        if mach == 'adv':
                            # n_splits, n_repeats = 5,3
                            cv = 5 #RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)  
                            clf = self.Choose_ML(ml,cv=cv)
                        else:
                            clf = self.Choose_ML(ml)
                        start_time = time.time()
                        clf,scre,MSE_ = self.Train_ML(clf,X_train, X_test, y_train, y_test)
                        total_time = (time.time() - start_time)/60
                        if disp_dash == 'all':  
                            print(f"  {mm:.1f}         {corr_per:.1f}    {df_.shape}       {len(corr_tmp)}        {ml}   {scre:.3f}     {MSE_:.3f}       {total_time:.2f} ")
                        if scre > file_name_scre: 
                            output_file.write(f"  {mm:.3f}         {corr_per:.3f}    {df_.shape}       {len(corr_tmp)}        {ml}   {scre:.3f}     {MSE_:.3f}       {total_time:.2f} \n")
                            if disp_dash == 'write':
                                print(f"  {mm:.1f}         {corr_per:.1f}    {df_.shape}       {len(corr_tmp)}        {ml}   {scre:.3f}     {MSE_:.3f}       {total_time:.2f} ") 
 
                        if scre_max < scre:
                            if disp_dash == 'sup':
                                print(f"  {mm:.1f}         {corr_per:.1f}    {df_.shape}       {len(corr_tmp)}        {ml}   {scre:.3f}     {MSE_:.3f}       {total_time:.2f} ") 
 


            

