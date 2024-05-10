import numpy as np

from sklearn.model_selection import GridSearchCV







 
class SuperLearning:
    def __init__(self,X_train, X_test, y_train, y_test,X_pred) :
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_pred = X_pred  

    def ChooseML(self,ml):

        if ml == 'LG' :
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(solver='lbfgs', max_iter=1000)
        elif ml == 'DT':
            from sklearn.tree import DecisionTreeClassifier
            clf = DecisionTreeClassifier()
        elif ml == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier()
        elif ml == 'SVC':
            from sklearn.svm import SVC 
            clf = SVC() 
        elif ml == 'NB':
            from sklearn.naive_bayes import GaussianNB
            clf = GaussianNB()
        elif ml == 'SGD':
            from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline 
            clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=2500, tol=1e-3, penalty = 'elasticnet'))

        clf.fit(self.X_train, self.y_train)
        return clf.predict(self.X_test),clf.predict(self.X_pred),clf.score(self.X_test, self.y_test)
        
 
    def ChooseML_adv(self,ml,cv):


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
                'priors': [None, [0.2, 0.8], [0.5, 0.5]],  # Example class priors
                'var_smoothing': [1e-9, 1e-7, 1e-5]   
            }
        
        clf_ = GridSearchCV(estimator=clf,param_grid=parameters,cv=cv)
        clf_.fit(self.X_train, self.y_train)
        return clf_.predict(self.X_test),clf_.predict(self.X_pred),clf_.score(self.X_test, self.y_test)
        

class SuperviseLearning:
    def __init__(self) : 
        pass

    def MSE(self,y_pred, y_real):
      return (np.square(y_pred - y_real)).mean()
    
    def ChooseML(self,X_train, X_test, y_train, y_test,ml):
        ml = ml.upper()

        if ml == 'LG' :
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(solver='lbfgs', max_iter=100)
        elif ml == 'DT':
            from sklearn.tree import DecisionTreeClassifier
            clf = DecisionTreeClassifier()
        elif ml == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier()
        elif ml == 'SVC':
            from sklearn.svm import SVC 
            clf = SVC() 
        elif ml == 'NB':
            from sklearn.naive_bayes import GaussianNB
            clf = GaussianNB()
        elif ml == 'SGD':
            from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline 
            clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, penalty = 'elasticnet'))
        else:
            from sklearn.linear_model import LinearRegression
            clf = LinearRegression()

        try:
            clf.fit(X_train, y_train)
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
        
 
    # def ChooseML_adv(self,ml,cv):


    #     if ml == 'LG' :
    #         from sklearn.linear_model import LogisticRegression
    #         clf = LogisticRegression()
    #         parameters ={'C':[0.01,0.1,1],
    #                     'penalty':['l2'],
    #                     'solver':['lbfgs'],
    #                     'max_iter':[1000]
    #         }
    #     elif ml == 'DT':
    #         from sklearn.tree import DecisionTreeClassifier
    #         clf = DecisionTreeClassifier()
    #         parameters = {'criterion': ['gini', 'entropy'],
    #                         'splitter': ['best', 'random'],
    #                         'max_depth': [2*n for n in range(1,5)],
    #                         # 'max_features': ['auto', 'sqrt'],
    #                         'min_samples_leaf': [1, 2, 4],
    #                         'min_samples_split': [2, 5, 10]
    #         }
    #     elif ml == 'KNN':
    #         from sklearn.neighbors import KNeighborsClassifier
    #         clf = KNeighborsClassifier()
    #         parameters = {'n_neighbors': np.arange(1,5),
    #                         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    #                         'p': [1,2]
    #         }
    #     elif ml == 'SVC':
    #         from sklearn.svm import SVC 
    #         clf = SVC() 
    #         parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
    #                         'C': np.logspace(-3, 3, 5),
    #                         'gamma':np.logspace(-3, 3, 5)
    #         }
    #     elif ml == 'NB':
    #         from sklearn.naive_bayes import GaussianNB
    #         clf = GaussianNB()
    #         parameters = {
    #             'priors': [None, [0.2, 0.8], [0.5, 0.5]],  # Example class priors
    #             'var_smoothing': [1e-9, 1e-7, 1e-5]   
    #         }
    #     elif ml == 'SGD':
    #         from sklearn.linear_model import SGDClassifier
    #         from sklearn.preprocessing import StandardScaler
    #         from sklearn.pipeline import make_pipeline 
    #         clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=2500, tol=1e-3, penalty = 'elasticnet'))
    #         parameters = {
    #             'priors': [None, [0.2, 0.8], [0.5, 0.5]],  # Example class priors
    #             'var_smoothing': [1e-9, 1e-7, 1e-5]   
    #         }
        
    #     clf_ = GridSearchCV(estimator=clf,param_grid=parameters,cv=cv)
    #     clf_.fit(self.X_train, self.y_train)
    #     return clf_,clf_.score(self.X_test, self.y_test)
        


import tensorflow as tf
 
class NN: 

    def __init__(self, X_train, X_test, y_train, y_test,X_pred,activation): 
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test =  y_test
        self.X_pred =  X_pred
        self.activation =  activation

    def train_model(self,num_node, dropout_prob, lr, batch_size, epoch):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(num_node, activation=self.activation, input_shape=(self.X_train.shape[1],)),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(num_node, activation=self.activation),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(lr), 
                         loss='mean_squared_error',
                         metrics=['accuracy']
                         )
        model.fit(self.X_train, self.y_train, epochs=epoch, batch_size=batch_size, validation_split=0.2, verbose=0)

        return model 

    def DNN(self,epochs,num_nodes,dropout_probs,lrs,batch_sizes):
        least_val_loss = float('inf')
        clf = None
        epoch=epochs  
        for num_node in num_nodes: 
            for dropout_prob in dropout_probs: 
                for lr in lrs: 
                    for batch_size in batch_sizes:  
                        print(f"{num_node} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
                        model= self.train_model(num_node, dropout_prob, lr, batch_size, epoch) 
                        val_loss = model.evaluate(self.X_test, self.y_test)[0]
                        if val_loss < least_val_loss:
                            least_val_loss = val_loss
                            clf = model
        return clf,clf.evaluate(self.X_test, self.y_test)[1]
    
    def predict(self,model):
         
        y_pred = model.predict(self.X_test)
        y_pred = (y_pred > 0.5).astype(int).reshape(-1,)

        y_predpred = model.predict(self.X_pred)
        y_predpred = (y_predpred > 0.5).astype(int).reshape(-1,)

        return y_pred,y_predpred




class LinearLearning:
    def __init__(self,X_train, X_test, y_train, y_test) :
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test  

    def MSE(self,y_pred, y_real):
      return (np.square(y_pred - y_real)).mean()

    def LinearRegression(self):
        from sklearn.linear_model import LinearRegression
        clf = LinearRegression()
        clf.fit(self.X_train, self.y_train)
        clf.predict(self.X_test)
        scre = clf.score(self.X_test, self.y_test)
        MSE_ = self.MSE(self.y_test, clf.predict(self.X_test))
        return clf,scre,MSE_
