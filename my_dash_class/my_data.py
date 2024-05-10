from sklearn.model_selection import train_test_split
from sklearn import preprocessing   
import pandas as pd
import numpy as np

 
import base64
import io  



class cleanData:

    def __init__(self):
        self.data = None 
        
    def CleaningVar(self,dfT):

        cmLabel = [ '`'+str(elm) for elm in dfT[dfT.columns[-1]].dropna().unique()]
        _,ncols = dfT.shape

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

        dfTemp = df
        _,ncols = df.shape

        if typOfVar is not None:
            for j in typOfVar:
                dfTemp[dfTemp.columns[j]] = df[df.columns[j]].map(mapping[df.columns[j]])


        for j in range(ncols):
            mode1 = dfTemp[dfTemp.columns[j]].mode()
            dfTemp[df.columns[j]] = dfTemp[df.columns[j]].fillna(mode1[0])


        return  dfTemp


    def Algorithm(self,df):

        cmLabel,typOfVar,mapping,swapMapping = self.CleaningVar(df)
         
        df1 = self.CleaningDF(df,typOfVar,mapping)
        df1.columns = ['Feature'+str(i) for i in range(df.shape[1])] #['`'+elm for elm in df.columns]
 
        X = df1[df1.columns[:-1]].values
        Y = df1[df1.columns[-1]].values
 
        X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=4)

        return  df1,X_train, X_test, y_train, y_test, cmLabel,typOfVar,swapMapping

# class getData:

#     def __init__(self,df,dfpred):
#         self.df = df
#         self.dfpred = dfpred

        
#     def CleaningVar(self,dfT):

#         cmLabel = [ '`'+str(elm) for elm in dfT[dfT.columns[-1]].dropna().unique()]
#         _,ncols = dfT.shape

#         typOfVar = []
#         for j in range(ncols):
#             for i,elm in dfT[dfT.columns[j]].dropna().items():
#                 if isinstance(elm,str):
#                     typOfVar.append(j)
#                     break

#         mapping = {}
#         swapMapping = {}

#         if typOfVar is not None:
#             for j in typOfVar:
#                 mapping[dfT.columns[j]] = {}
#                 uniq = dfT[dfT.columns[j]].dropna().unique()
#                 for i in range(len(uniq)):
#                     key = uniq[i]
#                     mapping[dfT.columns[j]][key] = i

#             if ncols in typOfVar:
#                 swapMapping = {v: k for k, v in mapping[dfT.columns[-1]].items()}

#         return  cmLabel,typOfVar,mapping,swapMapping

#     def CleaningDF(self,df,typOfVar,mapping):

#         dfTemp = df
#         _,ncols = df.shape

#         if typOfVar is not None:
#             for j in typOfVar:
#                 dfTemp[dfTemp.columns[j]] = df[df.columns[j]].map(mapping[df.columns[j]])


#         for j in range(ncols):
#             mode1 = dfTemp[dfTemp.columns[j]].mode()
#             dfTemp[df.columns[j]] = dfTemp[df.columns[j]].fillna(mode1[0])


#         return  dfTemp

    
 
#     def Algorithm(self):

#         cmLabel,typOfVar,mapping,swapMapping = self.CleaningVar(self.df)
         
#         df1 = self.CleaningDF(self.df,typOfVar,mapping)
#         df1.columns = ['Feature'+str(i) for i in range(self.df.shape[1])] #['`'+elm for elm in df.columns]

#         dfpred1 = self.CleaningDF(self.dfpred,typOfVar,mapping)
#         dfpred1.columns =['Feature'+str(i) for i in range(self.dfpred.shape[1])]

#         X = df1[df1.columns[:-1]].values
#         Y = df1[df1.columns[-1]].values

#         X_pred = dfpred1[df1.columns[:-1]].values

#         X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=4)

#         return  df1,X_train, X_test, y_train, y_test,X_pred, cmLabel,typOfVar,swapMapping

class DashToDataFrame:
    def parse_contents(self,contents, filename):
        _, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)
        try:
            if 'data' in filename or 'csv' in filename:  
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            print(e)
            return None   

        return df   

    def dash_to_df(self,list_of_contents, list_of_names):
        
        children_and_df = [self.parse_contents(c, n) for c, n in zip(list_of_contents, list_of_names)]

        df = [item for item in children_and_df if item is not None][0]   

        return df

    def text_to_list_float(self,input_text): 

        input_list = input_text.split(',') 
        input_list = [item.strip() for item in input_list] 
        input_list = [float(item) for item in input_list if item]
        return input_list

    def text_to_list_int(self,input_text): 

        input_list = input_text.split(',') 
        input_list = [item.strip() for item in input_list] 
        input_list = [int(item) for item in input_list if item]
        return input_list 



class download:

    def dfDownload(self,data):

        df = pd.DataFrame(data)
        csv_content = df.to_csv(index=False) 

        file_dict = {
            "content": csv_content,
            "filename": "prediction.csv",
            "type": "text/csv",   
        }

        return file_dict


class df_filter:
  def __init__(self) -> None:
    pass

  def filter_std(self,df,std_number,cols):
    lower_limit = df.mean() + std_number[0]*df.std()
    upper_limit = df.mean() + std_number[1]*df.std() 
    for i in cols:
      df_i = df[df.columns[i]] 
      ind = df_i[~ ((df_i>lower_limit[i])&(df_i<upper_limit[i]))].index 
      df = df.drop(ind)
    return df


  def filter_z_score(self,df,std_number,cols):

    for i in cols:
      df_i = df[df.columns[i]] 
      df_z=(df_i - df_i.mean())/df_i.std()
      ind = df_i[~ ((df_z>std_number[0])&(df_z<std_number[1]))].index 
      df = df.drop(ind)
    return df
  

  def filter_IQR(self,df,IQR_number,cols):

    for i in cols:
      df_i = df[df.columns[i]]
      q25,q75 = np.percentile(a = df_i,q=[25,75])
      IQR = q75 - q25 
      lower_limit = q25 + IQR_number[0] * IQR
      upper_limit = q75 + IQR_number[1] * IQR 
      if 10**-10*round(10**10*np.abs(IQR)) > 0.0: 
        ind = df_i[~((df_i>lower_limit)&(df_i<upper_limit))].index
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

class df_LearningData:
  def __init__(self) -> None:
    pass 
  def df_standardized(self,df):
    df_tmp = (df - df.mean()) / df.std()
    return df_tmp
  
  def Learning_data(self,df,pre_proc=0): 
    try:
        X = df[df.columns[:-1]].values
        Y = df[df.columns[-1]].values
        if pre_proc == 'XY':
            X = self.df_standardized(df[df.columns[:-1]]).values
            Y = self.df_standardized(df[df.columns[-1]]).values
        elif pre_proc == 'X':
            X = self.df_standardized(df[df.columns[:-1]]).values 
        elif pre_proc == 'Y': 
            Y = self.df_standardized(df[df.columns[-1]]).values
    
    except Exception as e: 
        print(f"Error: {e}") 
       
       
    
    # if pre_proc == 'X':
    #     transform = preprocessing.StandardScaler( copy=True)
    #     X = transform.fit_transform(X) 
    # elif pre_proc == 'Y':
    #     transform = preprocessing.StandardScaler( copy=True)
    #     Y = transform.fit_transform(Y)  
    # elif pre_proc == 'XY':
    #     transform = preprocessing.StandardScaler( copy=True)
    #     X = transform.fit_transform(X)
    #     Y = transform.fit_transform(Y) 
    # elif pre_proc == 'Y_R':
    #     transform = preprocessing.StandardScaler()
    #     Y = transform.fit_transform(Y.reshape(-1, 1))  
    # elif pre_proc == 'XY_R':
    #     transform = preprocessing.StandardScaler()
    #     X = transform.fit_transform(X)
    #     Y = transform.fit_transform(Y.reshape(-1, 1)) 
       
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=4)
    
    return X_train, X_test, y_train, y_test 
   



