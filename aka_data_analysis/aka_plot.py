

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff 
import plotly.express as px

import math
from plotly.subplots import make_subplots

import pandas as pd
 



class aka_plot :

    def __init__(self, tcouleur='plotly_dark', bcouleur='navy', fcouleur='white', fsize=20):
        self.tcouleur = tcouleur
        self.bcouleur = bcouleur
        self.fcouleur = fcouleur
        self.fsize = fsize 
        self.update_layout_parameter = dict(        
                                        barmode='overlay',  
                                        font=dict(color=fcouleur,size=fsize),  
                                        title_x=0.5,
                                        title_y=0.9,
                                        template=self.tcouleur
                                        )
        self.update_axes = dict(  
                            title_font = {"size": 14},
                            title_standoff = 25
                            )
    
    def plot_history(self,df,feat):
        fig = px.histogram(data_frame= df,x=feat,opacity= 0.7)
        fig.update_layout(**self.update_layout_parameter) 
        return fig

    def plot_history_compare(self,df,df_,feat):
      df_0 = df[df.columns[feat]]
      df_0['filtered'] = df_[df.columns[feat]]
      fig = px.histogram(data_frame= df_0,opacity= 0.7)
      fig.update_xaxes(categoryorder='total descending')
      fig.update_layout(**self.update_layout_parameter) 
      return fig

    def plot_history_all(self,df):
        fig  = px.histogram(data_frame= df,opacity= .7).update_xaxes(categoryorder='total descending')
        fig.update_layout(**self.update_layout_parameter) 
        fig.update_xaxes(**self.update_axes)
        return fig
        

        
    def plot_confusion_matrix(self,y,y_predict,cmLabel,lab):
        cm = confusion_matrix(y, y_predict)
        if lab == 1:
            fig = ff.create_annotated_heatmap(cm,
                                            x=cmLabel[:cm.shape[1]],
                                            y=cmLabel[:cm.shape[1]],
                                            colorscale='Viridis',showscale=True)
            fig.update_xaxes(
                    title_text='Predicted labels', 
                    side='bottom')
            fig.update_yaxes(title_text = 'True labels')
        else:
            annotation_text = [['' for _ in range(cm.shape[1])] for _ in range(cm.shape[0])]
            fig = ff.create_annotated_heatmap(cm,
                                            x=cmLabel[:cm.shape[1]],
                                            y=cmLabel[:cm.shape[1]],
                                            colorscale='Viridis',
                                            annotation_text=annotation_text,
                                            showscale=True)
            fig.update_xaxes(
                    title_text='Prediction', 
                    side='bottom')
            fig.update_xaxes( showticklabels=True )
            fig.update_yaxes(title_text = 'True Solution')
            fig.update_yaxes(showticklabels=True )

        fig.update_layout(title='Confusion Matrix') 
        fig.update_layout(**self.update_layout_parameter)
        fig.update_xaxes(**self.update_axes)
        fig.update_yaxes(**self.update_axes)

        return fig

    def plot_classification_report(self,y, y_predict,cmLabel,lab): 

        report_str = classification_report(y, y_predict,  zero_division=0)
        report_lines = report_str.split('\n')

        # Remove empty lines
        report_lines = [line for line in report_lines if line.strip()]
        data = [line.split() for line in report_lines[1:]]
        colss = ['feature', 'precision',   'recall',  'f1-score',   'support', 'n1one']

        # Convert to a DataFrame
        report_df = pd.DataFrame(data, columns = colss )
        report_df = report_df[report_df.columns[:-1]]
        cm = report_df.iloc[:-3,1:].apply(pd.to_numeric).values
        colss1 = [  'precision',   'recall',  'f1-score',   'support']

        if lab == 1:
            fig = ff.create_annotated_heatmap(cm,
                                                x = colss1,
                                                y = cmLabel[:cm.shape[0]],
                                                colorscale='Viridis' )
            # fig.update_yaxes(
            #         title_text = 'y', 
            #         showticklabels=False   
            #         )
        else:
            cmm =  cm[:,:-1]
            annotation_text = [['' for _ in range(cmm.shape[1])] for _ in range(cmm.shape[0])]
            fig = ff.create_annotated_heatmap(cmm,
                                                x = colss1[:-1],
                                                colorscale='Viridis',
                                                showscale=True,
                                                annotation_text=annotation_text )
            fig.update_yaxes(
                    title_text = 'y', 
                    showticklabels=False  
                    )
        fig.update_layout(title='Classification Report')
        fig.update_layout(**self.update_layout_parameter) 
        fig.update_xaxes(**self.update_axes)
        fig.update_yaxes(**self.update_axes) 

        return fig



class aka_correlation_analysis:
  def __init__(self, tcouleur='plotly_dark', bcouleur='navy', fcouleur='white', fsize=20):
      self.tcouleur = tcouleur
      self.bcouleur = bcouleur
      self.fcouleur = fcouleur
      self.fsize = fsize
      self.update_layout_parameter = dict(
                                      barmode='overlay',
                                      font=dict(color=fcouleur,size=fsize),
                                      title_x=0.5,
                                      title_y=0.9,
                                      template=self.tcouleur
                                      )
      self.update_axes = dict(
                          title_font = {"size": 14},
                          title_standoff = 25
                          )

  def Plot_Correlation_Matrix(self,df):
    cm = df.corr()

    fig = px.imshow(cm, labels=dict(color="Correlation"), x=cm.columns, y=cm.index)

    fig.update_layout(**self.update_layout_parameter)
    fig.update_xaxes(**self.update_axes)
    fig.update_layout(
        height=800 ,
        width=900
    )
    return fig

  def Plot_Correlate_Features(self,df,corr_tmp,fig_size_row,fig_size_col,subplot_col):

    if len(corr_tmp) > 0:
        nrow = math.ceil(len(corr_tmp)/subplot_col)
        # Create subplots
        fig = make_subplots(
            rows=nrow, cols=subplot_col,
            subplot_titles=[""] * len(corr_tmp))

        for i, corr in enumerate(corr_tmp, 1):
            scatter_fig = px.scatter(df, x=df.columns[corr[0]], y=df.columns[corr[1]],
                                    trendline="ols")
            ind_col = int((i+subplot_col-1)%subplot_col+1)
            ind_row = int(1+(i-ind_col)/subplot_col)

            for j,trace in enumerate(scatter_fig.data):
                if j != 1 or j != 2 or j != 3:
                    fig.add_trace(scatter_fig.data[j], row=ind_row, col=ind_col)
                fig.update_xaxes(title_text=df.columns[corr[0]], row=ind_row, col=ind_col)
                fig.update_yaxes(title_text=df.columns[corr[1]], row=ind_row, col=ind_col)

        fig.update_layout(**self.update_layout_parameter)
        # fig.update_xaxes(**update_axes)
        fig.update_layout(
            height=fig_size_row * nrow,
            width=fig_size_col * subplot_col           # Adjust the width as needed
        )
        return fig
    else:
        print("Empty list is provided.")

