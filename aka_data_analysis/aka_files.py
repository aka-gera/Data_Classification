
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff 
import plotly.express as px

import pandas as pd
 
# from dash import html



class dash_deco:
    def __init__(self) -> None:
        self.box_style={
            'width':'80%',
            'padding':'3px',
            'font-size': '20px',
            'text-align-last' : 'center' ,
            'margin': 'auto',   
            'background-color' : 'black',
            'color': 'black'
            } 
        self.default_style ={'textAlign': 'center',
                             'color': 'white', 
                             'background-color' : 'black',   
                             'font-size': 20
                             }
        self.default_style_buttom ={'textAlign': 'center',
                             'color': 'black', 
                             'background-color' : 'grey',   
                             'font-size': 20
                             }
        self.app_style ={
                            'color' : 'black',
                            'backgroundColor': 'black',  # Set the background color of the app here
                            'height': '100vh'  # Set the height of the app to fill the viewport
                        }
 
        self.signature = html.Div([
            html.A(
                html.Img(src='https://img.icons8.com/color/48/000000/github.png'),
                href='https://github.com/aka-gera',
                target='_blank'
            ),
            html.A(
                html.Img(src='https://img.icons8.com/color/48/000000/linkedin.png'),
                href='https://www.linkedin.com/in/aka-gera/',
                target='_blank'
            ),
            html.A(
                html.Img(src='https://img.icons8.com/color/48/000000/youtube.png'),
                href='https://www.youtube.com/@aka-Gera',
                target='_blank'
            ),
        ], style={'display': 'flex', 'justify-content': 'center'})


class plot_dash :

    def __init__(self,df, tcouleur='plotly_dark', bcouleur='navy', fcouleur='white', fsize=20):
        self.tcouleur = tcouleur
        self.bcouleur = bcouleur
        self.fcouleur = fcouleur
        self.fsize = fsize
        self.df = df 
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
    
    def plot_history_dash(self,feat):
        fig = px.histogram(data_frame= self.df,x=feat,opacity= 0.7)
        fig.update_layout(**self.update_layout_parameter) 
        return fig

    def plot_history_all_dash(self):
        fig  = px.histogram(data_frame= self.df,opacity= .7).update_xaxes(categoryorder='total descending')
        fig.update_layout(**self.update_layout_parameter) 
        fig.update_xaxes(**self.update_axes)
        return fig
        

        
    def plot_confusion_matrix_dash(self,y,y_predict,cmLabel,lab):
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

    def plot_classification_report_dash(self,y, y_predict,cmLabel,lab):

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
            fig.update_yaxes(
                    title_text = 'y', 
                    showticklabels=False   
                    )
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
  