import dash
from dash import dcc # import dash_core_components as dcc
from dash import html # import dash_html_components as html
from dash.dependencies import Input, Output

from keras.models import load_model, Sequential
from keras.layers import LSTM, Dropout, Dense

import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np


app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))

def predict(data):
    # Load the data
    # df_nse = pd.read_csv("./data/NSE-Tata-Global-Beverages-Limited.csv")
    df = pd.read_csv(data)

    # Create a new dataframe with only the 'Close' column
    df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
    df.index=df['Date']

    data=df.sort_index(ascending=True,axis=0)
    new_data=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

    for i in range(0,len(data)):
        new_data["Date"][i]=data['Date'][i]
        new_data["Close"][i]=data["Close"][i]

    new_data.index=new_data.Date
    new_data.drop("Date",axis=1,inplace=True)

    dataset=new_data.values

    train=dataset[0:987,:]
    valid=dataset[987:,:]

    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)

    x_train,y_train=[],[]

    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    
    x_train,y_train=np.array(x_train),np.array(y_train)

    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    # Create and fit the LSTM network
    # model=load_model("./saved_model.h5")
    model=Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    inputs=new_data[len(new_data)-len(valid)-60:].values
    inputs=inputs.reshape(-1,1)
    inputs=scaler.transform(inputs)

    X_test=[]
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test=np.array(X_test)

    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    closing_price=model.predict(X_test)
    closing_price=scaler.inverse_transform(closing_price)

    train=new_data[:987]
    valid=new_data[987:]
    valid['Predictions']=closing_price

    return train, valid

# Predict the closing prices
train_btc, valid_btc = predict("./data/BTC-USD.csv")
train_eth, valid_eth = predict("./data/ETH-USD.csv")
train_ada, valid_ada = predict("./data/ADA-USD.csv")

# Visualize the data with Plotly
app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='BTC-USD',children=[
            html.Div([
                html.H2("Actual closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=train_btc.index,
                                y=valid_btc["Close"],
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='Scatter Plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }

                ),
                html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=valid_btc.index,
                                y=valid_btc["Predictions"],
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='Scatter Plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }

                )                
            ])                
        ]),

        dcc.Tab(label='ETH-USD',children=[
            html.Div([
                html.H2("Actual closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=train_eth.index,
                                y=valid_eth["Close"],
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='Scatter Plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }

                ),
                html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=valid_eth.index,
                                y=valid_eth["Predictions"],
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='Scatter Plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }

                )                
            ])                
        ]),

        dcc.Tab(label='ADA-USD',children=[
            html.Div([
                html.H2("Actual closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=train_ada.index,
                                y=valid_ada["Close"],
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='Scatter Plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }

                ),
                html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=valid_ada.index,
                                y=valid_ada["Predictions"],
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='Scatter Plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }

                )                
            ])                
        ]),

    ])
])


if __name__=='__main__':
    app.run_server(debug=True)
