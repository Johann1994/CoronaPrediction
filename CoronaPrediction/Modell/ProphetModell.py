import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from datetime import timedelta

def trainModellWithProphet(df, folder):
    model = Prophet()
    model.fit(df)

    # define the period for which we want a prediction
    future = list()
    lastDate = df["ds"].iloc[-1]
    for i in range(1, 13):
	    date = lastDate + timedelta(days=i)
	    future.append([date])
    future = pd.DataFrame(future)
    future.columns = ['ds']
    # use the model to make a forecast
    forecast = model.predict(future)

    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(folder + "Prediction.csv")
    model.plot(forecast)
    plt.savefig(folder + "Trend.png")

