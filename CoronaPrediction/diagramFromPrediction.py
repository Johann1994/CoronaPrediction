import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import sys
sys.path.insert(0, "./DataReading")
from CSVDataReader import readInCSVFile
from DataCleaning import cleanDataFramForAustria, cleanDataFramWithoutAustria, addVaccination
sys.path.insert(1, "./DataDiscover")
from PrintInfoFromModel import printInformationToFile
sys.path.insert(1, "./Modell")
from PrepareForLSTM import prepareDatFramesForLSTMUniqueLands
from TrainModell import trainModel
from sklearn.model_selection import train_test_split
sys.path.insert(1,"./Tools")
from FolderCreator import createDirectory
from ModelPrediction import makePrediction
import joblib
from datetime import timedelta
from DataCleaning import cleanDataFramForAustria, cleanDataFramWithoutAustria, addWeatherInformation, addWeatherInformation_Val, addLockdownInformationForAustria


df = readInCSVFile("../Data/CovidFaelle_Timeline.csv")
df = cleanDataFramWithoutAustria(df)
df = addWeatherInformation(df)
df = addLockdownInformationForAustria(df)

selection = ["Time","BundeslandID", "AnzahlPlus10d", "AnzahlFaelle", "SiebenTageInzidenzFaelle" ]
#selection = ["Time","BundeslandID", "AnzahlPlus10d", "AnzahlFaelle", "SiebenTageInzidenzFaelle","Temperature","Relative Humidity","Precipitation", "Wind Speed", "LockDown" ]

df = df[selection]
df = df[df.BundeslandID == 3]
mask = (df.Time > pd.to_datetime("22.12.2021", format="%d.%m.%Y")) & (df.Time < pd.to_datetime("30.01.2022", format="%d.%m.%Y"))
dfTest = df[mask]
df = df[df.Time < pd.to_datetime("30.01.2022", format="%d.%m.%Y") ]
timeStep=10

folderPrediction = "../Output/LSTM_justCoronaDataTestLastDays/v4_MinMaxScalerScaleTo50Percent/"
x_Original,y_Original = prepareDatFramesForLSTMUniqueLands(dfTest,folderPrediction, timeSteps=timeStep, loadFromFile=True)
y_Prediction = makePrediction(f"{folderPrediction}model.h5", x_Original)
scalerY = joblib.load(f"{folderPrediction}myScalerY.pkl")

df = df[["Time", "AnzahlFaelle"]]
dfTest = dfTest[["Time", "AnzahlFaelle"]]
dfTest = dfTest.iloc[timeStep:]
dfTest["AnzahlPredicted"] = y_Prediction * scalerY
dfTest["Time"] = dfTest["Time"] - timedelta(days=10)

fig, ax = plt.subplots(1,1)
df = df[df.Time < pd.to_datetime("20.01.2022", format="%d.%m.%Y") ]

ax.plot(df.Time, df.AnzahlFaelle, 'b', label='Infizierte Corona Personen')
ax.plot(dfTest.Time, dfTest.AnzahlPredicted, 'r', label='Vorhergesagte Infizierte Personen')
ax.legend()
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(f"{folderPrediction}newDiagram.png")    