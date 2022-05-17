
import pandas as pd

import sys
sys.path.insert(0, "./DataReading")
from CSVDataReader import readInCSVFile
from DataCleaning import cleanDataFramWithoutAustriaForPrediction, addWeatherInformationPrediction, addLockdownInformationForAustria
sys.path.insert(1, "./DataDiscover")
from PrintInfoFromModel import printInformationToFile
sys.path.insert(1, "./Modell")
from PrepareForLSTM import prepareDatFramesForLSTMUniqueLands
from ModelPrediction import makePrediction7d
sys.path.insert(1,"./Tools")
from FolderCreator import createDirectory
import joblib

df = readInCSVFile("../Data/CovidFaelle_Prediction.csv")
df = cleanDataFramWithoutAustriaForPrediction(df)
df = addWeatherInformationPrediction(df)
df = addLockdownInformationForAustria(df)

df = df[df.BundeslandID == 6]
selection = ["Time","BundeslandID", "AnzahlPlus10d", "AnzahlFaelle", "SiebenTageInzidenzFaelle","Temperature","Relative Humidity","Precipitation", "Wind Speed", "LockDown" ]

df = df[selection]
df = df[df.Time > pd.to_datetime("04.02.2022", format="%d.%m.%Y")]

timeStep=10

folderBestPrediction = "../Output/OmikronWeatherLockdown7d/2/"

x_Original,y_Original = prepareDatFramesForLSTMUniqueLands(df,folderBestPrediction, timeSteps=timeStep, loadFromFile=True)

y_Prediction = makePrediction7d(f"{folderBestPrediction}model.h5", x_Original)
y_Prediction = y_Prediction[0]

scalerY = joblib.load(f"{folderBestPrediction}myScalerY.pkl")

df = df[["Time", "AnzahlFaelle"]]
df = df.iloc[timeStep:]
y_Prediction = y_Prediction * scalerY

df["AnzahlPredicted"] = 0
for i in range(0, 7):
    df[i, df.columns() == "AnzahlPredicted"] = y_Prediction[i]

createDirectory("../Output/Prediction7d/")
df.to_excel("../Output/Prediction7d/Prediction_Omikron.xlsx", index=False)
