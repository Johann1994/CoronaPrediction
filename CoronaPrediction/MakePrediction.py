import pandas as pd

import sys
sys.path.insert(0, "./DataReading")
from CSVDataReader import readInCSVFile
from DataCleaning import cleanDataFramWithoutAustriaForPrediction, addWeatherInformationPrediction, addLockdownInformationForAustria
sys.path.insert(1, "./DataDiscover")
from PrintInfoFromModel import printInformationToFile
sys.path.insert(1, "./Modell")
from PrepareForLSTM import prepareDatFramesForLSTMUniqueLands
from ModelPrediction import makePrediction
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

timeStep=10

folderBestPrediction = "../Output/OmikronWeatherLockdown/2/"

x_Original,y_Original = prepareDatFramesForLSTMUniqueLands(df,folderBestPrediction, timeSteps=timeStep, loadFromFile=True)

y_Prediction = makePrediction(f"{folderBestPrediction}model.h5", x_Original)


scalerY = joblib.load(f"{folderBestPrediction}myScalerY.pkl")

df = df[["Time", "AnzahlFaelle"]]
df = df.iloc[timeStep:]

df["AnzahlPredicted"] = y_Prediction * scalerY

createDirectory("../Output/Prediction/")
df.to_excel("../Output/Prediction/Prediction_Omikron.xlsx", index=False)
