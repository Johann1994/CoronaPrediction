import pandas as pd

import sys
sys.path.insert(0, "./DataReading")
from CSVDataReader import readInCSVFile
from DataCleaning import cleanDataFramForAustria, cleanDataFramWithoutAustria, addWeatherInformation
sys.path.insert(1, "./DataDiscover")
from PrintInfoFromModel import printInformationToFile
sys.path.insert(1, "./Modell")
from PrepareForLSTM import prepareDatFramesForLSTMUniqueLands
from TrainModell import trainModel
from sklearn.model_selection import train_test_split
sys.path.insert(1,"./Tools")
from FolderCreator import createDirectory

df = readInCSVFile("../Data/CovidFaelle_Timeline.csv")
df = cleanDataFramWithoutAustria(df)
df = addWeatherInformation(df)

#printInformationToFile(df, "../DataDiscovering/AustriaJustLands/")

selection = ["Time","BundeslandID", "AnzahlPlus10d", "AnzahlFaelle", "SiebenTageInzidenzFaelle", "Temperature" ]
df = df[selection]
folder = "../Output/WithWeather/v7_JustTemperature/2/"
createDirectory(folder)

dfVal = df[df.BundeslandID == 1]
dfTest = df[df.BundeslandID == 3]
df = df[df.BundeslandID != 1 ]
df = df[df.BundeslandID != 3 ]

timeStep=15

x_train,y_train = prepareDatFramesForLSTMUniqueLands(df, folder, timeSteps=timeStep, loadFromFile=False)

x_test, y_test = prepareDatFramesForLSTMUniqueLands(dfTest, folder, timeSteps=timeStep, loadFromFile=True)
x_val, y_val = prepareDatFramesForLSTMUniqueLands(dfVal, folder, timeSteps=timeStep, loadFromFile=True)

trainModel(x_train, y_train, x_val, y_val, x_test, y_test, folder)
