

import pandas as pd

import sys
sys.path.insert(0, "./DataReading")
from CSVDataReader import readInCSVFile
from DataCleaning import cleanDataFramForAustria, cleanDataFramWithoutAustria, addVaccination, addWeekDays
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
df = addVaccination(df)
df = addWeekDays(df)
dfValTest = readInCSVFile("../Data/CovidFaelle_TestVal.csv")
dfValTest = cleanDataFramWithoutAustria(dfValTest)
dfValTest = addVaccination(dfValTest)
dfValTest = addWeekDays(dfValTest)

#printInformationToFile(df, "../DataDiscovering/AustriaJustLands/")

selection = ["Time","BundeslandID", "AnzahlPlus10d", "AnzahlFaelle", "SiebenTageInzidenzFaelle","VacDose2", "VacDose3", "weekDay" ]
df = df[selection]
dfValTest = dfValTest[selection]
folder = "../Output/LockDownAndWeekDays_LastDays/3/"
createDirectory(folder)

dfVal = dfValTest[dfValTest.BundeslandID == 1]
dfTest = dfValTest[dfValTest.BundeslandID == 3]

timeStep=10

x_train,y_train = prepareDatFramesForLSTMUniqueLands(df, folder, timeSteps=timeStep, loadFromFile=False)

x_test, y_test = prepareDatFramesForLSTMUniqueLands(dfTest, folder, timeSteps=timeStep, loadFromFile=True)
x_val, y_val = prepareDatFramesForLSTMUniqueLands(dfVal, folder, timeSteps=timeStep, loadFromFile=True)

trainModel(x_train, y_train, x_val, y_val, x_test, y_test, folder)



