import pandas as pd

import sys
sys.path.insert(0, "./DataReading")
from CSVDataReader import readInCSVFile
from DataCleaning import cleanDataFramForAustria, cleanDataFramWithoutAustria
sys.path.insert(1, "./DataDiscover")
from PrintInfoFromModel import printInformationToFile
sys.path.insert(1, "./Modell")
from ProphetModell import trainModellWithProphet
from sklearn.model_selection import train_test_split
sys.path.insert(1,"./Tools")
from FolderCreator import createDirectory

df = readInCSVFile("../Data/CovidFaelle_Timeline.csv")
df = cleanDataFramWithoutAustria(df)
printInformationToFile(df, "../DataDiscovering/AustriaJustLands/")

selection = ["Time", "AnzahlFaelle"]
df = df[selection]
df.columns = ["ds", "y"]
folder = "../Output/v1/Prophet_Bundeslaender/"
createDirectory(folder)

trainModellWithProphet(df, folder)






