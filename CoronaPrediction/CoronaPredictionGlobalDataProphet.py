import pandas as pd

import sys
sys.path.insert(0, "./DataReading")
from CSVDataReader import readInCSVFile
from DataCleaning import prepareDataframeJohnHopkinsUniversity
sys.path.insert(1, "./Modell")
from ProphetModell import trainModellWithProphet
sys.path.insert(1,"./Tools")
from FolderCreator import createDirectory

df = readInCSVFile('../Data/COVID-19-master/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', ',')
ListOfDF = prepareDataframeJohnHopkinsUniversity(df)

appendAllTogether = pd.DataFrame()
selection = ["Time", "Confirmed"]
for myDf in ListOfDF:
    myDf = myDf[selection]
    myDf.columns = ["ds", "y"]

    appendAllTogether = appendAllTogether.append(myDf)

folder = "../Output/v1/Prophet_John_Hopkins/"
createDirectory(folder)

trainModellWithProphet(appendAllTogether, folder)