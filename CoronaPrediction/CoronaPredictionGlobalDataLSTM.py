import pandas as pd
import numpy as np

import sys
sys.path.insert(0, "./DataReading")
from CSVDataReader import readInCSVFile
from DataCleaning import prepareDataframeJohnHopkinsUniversity
sys.path.insert(1, "./Modell")
from PrepareForLSTM import prepareDatFrames
from TrainModell import trainModel
sys.path.insert(1,"./Tools")
from FolderCreator import createDirectory

df = readInCSVFile('../Data/COVID-19-master/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', ',')
ListOfDF = prepareDataframeJohnHopkinsUniversity(df)

selection = ["Time","Confirmed", "Prediction10" ]
loadFromFile = False

x_train = np.empty(1)
y_train = np.empty(1)
x_val = 0
y_val = 0
x_test = 0
y_test = 0
counter = 0
folder = '../Output/v2/Global/'
createDirectory(folder)
for myDf in ListOfDF :

    x,y = prepareDatFrames(myDf, folder, 60, loadFromFile)
    loadFromFile = True
    if counter == 10 :
        x_text = x
        y_test = y
    elif counter == 50 :
        x_val = x
        y_val = y
    elif x_train.shape[0] == 1:
        x_train = x
        y_train = y
    else:
        x_train = np.concatenate((x_train, x), axis=0)
        y_train = np.concatenate((y_train, y), axis=0)
        
    counter = counter + 1

trainModel(x_train, y_train, x_val, y_val, x_test, y_test, folder)