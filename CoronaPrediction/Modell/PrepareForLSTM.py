import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import  MinMaxScaler
import joblib

def prepareDatFramesForLSTMUniqueLands(df, folder, timeSteps=10, loadFromFile=False):
    df = df.drop("Time", axis=1).copy()
    BundeslandIDs = df.BundeslandID.unique()
    y = df.AnzahlPlus10d
    x = df.drop("AnzahlPlus10d", axis=1).copy()
    x = x.drop("BundeslandID", axis=1).copy()
    if loadFromFile:
        scalerX = joblib.load(folder + "myScalerX.pkl")
        scalerY = joblib.load(folder + "myScalerY.pkl")
    else:
        
        scalerX = MinMaxScaler(feature_range=(0,0.5))
        scalerX.fit(x)
        joblib.dump(scalerX, folder + "myScalerX.pkl")
        scalerY = y.max()*2
        joblib.dump(scalerY, folder + "myScalerY.pkl")
       

    x = scalerX.transform(x)

    xArray = np.empty([x.shape[0]-timeSteps, timeSteps, x.shape[1]])
    yArray = np.empty([x.shape[0]- timeSteps, 1])
    nextIndex = 0

    for ID in BundeslandIDs:
        selection = df.BundeslandID == ID
        xSelected = x[selection,]
        ySelected = y.loc[selection,]
        
        for startIndex in range(timeSteps, xSelected.shape[0]):
            xInsert = np.empty([timeSteps, x.shape[1]])
            yArray[startIndex- timeSteps + nextIndex] = ySelected.iloc[startIndex,]
            for i in range(0, timeSteps):
                xInsert[i] = xSelected[startIndex - i, ]
            xArray[startIndex- timeSteps + nextIndex] = xInsert
        nextIndex = nextIndex + xSelected.shape[0] -1

    yArray = yArray / scalerY
    return xArray, yArray

def prepareDatFrames(df, folder, timeSteps=60, loadFromFile=False):
    df = df.drop("Time", axis=1).copy()
    y = df.Prediction10
    x = df.drop("Prediction10", axis=1)

    if loadFromFile:
        scalerX = joblib.load(folder + "myScalerX.pkl")
    else:
        scalerX = MinMaxScaler()
        scalerX.fit(x)
        joblib.dump(scalerX, folder + "myScalerX.pkl")
       

    x = scalerX.transform(x)
    y =  y / y.max()

    xArray = np.empty([x.shape[0]-timeSteps, timeSteps, x.shape[1]])
    yArray = np.empty([x.shape[0]- timeSteps, 1])
    nextIndex = 0
                
    for startIndex in range(timeSteps, x.shape[0]):
         xInsert = np.empty([timeSteps, x.shape[1]])
         yArray[startIndex- timeSteps + nextIndex] = y.iloc[startIndex,]
         for i in range(0, timeSteps):
             xInsert[i] = x[startIndex - i, ]
         xArray[startIndex- timeSteps + nextIndex] = xInsert

    return xArray, yArray


def prepareDatFramesForLSTMUniqueLandsYMoreElements(df, folder, timeSteps=10, predictedValues=7, loadFromFile=False):
    df = df.drop("Time", axis=1).copy()
    BundeslandIDs = df.BundeslandID.unique()
    y = df.AnzahlFaelle
    x = df.drop("AnzahlPlus10d", axis=1).copy()
    x = x.drop("BundeslandID", axis=1).copy()
    if loadFromFile:
        scalerX = joblib.load(folder + "myScalerX.pkl")
        scalerY = joblib.load(folder + "myScalerY.pkl")
    else:
        
        scalerX = MinMaxScaler(feature_range=(0,0.5))
        scalerX.fit(x)
        joblib.dump(scalerX, folder + "myScalerX.pkl")
        scalerY = y.max()*2
        joblib.dump(scalerY, folder + "myScalerY.pkl")
       

    x = scalerX.transform(x)

    xArray = np.empty([x.shape[0]-timeSteps - predictedValues, timeSteps, x.shape[1]])
    yArray = np.empty([x.shape[0]- timeSteps - predictedValues, predictedValues])
    nextIndex = 0

    for ID in BundeslandIDs:
        selection = df.BundeslandID == ID
        xSelected = x[selection,]
        ySelected = y.loc[selection,]
        
        for startIndex in range(timeSteps, xSelected.shape[0] - predictedValues - 1):
            xInsert = np.empty([timeSteps, x.shape[1]])
#            yArray[startIndex- timeSteps + nextIndex] = ySelected.iloc[startIndex,]
            yInsert = np.empty([predictedValues])
            for i in range(0, timeSteps):
                xInsert[i] = xSelected[startIndex - i, ]
            for i in range(1, predictedValues +1):
                yInsert[i-1] = ySelected[startIndex + i]
            xArray[startIndex- timeSteps + nextIndex] = xInsert
            yArray[startIndex- timeSteps + nextIndex] = yInsert
        nextIndex = nextIndex + xSelected.shape[0] -1

    yArray = yArray / scalerY
    return xArray, yArray
