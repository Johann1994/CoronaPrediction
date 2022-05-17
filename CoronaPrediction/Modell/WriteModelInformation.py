import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation
from sklearn.preprocessing import StandardScaler
import joblib

def writeModelInformationToFile(model, outputfile):
    outputfile.write("Model:\r\n")
    outputfile.write(str(model.output))
    outputfile.write("\r\nModel Summary: \r\n")
    model.summary(print_fn=lambda x: outputfile.write(x + '\n'))

def plotHistorianTrend(history, fileToSave):
    mse = history.history['mse']
    val_mse = history.history['val_mse']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(mse) + 1)

    fig, ax = plt.subplots(2, 1)
    fig.subplots_adjust(bottom=0.1)
    ax[0].plot(epochs, mse, 'bo', label='Training')
    ax[0].plot(epochs, val_mse, 'b', label = 'Validierung')
    ax[1].plot(epochs, loss, 'bo', label='Training')
    ax[1].plot(epochs, val_loss, 'b', label = 'Validierung')
    ax[0].set_title("MSE Training/Validation")
    ax[1].set_title("LOSS Training/Validation")
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(fileToSave + "TrainingsHistory.png")


def plotPredictedvsReal(yReal, yPred, fileToSave, folderScaler):
    fig, ax = plt.subplots(1,1)
    scalerY = joblib.load(folderScaler + "myScalerY.pkl")
    ax.plot(range(0, yPred.size), yPred*scalerY, 'b', label='predicted')
    ax.plot(range(0, yReal.size), yReal*scalerY, 'bo', label='validation')
    ax.legend()
    fig.savefig(fileToSave)    


def plotHistorianTrendWithoutValidation(history, fileToSave):
    mse = history.history['mean_squared_error']
    loss = history.history['loss']
    epochs = range(1, len(mse) + 1)

    fig, ax = plt.subplots(2, 1)
    fig.subplots_adjust(bottom=0.1)
    ax[0].plot(epochs, mse, 'bo', label='Training')
    ax[1].plot(epochs, loss, 'bo', label='Training')
    ax[0].set_title("MSE Training")
    ax[1].set_title("LOSS Training")
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(fileToSave + "TrainingsHistory.png")
