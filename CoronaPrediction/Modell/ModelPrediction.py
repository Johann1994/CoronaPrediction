from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Activation, Dropout, Input, LeakyReLU
from sklearn.metrics import r2_score
from WriteModelInformation import writeModelInformationToFile, plotHistorianTrend, plotPredictedvsReal

def makePrediction(modelPath, inputValuesForPrediction):

    model = load_model(modelPath)

    y_pred = model.predict(inputValuesForPrediction)
    y_pred = y_pred.reshape(y_pred.shape[0])

    return y_pred


def makePrediction7d(modelPath, inputValuesForPrediction):

    model = load_model(modelPath)

    y_pred = model.predict(inputValuesForPrediction)

    return y_pred