from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, Dropout, Input, LeakyReLU
from sklearn.metrics import r2_score
from WriteModelInformation import writeModelInformationToFile, plotHistorianTrend, plotPredictedvsReal

def trainModel(xTrain, yTrain, xValidate, yValidate, xTest, yTest, folder=""):
    print(xTrain.shape)
    print(yTrain.shape)
    print(yTrain.max())

    if yTrain.max() > 1.0:
        exit()
    if yValidate.max() > 1.0:
        exit()
    if yTest.max() > 1.0:
        exit()

    n_features = xTrain.shape[2]
    timeSteps = xTrain.shape[1]
    predictedValues = yTrain.shape[1]

    leakyRelu = LeakyReLU(alpha=0.1)
#   Prediction with same Date in future sigmoid is not good
    model = Sequential()
    model.add(Input(shape=( timeSteps, n_features)))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(256) )
    model.add(LeakyReLU(alpha=0.1))
    model.add(LSTM(128,  return_sequences=False, activation='tanh') )
    model.add(Dropout(0.5))
    model.add(Dense(predictedValues, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mse', metrics=['mse'] )

    if folder != "":
        outputfile = open(folder + "ModelInfo.txt",'w')
        writeModelInformationToFile(model, outputfile)

    history = model.fit(xTrain, yTrain, epochs=50, validation_data=(xValidate, yValidate), batch_size = 2, shuffle=False, verbose=1)

    if folder != "":
        model.save(folder + "model.h5")

    if folder != "":
        plotHistorianTrend(history, folder)

    yPred = model.predict(xTest)

    mse = model.evaluate(xTest, yTest)
    if folder != "":
        outputfile.write("mse: "+ str(mse[0]))
        outputfile.close()

#    plotPredictedvsReal(yTest[0:100,], yPred[0:100,], folder + "PredictVsReal.png", folder)
#    plotPredictedvsReal(yTest[-100:-1,], yPred[-100:-1,], folder + "PredictVsReal2.png", folder)
#    plotPredictedvsReal(yTest, yPred, folder + "PredictVsRealAllData.png", folder)

    return mse[0]