import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "./Tools")
from FolderCreator import createDirectory

def printInformationToFile(df, folderName):
    createDirectory(folderName)
    printDataFrameInformationToFile(df, f"{folderName}Info.txt")
    plotDataFrameToTime(df.AnzahlFaelle, df.Time, "Fälle", f"{folderName}Faelle.png")
    plotDataFrameToTime(df.SiebenTageInzidenzFaelle, df.Time,"7 Tage Inzidenz", f"{folderName}7Inzidenz.png")
    plotScatterPlot(df.AnzahlFaelle, df.SiebenTageInzidenzFaelle, "Fälle", "7 Tage Inzidenz", f"{folderName}Faellevs7Tage.png" )
    plotScatterPlot(df.AnzahlFaelle, df.AnzahlGeheiltTaeglich, "Fälle", "Geheilg", f"{folderName}FaellevsGeheilt.png" )
    

def printDataFrameInformationToFile(df, fileName):
    outputfile = open(fileName, 'w')
    outputfile.write("shape: \r\n")
    outputfile.write(str(df.shape))
    outputfile.write("\r\nused columns: \r\n")
    outputfile.write(str(df.columns))
    outputfile.write("\r\nBeschreibung: \r\n")
    outputfile.write(str(df.describe().to_string(index = False)))
    outputfile.close()

def plotDataFrameToTime(y,Time, yLabel, fileName = ""):
    fig, axis = plt.subplots(1,1, figsize=(8,4))
    axis.plot(Time,y,'b')
    axis.set_ylabel(yLabel)
    axis.set_xlabel("Time")
    ymin = y.min() * 0.9
    if ymin < 0:
        ymin = 0
    axis.set_ylim([ymin, y.max() * 1.1])

    if fileName == "":
        plt.show()
    else:
        plt.savefig(fileName)

def plotScatterPlot(x,y,xLabel, yLabel, fileName = ""):
    fig, axis = plt.subplots(1,1, figsize=(8,4))
    axis.scatter(x,y)
    axis.set_ylabel(yLabel)
    axis.set_xlabel(xLabel)
    ymin = y.min() * 0.9
    if ymin < 0:
        ymin = 0
    xmin = x.min() * 0.9
    if xmin < 0:
        xmin = 0
    axis.set_ylim([ymin, y.max() * 1.1])
    axis.set_xlim([xmin, x.max() * 1.1])
    if fileName == "":
        plt.show()
    else:
        plt.savefig(fileName)