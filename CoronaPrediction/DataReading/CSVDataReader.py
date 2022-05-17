import numpy as np
import pandas as pd
import math

def readInCSVFile(fileName, sep = ";",encod=None):
    df = pd.read_csv(fileName, sep=sep, encoding=encod)
    return df