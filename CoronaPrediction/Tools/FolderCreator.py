import os

def createDirectory(filepath):
    if(os.path.exists(filepath)):
        return
    os.makedirs(filepath)


