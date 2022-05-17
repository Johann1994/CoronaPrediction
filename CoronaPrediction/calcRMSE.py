import joblib
import math

scaler = joblib.load("D:/FH/5.Semester/Bachelorarbeit/Projekt/CoronaPrediction/Output/WithWeatherAndLockdown_LastDays/1/myScalerY.pkl")

value = math.sqrt(0.0012641413137316704) * scaler 
print(value)