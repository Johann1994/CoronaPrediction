import pandas as pd
import numpy as np
from CSVDataReader import readInCSVFile

def cleanDataFramForAustria(df):
    indexAustria = 10
    df = df[df.BundeslandID == indexAustria]
    df['Time'] = pd.to_datetime(df.Time, format="%d.%m.%Y %H:%M:%S")
    df = df.drop(columns=['Bundesland', 'AnzahlTotTaeglich', 'AnzahlTotSum'])
    df = df[df.Time > pd.to_datetime("01.07.2020", format="%d.%m.%Y")]
    df.SiebenTageInzidenzFaelle = pd.to_numeric(df.SiebenTageInzidenzFaelle.str.replace(',','.'))
    df["AnzahlPlus10d"] = df.AnzahlFaelle.shift(periods=10)
    df.set_index(df['Time'], inplace=True)
    return df

def cleanDataFramWithoutAustria(df):
    indexAustria = 10
    df = df[df.BundeslandID != indexAustria]
    df['Time'] = pd.to_datetime(df.Time, format="%d.%m.%Y %H:%M:%S")
    df = df.drop(columns=['Bundesland', 'AnzahlTotTaeglich', 'AnzahlTotSum'])
    df = df[df.Time > pd.to_datetime("01.07.2020", format="%d.%m.%Y")]
    df.SiebenTageInzidenzFaelle = pd.to_numeric(df.SiebenTageInzidenzFaelle.str.replace(',','.'))
    df["AnzahlPlus10d"] = df.AnzahlFaelle.shift(periods=10)
    df.set_index(df['Time'], inplace=True)
    return df

def cleanDataFramWithoutAustriaForPrediction(df):
    indexAustria = 10
    df = df[df.BundeslandID != indexAustria]
    df['Time'] = pd.to_datetime(df.Time, format="%d.%m.%Y %H:%M:%S")
    df = df.drop(columns=['Bundesland', 'AnzahlTotTaeglich', 'AnzahlTotSum'])
    df = df[df.Time > pd.to_datetime("01.02.2022", format="%d.%m.%Y")]
    df.SiebenTageInzidenzFaelle = pd.to_numeric(df.SiebenTageInzidenzFaelle.str.replace(',','.'))
    df["AnzahlPlus10d"] = df.AnzahlFaelle.shift(periods=10)
    df.set_index(df['Time'], inplace=True)
    return df

def prepareDataframeJohnHopkinsUniversity(df):
    df["Province/State"] = df["Province/State"].fillna("")
    df["State_Country"] =  df["Country/Region"] + "_" + df["Province/State"]

    df = df.drop("Province/State", axis=1)
    df = df.drop("Country/Region", axis=1)
    df = df.drop("Lat", axis=1)
    df = df.drop("Long", axis=1)
    
    countryList = []
    for country in df["State_Country"].unique():
        pandasSeries = df.loc[df.State_Country == country, :]
        emptyDF = pd.DataFrame()
        emptyDF["ConfirmedCumulation"] = pandasSeries.loc[:,df.columns != "State_Country"].squeeze()
        emptyDF["Confirmed"] = emptyDF["ConfirmedCumulation"] - emptyDF["ConfirmedCumulation"].shift(1)
        emptyDF["Time"] = pd.to_datetime((pandasSeries.loc[:,df.columns != "State_Country"]).columns)
        emptyDF = emptyDF[emptyDF.Time > pd.to_datetime("01.07.2020", format="%d.%m.%Y")]
        emptyDF = emptyDF.loc[pd.notnull(emptyDF["Confirmed"]), :]
        emptyDF["Prediction10"] = emptyDF["Confirmed"].shift(-10)
        emptyDF = emptyDF.loc[pd.notnull(emptyDF["Prediction10"]), :]
        countryList.append(emptyDF)   
    
    return countryList

def addWeatherInformation(df):
    indexAustria = 10
    df = df[df.BundeslandID != indexAustria]

    dfNew = readinCSVAndMergeIt("../Data/weatherEisenstadt.csv", df[df.BundeslandID == 1])
    dfNew = dfNew.append(readinCSVAndMergeIt("../Data/weatherKlagenfurt.csv", df[df.BundeslandID == 2]))
    dfNew = dfNew.append(readinCSVAndMergeIt("../Data/weatherStPoelten.csv", df[df.BundeslandID == 3]))
    dfNew = dfNew.append(readinCSVAndMergeIt("../Data/weatherLinz.csv", df[df.BundeslandID == 4]))
    dfNew = dfNew.append(readinCSVAndMergeIt("../Data/weatherSalzburg.csv", df[df.BundeslandID == 5]))
    dfNew = dfNew.append(readinCSVAndMergeIt("../Data/weatherGraz.csv", df[df.BundeslandID == 6]))
    dfNew = dfNew.append(readinCSVAndMergeIt("../Data/weatherInnsbruck.csv", df[df.BundeslandID == 7]))
    dfNew = dfNew.append(readinCSVAndMergeIt("../Data/weatherBregenz.csv", df[df.BundeslandID == 8]))
    dfNew = dfNew.append(readinCSVAndMergeIt("../Data/weatherVienna.csv", df[df.BundeslandID == 9]))

    for column in dfNew.columns:
        dfNew = dfNew.loc[pd.notnull(dfNew[column])]
    return dfNew

def addWeatherInformation_Val(df):
    indexAustria = 10
    df = df[df.BundeslandID != indexAustria]

    dfNew = readinCSVAndMergeIt("../Data/weatherEisenstadt_Val.csv", df[df.BundeslandID == 1])
    dfNew = dfNew.append(readinCSVAndMergeIt("../Data/weatherKlagenfurt_Val.csv", df[df.BundeslandID == 2]))
    dfNew = dfNew.append(readinCSVAndMergeIt("../Data/weatherStPoelten_Val.csv", df[df.BundeslandID == 3]))
    dfNew = dfNew.append(readinCSVAndMergeIt("../Data/weatherLinz_Val.csv", df[df.BundeslandID == 4]))
    dfNew = dfNew.append(readinCSVAndMergeIt("../Data/weatherSalzburg_Val.csv", df[df.BundeslandID == 5]))
    dfNew = dfNew.append(readinCSVAndMergeIt("../Data/weatherGraz_Val.csv", df[df.BundeslandID == 6]))
    dfNew = dfNew.append(readinCSVAndMergeIt("../Data/weatherInnsbruck_Val.csv", df[df.BundeslandID == 7]))
    dfNew = dfNew.append(readinCSVAndMergeIt("../Data/weatherBregenz_Val.csv", df[df.BundeslandID == 8]))
    dfNew = dfNew.append(readinCSVAndMergeIt("../Data/weatherVienna_Val.csv", df[df.BundeslandID == 9]))

    for column in dfNew.columns:
        dfNew = dfNew.loc[pd.notnull(dfNew[column])]
    return dfNew

def addWeatherInformationPrediction(df):
    indexAustria = 10
    df = df[df.BundeslandID != indexAustria]

    dfNew = readinCSVAndMergeIt("../Data/weatherGraz_Prediction.csv", df[df.BundeslandID == 6])

    for column in dfNew.columns:
        dfNew = dfNew.loc[pd.notnull(dfNew[column])]
    return dfNew

def readinCSVAndMergeIt(csvFileName, df):
    weather = readInCSVFile(csvFileName, ',', "cp1252")
    weather["Date time"] = pd.to_datetime(weather["Date time"], format="%m/%d/%Y")
    weather = weather[weather["Date time"] > pd.to_datetime("01.07.2020", format="%d.%m.%Y")]
    weather.set_index(weather["Date time"], inplace=True)
    weather = weather[["Temperature","Relative Humidity","Precipitation", "Visibility", "Sea Level Pressure", "Wind Speed"] ]
    weather = weather[~weather.index.duplicated(keep='first')]

    df = pd.merge(df, weather, how='inner', left_index=True, right_index=True)
    #df.reset_index(drop=True, inplace=True)
    return df

def addLockdownInformationForAustria(df):
    df["LockDown"] = 0
    mask = (df.Time > pd.to_datetime("16.03.2020", format="%d.%m.%Y")) & (df.Time < pd.to_datetime("01.05.2020", format="%d.%m.%Y"))
    df[mask]["LockDown"] = 1
    mask = (df.Time > pd.to_datetime("17.11.2020", format="%d.%m.%Y")) & (df.Time < pd.to_datetime("07.12.2020", format="%d.%m.%Y"))
    df[mask]["LockDown"] = 1
    mask = (df.Time > pd.to_datetime("26.12.2020", format="%d.%m.%Y")) & (df.Time < pd.to_datetime("08.02.2021", format="%d.%m.%Y"))
    df[mask]["LockDown"] = 1
    mask = (df.Time > pd.to_datetime("22.11.2021", format="%d.%m.%Y")) & (df.Time < pd.to_datetime("12.12.2021", format="%d.%m.%Y"))
    df[mask]["LockDown"] = 1
    return df;

def addWeekDays(df):
    df["weekDay"] = df['Time'].dt.dayofweek
    return df

def addVaccination(df):
    dfVac = readInCSVFile("../Data/COVID19_vaccination_doses_timeline.csv",";" )
    indexAustria = 10
    dfVac.date = dfVac.date.str[0:10]
    dfVac['VacTime'] = pd.to_datetime(dfVac.date, format='%Y-%m-%d' )
    dfVac = dfVac[dfVac.state_id != indexAustria]
    dfVac = dfVac[dfVac.state_id != 0]
    dfVac = dfVac.groupby(['VacTime','state_id', 'dose_number'])['doses_administered_cumulative'].sum()
    dfVac = dfVac.reset_index()
    dfVacPrep = dfVac.groupby(['VacTime', 'state_id'])['dose_number'].sum()
    dfVacPrep = dfVacPrep.reset_index()
    dfVacPrep = dfVacPrep.drop('dose_number', axis=1)
    dfVacDos = dfVac[dfVac.dose_number == 1]
    dfVacDos.reset_index(inplace=True)
    dfVacPrep['VacDose1'] = dfVacDos['doses_administered_cumulative']
    dfVacDos = dfVac[dfVac.dose_number == 2]
    dfVacDos.reset_index(inplace=True)
    dfVacPrep['VacDose2'] = dfVacDos['doses_administered_cumulative']
    dfVacDos = dfVac[dfVac.dose_number == 3]
    dfVacDos.reset_index(inplace=True)
    dfVacPrep['VacDose3'] = dfVacDos['doses_administered_cumulative']


    dfReturn = pd.DataFrame()

    for index in dfVacPrep.state_id.unique():
        dfInsert = dfVacPrep[dfVacPrep.state_id == index]
        dfInsert.set_index(dfInsert.VacTime, inplace=True)
        if dfReturn.empty:
            dfReturn = mergeVaccinationWithDF(df[df.BundeslandID == index], dfInsert)
        else:
            dfReturn = dfReturn.append(mergeVaccinationWithDF(df[df.BundeslandID == index], dfInsert))
    
    return dfReturn

def mergeVaccinationWithDF(df, dfVac):
    df = pd.merge(df, dfVac, how='inner', left_index=True, right_index=True)
    df.reset_index(drop=True, inplace=True)
    return df

