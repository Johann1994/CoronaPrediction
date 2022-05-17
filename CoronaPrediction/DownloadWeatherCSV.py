import http.client

def getHistoryWeather(startDate, endDate, writeHeader=False):
    conn = http.client.HTTPSConnection("visual-crossing-weather.p.rapidapi.com")

    headers = {
        'x-rapidapi-host': "visual-crossing-weather.p.rapidapi.com",
        'x-rapidapi-key': "77a5f2e646msha26003a778835cep1ab1c8jsn16747f03cd7b"
    }

    conn.request("GET", f"/history?startDateTime={startDate}T12%3A00%3A00&aggregateHours=24&location=Bregenz%2CAustria&endDateTime={endDate}T12%3A00%3A00&unitGroup=metric&dayStartTime=8%3A00%3A00&contentType=csv&dayEndTime=17%3A00%3A00&shortColumnNames=0", headers=headers)

    res = conn.getresponse()
    data = res.read()

    #print(data.decode("utf-8"))
    f = open("../Data/weatherBregenz_Val.csv", "a")
    if not writeHeader:
        f.write(data.decode("utf-8").split("\n",1)[1])
    else:
        f.write(data.decode("utf-8"))
    f.close()

getHistoryWeather("2022-02-01", "2022-02-28", False)
#getHistoryWeather("2021-12-21", "2022-01-19", False)
#getHistoryWeather("2022-01-20", "2022-01-31", False)