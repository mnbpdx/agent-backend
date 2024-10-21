import requests
from dataclasses import dataclass
from business_logic import API_HEADER

#Define the forecast data class
@dataclass
class Forecast:
    day: str
    temperature: int
    chance_rain: int
    icon: str
    detailed_forecast: str

def get_forecast(forecast_url: str):
    print(forecast_url)
    response = requests.get(forecast_url)
    data = response.json()
    if response.status_code == 200:
        data = response.json()
        forecasts = [
            Forecast(
                day = period['name'],
                temperature = period['temperature'],
                chance_rain = period['probabilityOfPrecipitation']['value'],
                icon = period['icon'],
                detailed_forecast = period['detailedForecast'],
            )
            for period in data['properties']['periods']
        ]

        # for f in forecasts:
        #     print(f.detailed_forecast)

        return forecasts
            # print(f'{f.day}: {f.temperature} degrees')
            # if f.chance_rain != None: print(f'{f.chance_rain}% chance of rain') 
            # else: print("No rain!")
            # print(f.icon)

    #     stuff = json.dumps(data, indent=4)
    #     print(stuff)
    else: print("Error.")