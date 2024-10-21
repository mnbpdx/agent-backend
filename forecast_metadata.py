import requests
import json
from dataclasses import dataclass
from business_logic import API_HEADER
from geolocation import PersonalInfo

#Define the forecast data class
@dataclass
class ForecastMetadata:
    forecast_url: str
    # forecast_hourly: str

def get_forecast_metadata(latitude: str, longitude: str):
    url = f'https://api.weather.gov/points/{latitude},{longitude}'
    response = requests.get(url, headers=API_HEADER)
    if response.status_code == 200:
        data = response.json()
        ## Un-comment to see full json in output:
        #forecast_metadata = json.dumps(data, indent=4)
        forecast_metadata = ForecastMetadata(
            forecast_url = data['properties']['forecast']
        )
        return forecast_metadata
    
    else: print(f"Error: Status code {response.statuscode} (not 200).")