from geopy.geocoders import Nominatim
from dataclasses import dataclass

@dataclass
class PersonalInfo:
    latitude: str
    longitude: str

def get_user_info(address):
    if address == "": 
        address = input("Input your street address, city, state: ") # TODO: Error handling unclear, but city/state acceptable as well
    geolocator = Nominatim(user_agent="weather_app")
    location = geolocator.geocode(f"{address}")
    return PersonalInfo(
        latitude = location.latitude,
        longitude = location.longitude,
    )