import requests
import pandas as pd
from datetime import datetime

def fetch_iot_data(cities=["Delhi", "Mumbai", "Bangalore"]):
    api_key = "f22f383352f7c43e197ef99edf3a290d"  # Replace
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    data = []
    for city in cities:
        params = {"q": city, "appid": api_key, "units": "metric"}
        response = requests.get(base_url, params=params)
        weather_data = response.json()
        data.append({
            "city": city,
            "temperature_c": weather_data["main"]["temp"],
            "humidity_percent": weather_data["main"]["humidity"],
            "pressure_mb": weather_data["main"]["pressure"],
            "wind_speed_mps": weather_data["wind"]["speed"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    df = pd.DataFrame(data)
    df.to_csv("data/iot_data.csv", index=False)
    print("IoT data saved to data/iot_data.csv")
    return df

if __name__ == "__main__":
    fetch_iot_data()