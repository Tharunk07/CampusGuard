import requests
import os
from dotenv import load_dotenv
import geocoder

load_dotenv()

g = geocoder.ip('103.130.89.67')

def send_msg(img_path, class_name):
   token = os.getenv("TELEGRAM_TOKEN")
   text = f"Warning! {class_name} Detected. Location: {listToString(g.latlng)}"
   ids = ["2087668156", "1800886125", "1214661913", "1346063520"]

   for id in ids:
       try:
           with open(img_path, "rb") as image_file:
               files = {"photo": image_file}
               data = {"chat_id": id, "caption": text}
               response = requests.post(
                   f"https://api.telegram.org/bot{token}/sendPhoto", files=files, data=data
               )
               print(f"Alert sent successfully!", response)
       except Exception as e:
           print(f"Error sending message to {id}: {e}")


    

def listToString(s): 
    float_string = ""
    for num in s:
        float_string = float_string + str(num) + " "

    return float_string

if __name__ == "__main__":
    send_msg()