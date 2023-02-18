from pymongo import MongoClient
from bson import ObjectId

client = MongoClient("mongodb://localhost:27017")
db = client.seedlings
image_details = db.imageData


def addNewImage(i_name, Type, time, url):
    image_details.insert({
        "file_name": i_name,
        "prediction": Type,
        "upload_time": time,
        "url": url
    })


def cropRecommendation(Type, time):
    image_details.insert({
        "prediction": Type,
        "upload_time": time
    })


def pollutionReport(image, time, location):
    image_details.insert({
        "prediction": image,
        "upload_time": time,
        "upload_time": location
    })