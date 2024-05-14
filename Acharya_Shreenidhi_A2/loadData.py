import pandas as pd
from pymongo import MongoClient

def csv_to_mongodb(csv_path, uri, db_name, collection_name):
    # Read CSV into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Convert DataFrame to a list of dictionaries (each row becomes a dictionary)
    data = df.to_dict(orient='records')

    # Connect to MongoDB
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]

    # Insert data into MongoDB collection
    collection.insert_many(data)

    # Close MongoDB connection
    client.close()


if __name__ == "__main__":
    #mongodb connection details
    mongo_uri = "mongodb+srv://root1234:root1234@cluster0.c0ggoiz.mongodb.net/LaptopDetails"
    database_name = "LaptopDetails"
    collection_name = "laptops"
    csv_to_mongodb("data - Sheet1.csv",
                   mongo_uri, database_name, collection_name)