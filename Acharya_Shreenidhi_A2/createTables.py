from pymongo import MongoClient

def create_database():
    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    CONNECTION_STRING = "mongodb+srv://root1234:root1234@cluster0.c0ggoiz.mongodb.net/LaptopDetails"

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    client = MongoClient(CONNECTION_STRING)

    # Create the database for our example (we will use the same database throughout the tutorial)
    # Use the same database for both collections
    database = client['LaptopDetails']

    # Create the userDetails collection within the same database
    userDetails_collection = database['userDetails']
    userCommunication_collection = database['userCommunication']

    # Define the schema for userDetails collection
    userDetails_schema = {
        "username": str,  # string data type for username
        "password": str   # string data type for password
    }

    database.create_collection('userDetails')

    userCommunication_schema = {
        "username": str,  # string data type for username
        "task_id": int   # string data type for password

    }
    database.create_collection('userCommunication')

    # Inserting data into userDetails collection
    user_data = {
        "username": "dummy",
        "password": "dummy123"
    }
    userDetails_collection.insert_one(user_data)

    # Inserting data into userCommunication collection
    communication_data = {
        "username": "dummy",
        "task_id": 1
    }
    userCommunication_collection.insert_one(communication_data)

    # Create indexes if needed
    # userDetails_collection.create_index([("username", 1)], unique=True)


# This is added so that many files can reuse the function get_database()
if __name__ == "__main__":
    # Get the userDetails collection within the same database
    create_database()
