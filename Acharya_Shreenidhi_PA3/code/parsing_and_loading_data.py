import json
import pprint as pp
from pymongo import MongoClient

# Create the database
def create_database(mongo_uri):
    '''
    Desc: Create the database in mongodb
    '''
    client = MongoClient(mongo_uri)
    database = client['DATA']
    database.create_collection('API_DATA')
    database.create_collection('MASHUP_DATA')

def load_data_mongo(data, uri, db_name, db_collection):
    '''
    Desc: load data to MongoDB
    input: data dictionary, url for the mongodb, database name, collection name
    '''
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[db_collection]

    # Inserting data into MongoDB collection
    collection.insert_many(data)

    # Closing MongoDB connection
    client.close()


def parse_and_load_api(api_file_path, uri, db_name, db_collection):
    '''
    Desc: Parse the API data file
    input: file path, url for the mongodb, database name, collection name
    '''
    api_data_list = []
    with open(api_file_path, "r", encoding="ISO-8859-1") as file:

        for current_line in file:
            field = current_line.split("$#$")
            current_api_data = {
                "id": field[0],
                "title": field[1],
                "summary": field[2],
                "rating": float(field[3]) if field[3] else 0.0,
                "name": field[4],
                "label": field[5],
                "author": field[6],
                "description": field[7],
                "type": int(field[8]),
                "downloads": field[9],
                "useCount": int(field[10]) if field[10] else 0.0,
                "sampleUrl": field[11],
                "downloadUrl": field[12],
                "dateModified": field[13],
                "remoteFeed": field[14],
                "numComments": int(field[15]) if field[15] else 0.0,
                "commentsUrl": field[16],
                "Tags": field[17].split("###"),
                "category": field[18],
                "protocols": field[19],
                "serviceEndpoint": field[20],
                "version": field[21],
                "wsdl": field[22],
                "data":  field[23],
                "apigroups": field[24],
                "example": field[25],
                "clientInstall": field[26],
                "authentication": field[27],
                "ssl": field[28],
                "readonly": field[29],
                "VendorApiKits": field[30],
                "CommunityApiKits": field[31],
                "blog": field[32],
                "forum": field[33],
                "support": field[34],
                "accountReq": field[35],
                "commercial": field[36],
                "provider": field[37],
                "managedBy": field[38],
                "nonCommercial": field[39],
                "dataLicensing": field[40],
                "fees": field[41],
                "limits": field[42],
                "terms": field[43],
                "company": field[44],
                "updated": field[45],

            }
            api_data_list.append(current_api_data)

    # load data
    load_data_mongo(api_data_list, uri, db_name, db_collection)


def parse_and_load_mashup(mashup_file_path, uri, db_name, db_collection):
    '''
    Desc: Parse the Mashup data file
    input: file path, url for the mongodb, database name, collection name
    '''
    mashup_data_list = []
    API_list = {}
    API_len_list = {}
    with open(mashup_file_path, "r", encoding="ISO-8859-1") as file:

        for current_line in file:
            field = current_line.split("$#$")
            api_used_list = []

            # check if the field 16 exists
            if field[16]:
                api_list = field[16].split("###")
                API_len_list[field[4]] = len(api_list)
                for api in field[16].split("###"):
                    api_split = api.split("$$$")
                    api_used_list.append(api_split[0])
                    API_list[api_split[0]] = API_list.get(api_split[0], 0) + 1

            current_api_data = {
                "id": field[0],
                "title": field[1],
                "summary": field[2],
                "rating": float(field[3]) if field[3] else 0.0,
                "name": field[4],
                "label": field[5],
                "author": field[6],
                "description": field[7],
                "type": field[8],
                "downloads": field[9],
                "useCount": int(field[10]) if field[10] else 0.0,
                "sampleUrl": field[11],
                "dateModified": field[12],
                "numComments": int(field[13]) if field[13] else 0.0,
                "commentsUrl": field[14],
                "Tags": field[15].split("###"),
                "APIs": api_used_list,
                "updated": field[17],

            }
            mashup_data_list.append(current_api_data)

    # load data
    load_data_mongo(mashup_data_list, uri, db_name, db_collection)

    # sorting the dictionary based on number of API for every record
    api_dictionary = dict(
        sorted(API_list.items(), key=lambda x: x[1], reverse=True))
    
    # sorting the dictionary based most frequently used API
    API_len_list_sorted = dict(
        sorted(API_len_list.items(), key=lambda x: x[1], reverse=True))

    # creating a file containing count of each API after the mashup data is loaded.
    with open("data_api.json", "w") as json_file:
        json.dump(api_dictionary, json_file)

    # creating a file for storing the data about number of API for each record
    with open("data_api_len.json", "w") as json_file:
        json.dump(API_len_list_sorted, json_file)


if __name__ == "__main__":
    # create the database
    mongo_uri = "mongodb+srv://root1234:root1234@cluster0.c0ggoiz.mongodb.net/"
    create_database(mongo_uri)

    # set database details
    database_name = "DATA"
    collection_name_api = "API_DATA"
    collection_name_mashup = "MASHUP_DATA"

    # loading the data from the given files
    api_file_path = "/Users/shreenidhi/Desktop/webservices/data/api.txt"
    mashup_file_path = "/Users/shreenidhi/Desktop/webservices/data/mashup.txt"

    # load API data
    api_data = parse_and_load_api(
        api_file_path, mongo_uri, database_name, collection_name_api)
    
    # load Mashup data
    mashup_data = parse_and_load_mashup(
        mashup_file_path, mongo_uri, database_name, collection_name_mashup)
