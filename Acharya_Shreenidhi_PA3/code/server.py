import json
from flask import Flask, request, render_template
from waitress import serve
from pymongo import MongoClient

app = Flask(__name__)

# database details
mongo_uri= "mongodb+srv://root1234:root1234@cluster0.c0ggoiz.mongodb.net/"
database_name = "DATA"
collection_name = "API_DATA"
client = MongoClient(mongo_uri)

# Create the database for our example, Use the same database for both collections
database = client['DATA']

# Create the collection within the same database
api_collection = database['API_DATA']
mashup_collection = database['MASHUP_DATA']

# helper functions
def query1(update_year, protocol, rating_type, rating, tags, category):
    '''
    Desc: Performs query 1
    input: year, protocol type, rating_type(low,high,equal), rating, tags, category
    return: api names
    '''
    update_year = '^'+update_year
    if rating_type == "lower":
        query = {"updated": {"$regex": update_year}, "category": category, "protocols": protocol, "rating": {"$lt": rating},
                 "Tags": {"$in": tags}
                 }
    elif rating_type == "higher":
        query = {"updated": {"$regex": update_year}, "category": category, "protocols": protocol, "rating": {"$gt": rating},
                 "Tags": {"$in": tags}
                 }
    else:
        query = {"updated": {"$regex": update_year}, "category": category, "protocols": protocol, "rating": rating,
                 "Tags": {"$in": tags}
                 }
    api_collection = database['API_DATA']
    result = api_collection.find(query)
    api_names = [record['name'] for record in result]
    return api_names

def query2(update_year, apis, tags):
    '''
    Desc: Performs query 2
    input: year, api, tags
    return: api names
    '''
    update_year = '^'+update_year
    query = {"updated": {"$regex": update_year},  "APIs": {"$in":apis},
             "Tags": {"$in": tags}
             }

    mashup_collection = database['MASHUP_DATA']
    result = mashup_collection.find(query)
    api_names = [record['name'] for record in result]
    return api_names

def query3(keywords):
    '''
    Desc: Performs query 3
    input: keywords
    return: api names
    '''
    api_collection = database['API_DATA']
    query = {"$and": []}

    for keyword in range(len(keywords)):
        query["$and"].append({"$or": [
            {"summary": {"$regex": f".*{keywords[keyword]}.*", "$options": "i"}},
            {"title": {"$regex": f".*{keywords[keyword]}.*", "$options": "i"}},
            {"description": {"$regex": f".*{keywords[keyword]}.*", "$options": "i"}}
        ]})

    result = api_collection.find(query)

    api_names = [record["name"] for record in result]
    return api_names


def query4(keywords):
    '''
    Desc: Performs query 4
    input: keywords
    return: api names
    '''
    mashup_collection = database['MASHUP_DATA']
    query = {"$and": []}

    for keyword in range(len(keywords)):
        query["$and"].append({"$or": [
            {"summary": {"$regex": f".*{keywords[keyword]}.*", "$options": "i"}},
            {"title": {"$regex": f".*{keywords[keyword]}.*", "$options": "i"}},
            {"description": {"$regex": f".*{keywords[keyword]}.*", "$options": "i"}}
        ]})

    result = mashup_collection.find(query)
    api_names = [record["name"] for record in result]
    return api_names


def query5(k):
    '''
    Desc: Performs query 5
    input: k
    return: names of top K APIs that are most frequently used in mashups
    '''
    with open("data_api.json", "r") as json_file:
        data_from_file = json.load(json_file)
        top_k_values = list(data_from_file.items())[:k]
        first_params = [param for param, _ in top_k_values[:k]]
        return first_params


def query6(k):
    '''
    Desc: Performs query 6
    input: k
    return: names of top K mashups that have the greatest number of APIs
    '''
    with open("data_api_len.json", "r") as json_file:
        data_from_file = json.load(json_file)
        top_k_values = list(data_from_file.items())[:k]
        first_params = [param for param, _ in top_k_values[:k]]
        return first_params


# route for home page
@app.route("/index", methods=["GET", "POST"])
def index():
    '''
    Desc: returns the home page
    '''
    return render_template("index.html")

# route for query 1 and 2
@app.route("/searchtype1")
def searchtype1():
    '''
    return: returns the page for searching records based on 
    update years, protocols, tags,
    '''
    return render_template("searchtype1.html")

# route for query 3 and 4
@app.route("/submitKeywords", methods=["GET", "POST"])
def submitKeywords():
    '''
    Desc: Performs the filtering based on keywords data
    return: returns the page for query 3 and 4
    '''
    if request.method == "POST":
        # Access form data
        mashup = request.form.get("checkbox1")
        inputField = request.form.get("tags")
        return_data = []
        temp_list = inputField.split(",")

        # condition for returning the type for query to perform
        if mashup:
            return_data = query4(temp_list)
        else:
            return_data = query3(temp_list)

        return render_template("searchtype1.html", results=return_data)

# route for query 5 and 6
@app.route("/apinames", methods=["GET", "POST"])
def apinames():
    '''
    Desc: Performs query based on checkbox selected by the user
    return: returns the page for query 5 and 6
    '''
    if request.method == "POST":
        # Access form data
        mashup = request.form.get("checkbox1")
        api = request.form.get("checkbox2")
        inputField = request.form.get("inputField")
        return_data = []
        # condition for returning the type for query to perform
        if mashup:
            return_data = query5(int(inputField))
        else:
            return_data = query6(int(inputField))

        return render_template("apiname.html", results=return_data)
    return render_template("apiname.html", results=[])

# route for query 1
@app.route("/resultsapi",  methods=["GET", "POST"])

def resultsapi():
    '''
    Desc: Performs query based on input provided by the user
    return: returns the page for query 1
    '''
    return_data = []
    tags_list = []
    if request.method == "POST":
    # Access form data
        updated_year = request.form.get("updated_year")
        protocol = request.form.get("protocol")
        category = request.form.get("Category")
        ratings = int(request.form.get("ratings"))
        ratings_drop = request.form.get("ratings_drop")
        tags  = request.form.get("tags")
        tags_list = tags.split(",")
        return_data = query1(updated_year, protocol, ratings_drop, ratings, tags_list, category)
        return render_template("apiquery.html", results=return_data)

    return render_template("apiquery.html", results=[])


# route for query 2
@app.route("/resultsmashup",  methods=["GET", "POST"])
def resultsmashup():
    '''
    Desc: Performs query based on input provided by the user
    return: returns the page for query 2
    '''
    return_data = []
    tags_list = []
    apis_list =[]
    if request.method == "POST":
    # Access form data
        updated_year = request.form.get("updated_year")
        tags  = request.form.get("tags")
        tags_list = tags.split(",")
        apis = request.form.get("apis")
        apis_list = apis.split(",")
        return_data = query2(updated_year, apis_list, tags_list)
        print(return_data)

        return render_template("mashupquery.html", results=return_data)
    return render_template("mashupquery.html", results=[])



if __name__ == "__main__":
    # main
    serve(app, host="127.0.0.1", port=8000)
