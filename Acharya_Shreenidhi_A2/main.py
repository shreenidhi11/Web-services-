 #####################################################################
# Author: Shreenidhi Acharya (sa8267)
# Description: The main.py file is the server which defines all the
# endpoints to be utilized. Also it performs validation of the input
# and output data such as the client credentials and laptops. This file
# defines the web coordination policy between operation such as login, 
# placing order, return order etc.
#####################################################################

from typing import Annotated
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Header, HTTPException, Depends, Request, Response
from pydantic import BaseModel
from pymongo import MongoClient
import models as models
import motor.motor_asyncio
from bson import ObjectId
from pymongo import ReturnDocument
from pydantic.functional_validators import BeforeValidator
from typing import Optional, List, Union

app = FastAPI()
# define the sequence of the operation at server side.
sequence = {"login": 1, "displayLaptops": 2, "placeOrder": 3, "returnOrder": 4}

# initialize the database 
mongo_uri = "mongodb+srv://root1234:root1234@cluster0.c0ggoiz.mongodb.net/LaptopDetails?ssl=true"
client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
db = client.LaptopDetails
laptop_collection = db.get_collection("laptops")
user_collection = db.get_collection("userDetails")
user_communication = db.get_collection("userCommunication")

# created a basemodel for user validation
class UserCredentailModel(BaseModel):
    username: str
    password: str

#1. endpoint for login
@app.get("/login")
async def root(request: Request, response: Response):
    """
    Store client credentials in database
    """
    usr = request.headers.get("username")
    pwd = request.headers.get("password")
    user = await user_collection.find_one({"username": usr})
    if user is None:
        # add this user to the database
        user_data = {
            "username": usr,
            "password": pwd
        }
        user_collection.insert_one(user_data)
        # also add this communication to the user communication table
        user_communicate = {
            "username": usr,
            "task_id": 1
        }
        user_communication.insert_one(user_communicate)
    return ("You are ready to shop!")

#2. endpoint for display laptop
@app.get("/displayLaptops", response_description="Display all laptops", response_model=Union[List[models.LaptopDetails],str])
async def get_laptops(request: Request, response: Response):
    """
    Display all the laptops to the client
    """
    get_the_user = request.headers.get('username')
    user_curr_comm_id = await db.userCommunication.find_one({"username": get_the_user}, {"task_id": 1, "_id": 0})
    # you can get None or some result
    if user_curr_comm_id:
        if user_curr_comm_id.get("task_id") is not None:
            # update the sequence order now for the current user
            new_user_comm_id = user_curr_comm_id.get("task_id") + 1
            # update the sequence of communication in the database
            db.userCommunication.update_one(
                    {"username": get_the_user},
                    {"$set": {"task_id": new_user_comm_id}}
                )
            orders = await laptop_collection.find().to_list(None)
            return orders
    else:
        return "User not logged"
        
#3. endpoint for placing order
@app.get("/placeOrder")
async def place_order(Sequential_ID, request: Request):
    """
    Place order for the client
    """
    get_the_user = request.headers.get('username')
    user_curr_comm_id = await user_communication.find_one({"username": get_the_user}, {"task_id": 1, "_id": 0})
    # User login present check
    if user_curr_comm_id is not None:
        if user_curr_comm_id.get("task_id") + 1 == sequence["placeOrder"]:
            # check for the laptop in the database
            laptop_result = await laptop_collection.find_one({"Sequential_ID": int(Sequential_ID)}, {"Sequential_ID": 1, "_id": 0})
            if len(laptop_result) >= 1:
                # update the sequence order now for the current user
                new_user_comm_id = user_curr_comm_id.get("task_id") + 1
                await db.userCommunication.update_one(
                    {"username": get_the_user},
                    {"$set": {"task_id": new_user_comm_id}}
                )
                return "Order Placed"
            else:
                return "Product out of stock"
        else:
            return "Wrong sequence, you need to display all the laptops first"
    else:
        return "User not logged"

#4. endpoint for return order
@app.get("/returnOrder")
async def return_order(Sequential_ID, request: Request):
    """
    Place return order for the client
    """
    get_the_user = request.headers.get('username')
    user_curr_comm_id = await db.userCommunication.find_one({"username": get_the_user}, {"task_id": 1, "_id": 0})
    #User login present check
    if user_curr_comm_id is not None:
        if user_curr_comm_id.get("task_id") + 1 == sequence["returnOrder"]:
            # check for the laptop in the database
            db.userCommunication.update_one(
                    {"username": get_the_user},
                    {"$set": {"task_id": user_curr_comm_id.get("task_id") + 1}}
                )
            return "Return Order Placed"
        else:
            return "Wrong sequence, you need to place the laptop order first"
    else:
        return "User not logged"


#5. endpoint for logout
@app.get("/logout")
async def root(request: Request, response: Response):
    """
    Log the client out
    """
    usr = request.headers.get("username")
    pwd = request.headers.get("password")
    user = await user_collection.find_one({"username": usr})
    if user is not None:
        await db.userDetails.delete_one({"username":usr})
        await db.userCommunication.delete_one({"username":usr})
        return "User logged out!"
    else:
        return "User does not exist"
