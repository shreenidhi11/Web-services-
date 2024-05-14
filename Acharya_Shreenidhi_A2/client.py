#####################################################################
# Author: Shreenidhi Acharya (sa8267)
# Description: The client.py file utilizes the webservices provided by
# server(main.py). The client invokes all the endpoints in order and
# out of order. This file tests the working of the web coordination
# policy for login and place order.
#####################################################################

import requests
from pprint import PrettyPrinter
pp = PrettyPrinter()
s = requests

# login first
def login():
    response_login = s.get("http://127.0.0.1:8000/login",
                           headers={"username": "Shree", "password": "shree123"})
    print(response_login.text)
    

# Make a request to /displayLaptops to get all the laptops
def display():
    headers = {"username": "Shree", "password": "shree123"}
    data = {"action": "displayLaptops"}
    response_get_all_laptops = s.get(
        "http://127.0.0.1:8000/displayLaptops", headers=headers, params=data)
    pp.pprint(response_get_all_laptops.text)

# Make a request to place the order
def placeOrder():
    headers = {"username": "Shree", "password": "shree123"}
    data = {"action": "placeOrder", "Sequential_ID": "21"}
    response_get_placed_laptop = s.get(
        "http://127.0.0.1:8000/placeOrder", headers=headers, params=data)
    print(response_get_placed_laptop.text)

# Return a laptop
def returnOrder():
    headers = {"username": "Shree", "password": "shree123"}
    data = {"action": "returnOrder", "Sequential_ID": "21"}
    response_get_placed_laptop = s.get(
        "http://127.0.0.1:8000/returnOrder", headers=headers, params=data)
    print(response_get_placed_laptop.text)


def logout():

    # Logout
    response_login = s.get("http://127.0.0.1:8000/logout",
                           headers={"username": "Shree", "password": "shree123"})
    print(response_login.text)


def runOperationsInOrder_test1():
    login()
    display()
    placeOrder()
    returnOrder()
    logout()

def runOperationsInOrder_test2():
    login()
    placeOrder()
    logout()

def runOperationsInOrder_test3():
    login()
    display()
    returnOrder()
    logout()



if __name__ == "__main__":
    print("******************Order your Laptop******************\n\n")
    # Run all the functions in order
    print("Test 1: All the operations run in order") 
    runOperationsInOrder_test1()
    print("\n\n")

    # Try to place an order without displaying all the laptop details
    print("Test 2: User tries to place order before displaying the laptops")
    runOperationsInOrder_test2()
    print("\n\n")

    # Try to return an order without placing it
    print("Test 3: User tries to return the laptop before placing the order")
    runOperationsInOrder_test3()
    print("\n\n")






