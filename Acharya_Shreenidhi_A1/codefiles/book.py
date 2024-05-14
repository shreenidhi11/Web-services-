#####################################################################
#Author: Shreenidhi Acharya (sa8267)
#Description: The book.py file serves as the core of this flask based
#application. It contains the operators to be performed when the user
#submits the form(for endpoint /book). This file contains the logic
#for handling API requests. This file implements the necessary
#business logic and displays the appropriate responses to user
#####################################################################
from dotenv import load_dotenv
from pprint import pprint as pp
import requests
import os
from zeep import Client

# Loads the env variables
load_dotenv()

def get_book_author_details(genre="Python"):
    """
    Description: Fetch data related to entered genre such as author name,
    book title, price, page count and topwork
    return: author name, book title, price or page count of book and topwork of author
    """

    flag = False

    # RESTAPI Call to Google Books API
    requestData = requests.get(
        "https://www.googleapis.com/books/v1/volumes?q={}&key={}".format(genre, os.getenv("GOOGLE_KEY")))
    result = requestData.json()

    # condition for deciding whether to return pagcount or price of the book 
    if not ('listPrice' in result['items'][0]['saleInfo']):
        flag = True
        pageCount = result['items'][0]['volumeInfo']['pageCount']
    else:
        amount = result['items'][0]['saleInfo']['listPrice']['amount']
        resultamount = None

    # fetching author and title
    author = result['items'][0]['volumeInfo']['authors']
    title = result['items'][0]['volumeInfo']['title']

    # SOAP API Call to Data Access
    client = Client(
        'https://www.dataaccess.com/webservicesserver/numberconversion.wso?WSDL')
    
    # condition for returning page count or price of the book based on flag value
    if flag:
        resultPageCount = (client.service.NumberToWords(
            pageCount) + " pages").capitalize()
    else:
        resultamount = client.service.NumberToDollars(amount)
        if not resultamount:
            resultamount = "Zero dollar"
        else:
            resultamount = resultamount.capitalize()

    # RESTAPI Call to Open Library API
    authorData = requests.get(
        "https://openlibrary.org/search/authors.json?q={}".format(author[0]))
    resultauthor = authorData.json()

    #  return the first author, title of the book, page count or price of the book and top work of the first author
    return author[0], title, resultamount if not flag else resultPageCount, resultauthor['docs'][0]['top_work']


if __name__ == "__main__":
    # Logging for backend server
    print('\n*** Enter Genre *** \n')
    genre = input("\n Please enter the genre name ")

    # Check for empty strings or string with only spaces
    if not bool(genre.strip()):
        genre = 'Python'

    # Get the details
    book_data = get_book_author_details(genre)
    print("\n")
    pp(book_data)
