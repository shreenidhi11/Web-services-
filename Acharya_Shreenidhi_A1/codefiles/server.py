#####################################################################
#Author: Shreenidhi Acharya (sa8267)
#Description: The server.py file serves as the entry point of this flask
#based application. This file acts as a mediator between UI and backend.
#This file exposes all the endpoints required for this application
#####################################################################
from flask import Flask, render_template, request
from waitress import serve
from book import get_book_author_details

# Initializing the Flask instance
app = Flask(__name__)

# Endpoint for Landing Page
@app.route("/")
@app.route("/index")
def index():
    """
    Desc: Renders the home page
    return: returns the template for home page
    """
    return render_template('index.html')


#Endpoint for displaying details
@app.route("/book")
def get_details():
    """
    Desc: Renders the book details page
    return: returns the template for book details page
    """

    # Flag for deciding whether to display page count or price of book
    defaultflag = False

    # Fetch the genre from URL
    genre = request.args.get('genre')

    # Handling default genre as python
    if not bool(genre.strip()):
        defaultflag = True
        genre = 'Python'

    # Call to the main function from book.py that interacts with API
    details = get_book_author_details(genre)

    # store the details
    author_name = details[0]
    book_title = details[1]
    book_page_or_price = details[2]
    top_work = details[3]
 
    return render_template("book.html", flag=defaultflag, author_name=details[0],
                           book_title=details[1],
                           book_page_or_price=details[2],
                           top_work=details[3])


# Running the app on localhost port 8000
if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)
