1. Upon extracting the zip folder, open the folder "Shreenidhi_Acharya_A3" in Visual Studio Code or pycharm.

2. Run the command "python -m venv ." in a terminal (CTRL+~) to create a virtual environment for this application.
Note: If you get error in the above command, you can use python3 -m venv .

3. Run the command "source bin/activate" to activate the created environment.

4. Run the command "pip install -r requirements.txt" to install all dependencies.
Note: if you encounter error with pip command use "pip3 install -r requirements.txt"  instead.

Part 1: Load the database. (This step can be skipped as I have already loaded the data in Mongodb atlas)

5. Open a new terminal and run the Python file “parsing_and_loading_data.py”. This command will create the databases at the backend and load them with values
Command: python parsing_and_loading_data.py
Note: if you encounter error use "python3 parsing_and_loading_data.py"  instead.
After running the above command two new JSON files namely data_api_len.json and data_api.json will be created. These two files will be used for running query 5 and query 6.
Note: These files are already provided along with the code files

Part 2: Running the application 

If you have skipped Part 1, directly run the below command
7. Now it's time to run the entire application. Run the Python file server.py to start the application. Once started open your browser and go to http://localhost:8000/index.
Command: python server.py
Note: if you encounter error use "python3 server.py"  