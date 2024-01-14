# Disaster Response Pipeline Project
As a part of Data Science Nanodegree program by Udacity, this is second project named “Disaster Response Pipeline Project”. Dataset used for this project is based on pre-labelled tweet and messages from real-life disasters. Goal of this project is to create a machine learning pipelines to categorize the real messages that were sent during disaster events. Based on this categorization, we can send the messages to an appropriate disaster relief agency.

The Project is divided into the following Sections:
1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper database structure.
2. Machine Learning Pipeline to train a model which is able to classify text messages in 36 categories.
3. Web Application using Flask to show model results and predictions in real time.
# Folder Structure:

* app
    * templates
* master.html
* go.html
* run.py
* data
* disaster_categories.csv
* ML Pipeline Preparation.ipynb
* ETL Pipeline Preparation.ipynb
* disaster_messages.csv
* process_data.py
* Disaster_Response.db
* models
* train_classifier.py
* classifier.pkl
* README.md

# Disaster Response Pipeline Project at Github
### [Github Link for Project](https://github.com/parulgangwar/NanoDegree_Disaster_Mgmt_Project)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
