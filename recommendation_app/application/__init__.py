import os
from flask import Flask

# Create an instance of Flask app indicating its location
app_path = os.path.dirname(__file__)
app = Flask(__name__, static_folder=app_path+'/static') 
#create secret key for security
app.config['SECRET_KEY'] = 'f91fa17c60424fe9af925e3f3814f9cc'

from application import routes
