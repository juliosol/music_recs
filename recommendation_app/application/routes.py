from application import app
from flask import Flask, render_template, request
#from application.features import *
#from application.model import *

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")