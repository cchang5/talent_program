from flask import render_template
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2

from flask import request

# Python code to connect to Postgres
# You may need to modify this based on your OS,
# as detailed in the postgres dev setup materials.
user = "admin"  # add your Postgres username here
host = "localhost"
dbname = "talentdb"
port = "5430"
db = create_engine("postgres://%s%s/%s" % (user, host, dbname))
con = None
con = psycopg2.connect(
    database=dbname, user=user, host=host, password="password", port=port
)  # add your Postgres password here


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/db")
def streamers():
    sql_query = """                                                                       
                SELECT * FROM streamer;
                """
    query_results = pd.read_sql_query(sql_query, con)
    streamers = ""
    for i in range(len(query_results)):
        streamers += query_results.iloc[i]["display_name"]
        streamers += "<br>"
    return streamers


@app.route("/db_fancy")
def streamers_fancy():
    sql_query = """
                SELECT * FROM streamer;
                """
    query_results = pd.read_sql_query(sql_query, con)
    streamers = []
    for i in range(0, query_results.shape[0]):
        streamers.append(
            dict(
                id=query_results.iloc[i]["id"],
                display_name=query_results.iloc[i]["display_name"],
                tier=query_results.iloc[i]["tier"],
            )
        )
    return render_template("streamers.html", streamers=streamers)

@app.route("/output")
def streamers_output():
    # pull 'tier' from input field and store it
    tier = dict()
    tier['CHALLENGER'] = request.args.get("tier_challenger")
    tier['GRANDMASTER'] = request.args.get("tier_grandmaster")
    tier['MASTER'] = request.args.get("tier_master")
    tier['DIAMOND'] = request.args.get("tier_diamond")
    tier['PLATINUM'] = request.args.get("tier_platinum")
    tier['GOLD'] = request.args.get("tier_gold")
    tier['SILVER'] = request.args.get("tier_silver")
    tier['BRONZE'] = request.args.get("tier_bronze")
    tier['IRON'] = request.args.get("tier_iron")
    join_tier = "' OR tier = '".join([key for key in tier if tier[key] == 'on'])
    query_tier = f"(tier='{join_tier}')"
    rank = request.args.get("rank")
    print("OUTPUT")
    print(tier)
    print(rank)
    # select from the database for the tier that the user inputs
    query = f"SELECT * FROM streamer WHERE {query_tier} AND rank='{rank}' ORDER BY tier, rank;"
    print(query)
    query_results = pd.read_sql_query(query, con)
    print(query_results)
    streamers = []
    for i in range(0, query_results.shape[0]):
        streamers.append(
            dict(
                id=query_results.iloc[i]["id"],
                display_name=query_results.iloc[i]["display_name"],
                tier=query_results.iloc[i]["tier"],
                rank=query_results.iloc[i]["rank"]
            )
        )
        the_result = ""
    return render_template("output.html", streamers=streamers, the_result=the_result)
