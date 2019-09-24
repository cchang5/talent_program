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


@app.route("/output")
def streamers_output():
    # pull 'tier' from input field and store it
    tier = dict()
    tier["CHALLENGER"] = request.args.get("tier_challenger")
    tier["GRANDMASTER"] = request.args.get("tier_grandmaster")
    tier["MASTER"] = request.args.get("tier_master")
    tier["DIAMOND"] = request.args.get("tier_diamond")
    tier["PLATINUM"] = request.args.get("tier_platinum")
    tier["GOLD"] = request.args.get("tier_gold")
    tier["SILVER"] = request.args.get("tier_silver")
    tier["BRONZE"] = request.args.get("tier_bronze")
    tier["IRON"] = request.args.get("tier_iron")
    join_tier = "' OR tier = '".join([key for key in tier if tier[key] == "on"])
    query_tier = f"(tier='{join_tier}')"
    print("OUTPUT")
    print(tier)
    # select from the database for the tier that the user inputs
    query = f"SELECT * FROM streamer WHERE {query_tier} ORDER BY tier, rank;"
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
                rank=query_results.iloc[i]["rank"],
                broadcaster_type=query_results.iloc[i]["broadcaster_type"],
            )
        )
    return render_template("output.html", streamers=streamers)


@app.route("/portfolio")
def streamers_portfolio():
    display_name = request.args.get("display_name")
    query = f"SELECT * FROM streamer WHERE display_name = '{display_name}';"
    print(query)
    query_results = pd.read_sql_query(query, con)
    streamer_id = query_results["id"].iloc[0]
    print(streamer_id)
    profile_image_url = query_results["profile_image_url"].iloc[0]
    tier = query_results["tier"].iloc[0]
    rank = query_results["rank"].iloc[0]
    description = query_results["description"].iloc[0]
    return render_template(
        "portfolio.html",
        display_name=display_name,
        streamer_id=streamer_id,
        profile_image_url=profile_image_url,
        tier=tier,
        rank=rank,
        description=description,
    )
