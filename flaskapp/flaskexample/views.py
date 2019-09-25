from flask import render_template
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import numpy as np
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
    where_query = []
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
    if join_tier == "":
        pass
    else:
        where_query.append(f"(tier='{join_tier}')")
    # pull day of week
    week = dict()
    week["mon"] = request.args.get("tow_monday")
    week["tue"] = request.args.get("tow_tuesday")
    week["wed"] = request.args.get("tow_wednesday")
    week["thu"] = request.args.get("tow_thursday")
    week["fri"] = request.args.get("tow_friday")
    week["sat"] = request.args.get("tow_saturday")
    week["sun"] = request.args.get("tow_sunday")
    join_week = [f"{key} > 4" for key in week if week[key] == "on"]
    if join_week == []:
        pass
    else:
        where_query.append("(%s)" % " AND ".join(join_week))
    # pull time of day
    request_start = request.args.get("tod_start")
    request_end = request.args.get("tod_end")
    if request_start == "" or request_end == "":
        pass
    else:
        try:
            hour_start = int(request.args.get("tod_start"))
            hour_end = int(request.args.get("tod_end"))
        except:
            return render_template("index.html", error_message = "Start and End time needs to be a number!")
        if hour_start < hour_end:
            join_hour = np.arange(hour_start, hour_end + 1)
        else:
            join_hour = np.concatenate(
                (np.arange(hour_end, 24), np.arange(0, hour_start + 1))
            )
        join_hour = [f"day_part{hour} > 0.25" for hour in join_hour]
        where_query.append("(%s)" % " AND ".join(join_hour))
    # join where query
    if where_query == []:
        where_query = ""
    elif len(where_query) == 1:
        where_query = f"AND {where_query[0]}"
    else:
        where_query = " AND %s " % " AND ".join(where_query)
    # select from the database for the tier that the user inputs
    query = f"SELECT * FROM twitch_talent WHERE (proba > 0.1) {where_query} ORDER BY proba DESC;"
    print(query)
    query_results = pd.read_sql_query(query, con)
    print(query_results)
    streamers = []
    for i in range(0, query_results.shape[0]):
        streamers.append(
            dict(
                id=query_results.iloc[i]["id"],
                proba=int(100*query_results.iloc[i]["proba"]),
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
