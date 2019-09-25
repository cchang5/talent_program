# Build features for Star prediction
# Notes
# FEATURES
# 1) Time of week DONE
# Monday - Sunday, median hours (float)
# 2) Time of day DONE
# n-hours bin, sum(count) / nbins / days (float)
# 3) summonerLevel DONE
# integer
# 4) rank tier DONE
# percentile (float)
# 5) description
# Boolean (exist or not)

# Features from VODs
# 6) 19 robot / turbo emotes, scaled frequency
# 7) (many) global emotes, scaled frequency
# 8) 9 chat tags (admin, bits, broadcaster, global_mod, moderator, premium, staff, sub, turbo), frequency
## 8.1) broadcaster lines / total lines (or per minute),   mod / total lines
# 9) relative excitement
# lines per time_interval + 1 / median lines (+1 to account for zero?)

# LABELS
# 1) total viewer count (int)
# 2) broadcaster type ("", affiliate, partner)

import schedule as sch
import chatlog
import talent_program as tp
import pandas as pd
import numpy as np
import pickle
import sys
import json
import features_chatlog as fchat


def timeofweek(features_dict, flag, units):
    streamers = chatlog.get_display_name()
    if flag == "median":
        try:
            file = open("./features/timeofweek_median.pickle", "rb")
            preload = pickle.load(file)
            print("Preloading timeofweek feature")
            for key in preload:
                if key in ["columns"]:
                    features_dict[key].extend(preload[key])
                    continue
                if units == "seconds":
                    features_dict[key].extend(preload[key])
                elif units == "minutes":
                    features_dict[key].extend(list(np.array(preload[key]) / 60.0))
                elif units == "hours":
                    features_dict[key].extend(list(np.array(preload[key]) / 3600.0))
        except:
            raise Exception
            print("Generating timeofweek feature")
            columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            features_dict["columns"].extend(columns)
            for_pickle = make_empty_dict()
            for_pickle["columns"].extend(columns)
            for streamer in streamers:
                try:
                    result = sch.weekly_schedule(streamer, False)
                except:
                    result = {dayofweek: [] for dayofweek in range(7)}
                median = []
                for dayofweek in range(7):
                    median.append(np.median(result[dayofweek]))
                if units == "seconds":
                    features_dict[streamer].extend(median)
                elif units == "minutes":
                    features_dict[streamer].extend(list(np.array(median) / 60.0))
                elif units == "hours":
                    features_dict[streamer].extend(list(np.array(median) / 3600.0))
                for_pickle[streamer].extend(median)
            file = open("./features/timeofweek_median.pickle", "wb")
            pickle.dump(for_pickle, file)
            file.close()
    return features_dict


def timeofday(features_dict, bins):
    try:
        file = open(f"./features/timeofday_{bins}bins.pickle", "rb")
        preload = pickle.load(file)
        print(f"Preloading timeofday {bins} bins feature")
        for key in preload:
            features_dict[key].extend(preload[key])
    except:
        print(f"Generating timeofday {bins} bins feature")
        columns = [f"day_part{bin}" for bin in range(bins)]
        features_dict["columns"].extend(columns)
        # set up dict to pickle
        for_pickle = make_empty_dict()
        for_pickle["columns"].extend(columns)
        # end
        streamers = chatlog.get_display_name()
        for streamer in streamers:
            try:
                result, days = sch.daily_schedule(streamer, False)
                result_list = np.array(result.values).flatten()
            except:
                print("Unexpected error:", sys.exc_info())
                freshday = sch.make_freshday()
                result_list = np.zeros(len(freshday))
                days = 0
            binsize = int(np.ceil(len(result_list) / bins))
            result_binned = np.nanmax(
                np.pad(
                    result_list.astype(float),
                    (0, binsize - result_list.size % binsize),
                    mode="constant",
                    constant_values=np.NaN,
                ).reshape(-1, binsize),
                axis=1,
            )
            # number of sessions per day during time of day
            result_norm = result_binned / days
            features_dict[streamer].extend(result_norm)
            for_pickle[streamer].extend(result_norm)
        file = open(f"./features/timeofday_{bins}bins.pickle", "wb")
        pickle.dump(for_pickle, file)
        file.close()
    if False: # write to sql
        for streamer in preload:
            if streamer == "columns":
                continue
            query = f"SELECT id FROM streamer WHERE display_name='{streamer}'"
            postgres = tp.Postgres()
            streamer_id = np.array(postgres.rawselect(query))[0,0]
            postgres.close()
            hourlyQ = ", ".join([str(i) for i in preload[streamer]])
            coltag = ["day_part%s" %idx for idx in range(24)]
            coltagQ = ", ".join(coltag)
            query = f"INSERT INTO hourly_proba (streamer_id, {coltagQ}) VALUES ({streamer_id}, {hourlyQ});"
            postgres = tp.Postgres()
            postgres.rawsql(query)
            postgres.close()

    return features_dict


def summonerlevel(features_dict):
    query = "SELECT display_name, summonerlevel FROM streamer;"
    postgres = tp.Postgres()
    records = postgres.rawselect(query)
    postgres.close()
    records_dict = {record[0]: [record[1]] for record in records}
    records_dict["columns"] = ["summonerlevel"]
    for display_name in features_dict:
        features_dict[display_name].extend(records_dict[display_name])
    return features_dict


def tier_rank(features_dict):
    query = "SELECT display_name, tier, rank FROM streamer;"
    postgres = tp.Postgres()
    records = postgres.rawselect(query)
    postgres.close()
    # load percentile of tier rank
    with open("./lookup/ranktier.json", "r") as file:
        percentile = json.load(file)
    records_dict = {record[0]: [percentile[record[1]][record[2]]] for record in records}
    records_dict["columns"] = ["ranktier"]
    # push into features
    for display_name in features_dict:
        features_dict[display_name].extend(records_dict[display_name])
    return features_dict


def parse_emoji():
    pass


def make_empty_df():
    streamers = chatlog.get_display_name()
    df = pd.DataFrame(index=streamers)
    df.index.name = "display_name"
    return df


def make_empty_dict():
    streamers = chatlog.get_display_name()
    features_dict = {streamer: [] for streamer in streamers}
    features_dict["columns"] = []
    return features_dict


def make_features():
    features_dict = make_empty_dict()

    # LABEL DATA
    #features_dict = label_data(features_dict, success_list=["partner", "affiliate"])
    features_dict = label_data(features_dict, success_list=["partner"])

    # switch on and off features here

    # FEATURES FROM TWITCH AND RIOT API
    # Feature for Time of Week

    # get median amount of time streamed
    features_dict = timeofweek(features_dict, "median", "hours")

    # Feature for Time of Day
    features_dict = timeofday(features_dict, bins=10)

    # Feature for summoner level
    features_dict = summonerlevel(features_dict)

    # Feature for tier rank
    features_dict = tier_rank(features_dict)

    ### CHAT FEATURES
    features_dict = fchat.tarray_feature(features_dict, binsize=60)

    # make into dataframe
    columns = features_dict["columns"]
    del features_dict["columns"]
    features = pd.DataFrame.from_dict(
        data=features_dict, orient="index", columns=columns
    )

    # FEATURES FROM CHAT
    print(features.dropna())
    print("feature length:", len(features))
    print("dropna length:", len(features.dropna()))
    return features


def label_data(features_dict, success_list=["partner"]):
    query = "SELECT display_name, broadcaster_type FROM streamer;"
    postgres = tp.Postgres()
    records = postgres.rawselect(query)
    postgres.close()
    records_dict = {record[0]: record[1] for record in records}
    records_dict["columns"] = "label"
    for display_name in features_dict:
        if display_name in ["columns"]:
            features_dict[display_name].extend(["label"])
            continue
        if records_dict[display_name] in success_list:
            features_dict[display_name].extend([1])
        else:
            features_dict[display_name].extend([0])
    return features_dict


if __name__ == "__main__":
    features = make_features()
