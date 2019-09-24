# library that parses chatlog for features
# import into features.py after this is finished.

import talent_program as tp
import chatlog
import features
import pandas as pd
import numpy as np
import pickle
import sys
import json
import datetime
import matplotlib.pyplot as plt

def strip_timestamp(logfile):
    pass


def check_logdir(logdir):
    if logdir[-3:] == "log":
        try:
            open(logdir, "r")
            return True
        except:
            print(f"corrupt file: {logdir}")
            return False
    else:
        print(f"bad directory to {logfile}, skipping")
        return False


def chat_frequency():
    # creates timearray and inserts into database
    streamers = chatlog.get_display_name()
    for streamer in streamers:
        # get streamer_id
        query = f"SELECT id FROM streamer WHERE display_name='{streamer}';"
        postgres = tp.Postgres()
        streamer_id = np.array(postgres.rawselect(query))[0, 0]
        postgres.close()
        # get chatlog list
        #query = f"SELECT id, chatlog FROM vod WHERE streamer_id='{streamer_id}' AND created_at > '2019-07-01T00:00:00Z'"
        query = f"SELECT v.id, v.chatlog FROM vod v LEFT JOIN chatlog c ON v.id = c.vod_id WHERE v.streamer_id='{streamer_id}' AND v.created_at > '2019-07-01T00:00:00Z' AND timearray IS null;"
        postgres = tp.Postgres()
        logdirs = np.array(postgres.rawselect(query))
        postgres.close()
        # strip for time stamps
        for logdir in logdirs:
            vod_id = logdir[0]
            # check if directory is good
            if not check_logdir(logdir[1]):
                continue
            # if good path
            timearray = []
            with open(logdir[1], "r") as chat:
                for line in chat:
                    timestamp = line[1:].split("]")[0].split(":")
                    minute = int(timestamp[1])
                    second = int(timestamp[2])
                    try:
                        hour = int(timestamp[0])
                        tdelta = int(
                            datetime.timedelta(
                                hours=hour, minutes=minute, seconds=second
                            ).total_seconds()
                        )
                    except:
                        dayhour = timestamp[0].split(" day, ")
                        day = int(dayhour[0])
                        hour = int(dayhour[1])
                        tdelta = int(
                            datetime.timedelta(
                                days=day, hours=hour, minutes=minute, seconds=second
                            ).total_seconds()
                        )

                    timearray.append(tdelta)
            if len(timearray) == 0:
                timearray = [0]
            query = f"INSERT INTO chatlog (streamer_id, vod_id, timearray) VALUES ('{streamer_id}', '{vod_id}', ARRAY{timearray});"
            postgres = tp.Postgres()
            postgres.rawsql(query)
            postgres.close()


def transformed_cut(median):
    B = 5600./1599.
    A = 5. - B
    with np.errstate(divide='ignore'):
        cutoff = A*median + B/median
    return cutoff

def tarray_feature(features_dict, binsize=60):
    # make features out of timearray of chatlogs
    try:
        file = open(f"./features/excitement.pickle", "rb")
        excitement_dict = pickle.load(file)
        print(f"Preloading excitement feature")
    except:
        excitement_dict = dict()
        excitement_dict["columns"] = ["excitement"]
        streamers = chatlog.get_display_name()
        for streamer in streamers:
            # get streamer_id
            query = f"SELECT id FROM streamer WHERE display_name='{streamer}';"
            postgres = tp.Postgres()
            streamer_id = np.array(postgres.rawselect(query))[0, 0]
            postgres.close()
            query = f"SELECT timearray FROM chatlog WHERE streamer_id={streamer_id};"
            postgres = tp.Postgres()
            records = postgres.rawselect(query)
            postgres.close()
            """
            for record in records[:1]:
                binned_chat = np.array(record[0])//binsize
                binned_counter = np.bincount(binned_chat)
                bins = len(binned_counter)
                fig = plt.figure(figsize=(7,4))
                ax = plt.axes([0.15, 0.15, 0.8, 0.8])
                ax.hist(binned_chat, bins=bins)
                plt.show()
            """
            concat_chat = [0]
            for record in records:
                concat_chat.extend(np.array(record[0])+concat_chat[-1])
            binned_chat = np.array(concat_chat) // binsize
            binned_counter = np.bincount(binned_chat)
            bins = len(binned_counter)
            sort_counter = np.sort(binned_counter)
            median = np.median(sort_counter)
            #print("median:", sort_counter[bins//2])
            #print("middle 90%:", sort_counter[int(bins*0.95)], sort_counter[int(bins*0.05)])
            excite_count = 0
            for event in sort_counter:
                if event > transformed_cut(median):
                    excite_count += 1
            #if sort_counter[bins//2] == 0:
            #    excite_count = 0
            #print("cutoff:", transformed_cut(median))
            #print(excite_count)
            excitements_per_hour = excite_count/(bins*binsize)*3600.
            excitement_dict[streamer] = [excitements_per_hour]
            #fig = plt.figure(figsize=(7, 4))
            #ax = plt.axes([0.15, 0.15, 0.8, 0.8])
            #ax.hist(binned_chat, bins=bins)
            #plt.show()
            file = open(f"./features/excitement.pickle", "wb")
            pickle.dump(excitement_dict, file)
            file.close()
    for key in features_dict:
        features_dict[key].extend(excitement_dict[key])
    return features_dict

def main():
    features_dict = features.make_empty_dict()
    #chat_frequency()
    features_dict = tarray_feature(features_dict)
    print(features_dict)


if __name__ == "__main__":
    main()
