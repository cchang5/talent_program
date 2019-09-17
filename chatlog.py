# get chat logs of VODs
import talent_program as tp
import numpy as np
import subprocess

def get_display_name():
    query = "SELECT DISTINCT display_name FROM streamer;"
    postgres = tp.Postgres()
    records = np.array(postgres.rawselect(query)).flatten()
    postgres.close()
    return records

def parse_duration(duration):
    if 'h' in duration:
        hour = int(duration.split('h')[0]) * 3600
        minute = int(duration.split('h')[1].split('m')[0]) * 60
        second = int(duration.split('h')[1].split('m')[1].split('s')[0])
    elif 'm' in duration:
        hour = int(0)
        minute = int(duration.split('m')[0]) * 60
        second = int(duration.split('m')[1].split('s')[0])
    else:
        hour = int(0)
        minute = int(0)
        second = int(duration.split('s')[0])
    return hour + minute + second

def get_vod_id(streamer):
    query = f"SELECT v.id, v.duration FROM streamer s JOIN vod v ON s.id = v.streamer_id where s.display_name = '{streamer}';"
    postgres = tp.Postgres()
    records = np.array(postgres.rawselect(query))
    postgres.close()
    vodtimes = np.array([[int(record[0]), int(parse_duration(record[1]))] for record in records])
    if len(vodtimes) == 0:
        return 0
    #for idx, vodtime in enumerate(vodtimes):
    #    print(vodtime, records[idx])
    #print(vodtimes[[entry[1] for entry in np.argsort(vodtimes, axis=0)]])
    vodid = vodtimes[np.argmax(vodtimes[:,1])][0]
    query = f"SELECT chatlog FROM vod where id = {vodid};"
    postgres = tp.Postgres()
    records = np.array(postgres.rawselect(query))
    postgres.close()
    if type(records[0,0]) == type(None):
        output = f"/Users/cchang5/PycharmProjects/talent_program/chatlog"
        command = f"tcd --video {vodid} --format irc --output {output}"
        subprocess.run(command, shell=True)
        query = f"UPDATE vod SET chatlog = '{output}/{vodid}.log' WHERE id = {vodid};"
        postgres = tp.Postgres()
        postgres.rawsql(query)
        postgres.close()
    else:
        print("chatlog already downloaded")

if __name__=="__main__":
    #streamers = get_display_name()
    #for streamer in streamers:
    #    get_vod_id(streamer)
    get_vod_id("Darkfirfox")