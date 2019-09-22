import requests
import urllib.parse as parse
import yaml
from riotwatcher import RiotWatcher, ApiError
import psycopg2 as psql
import sys
from time import sleep


class Postgres:
    def __init__(self):
        hostname = "localhost"
        database = "talentdb"
        username = "admin"
        password = "password"
        port = "5430"
        self.conn = psql.connect(
            host=hostname,
            port=port,
            database=database,
            user=username,
            password=password,
        )
        self.cur = self.conn.cursor()

    def rawsql(self, query):
        try:
            print(query)
            self.cur.execute(query)
            self.conn.commit()
            print("SUCCESS")
        except:
            print("Unexpected error:", sys.exc_info())
            return True

    def rawselect(self, query):
        try:
            print(query)
            self.cur.execute(query)
            records = self.cur.fetchall()
            print("SUCCESS")
            return records
        except:
            print("Unexpected error:", sys.exc_info())
            return True

    def close(self):
        self.conn.close()


class TwitchClient:
    def __init__(self):
        self.twitch_id = yaml.safe_load(open("/Users/cchang5/PycharmProjects/talent_program/client_id.yaml"))["twitch_id"]
        self.headers = {"Client-ID": "%s" % self.twitch_id}


class TopGames(TwitchClient):
    def get_top_games(self, ngames):
        url = "https://api.twitch.tv/helix/games/top?first=%s" % ngames
        get = requests.get(url, headers=self.headers).json()
        return get


class GameProfile(TwitchClient):
    def get_game_metadata(self, **kwargs):
        query = parse.urlencode(kwargs)
        url = "https://api.twitch.tv/helix/games?%s" % query
        self.result = requests.get(url, headers=self.headers).json()


class VideoProfile(TwitchClient):
    def get_videos(self, **kwargs):
        query = parse.urlencode(kwargs)
        url = "https://api.twitch.tv/helix/videos?%s" % query
        temp_result = requests.get(url, headers=self.headers).json()
        self.result = temp_result
        result_count = len(temp_result["data"])
        # get all videos that are type "archive"
        while result_count > 0:
            temp_result = requests.get(
                "%s&after=%s" % (url, temp_result["pagination"]["cursor"]),
                headers=self.headers,
            ).json()
            self.result["data"].extend(temp_result["data"])
            result_count = len(temp_result["data"])

        self.extract_metadata()

    def extract_metadata(self):
        for idx, entry in enumerate(self.result["data"]):
            streamer_id = entry["user_id"]
            id = entry["id"]
            title = (
                str(entry["title"].encode("utf-8"))[1:]
                .replace('"', "")
                .replace("'", "")
            )
            created_at = entry["created_at"]
            published_at = entry["published_at"]
            url = entry["url"]
            view_count = entry["view_count"]
            duration = entry["duration"]
            query = f"INSERT INTO vod (streamer_id, id, title, created_at, published_at, url, view_count, duration) VALUES ('{streamer_id}', '{id}', '{title}', '{created_at}', '{published_at}', '{url}', '{view_count}', '{duration}');"
            postgres = Postgres()
            postgres.rawsql(query)
            postgres.close()


class Stream(TwitchClient):
    def metadata(self, game_id):
        self.game_id = game_id

    def get_stream(self, **kwargs):
        # create pgsql instance
        # get stream metadata
        query = parse.urlencode(kwargs)
        url = "https://api.twitch.tv/helix/streams?%s" % query
        self.result = requests.get(url, headers=self.headers).json()
        # get streamer metadata
        streamer = StreamerProfile()
        streamer.get_user_metadata(id=self.result["data"][0]["user_id"])
        # print(streamer.result)
        # try to get summoner profile
        summoner = SummonerProfile()
        try:
            summoner.get_summoner(streamer.result["data"][0]["display_name"])
            flag = True
        except:
            flag = False
        # if summoner exists insert profile into pgsql
        if flag:
            try:
                streaminfo = self.result["data"][0]
                twitchinfo = streamer.result["data"][0]
                riotinfo = summoner.summoner_meta
                rankinfo = summoner.ranked_stats
                id = twitchinfo["id"]
                login = twitchinfo["login"]
                display_name = twitchinfo["display_name"]
                broadcaster_type = twitchinfo["broadcaster_type"]
                description = (
                    str(twitchinfo["description"].encode("utf-8"))[1:]
                    .replace('"', "")
                    .replace("'", "")
                )
                profile_image_url = twitchinfo["profile_image_url"]
                view_count = str(twitchinfo["view_count"])
                summonerlevel = str(riotinfo["summonerLevel"])
                live_count = streaminfo["viewer_count"]
                tier = rankinfo["tier"]
                rank = rankinfo["rank"]
                query = f"INSERT INTO streamer (id, login, display_name, broadcaster_type, description, profile_image_url, view_count, summonerlevel, tier, rank, live_count) VALUES ('{id}', '{login}', '{display_name}', '{broadcaster_type}', '{description}', '{profile_image_url}', '{view_count}', '{summonerlevel}', '{tier}', '{rank}', '{live_count}');"
                postgres = Postgres()
                signal = postgres.rawsql(query)
                postgres.close()
                if signal:
                    return 0
            except:
                print("Unexpected error:", sys.exc_info())
                return 0
        # if summoner exists then get vods
        if flag:
            videos = VideoProfile()
            videos.get_videos(user_id=streamer.result["data"][0]["id"], type="archive")
            # print(videos.result)


class StreamerProfile(TwitchClient):
    def get_user_metadata(self, **kwargs):
        query = parse.urlencode(kwargs)
        url = "https://api.twitch.tv/helix/users?%s" % query
        self.result = requests.get(url, headers=self.headers).json()
        # print(self.result)


class RiotClient:
    def __init__(self):
        self.riot_id = yaml.safe_load(open("/Users/cchang5/PycharmProjects/talent_program/client_id.yaml"))["riot_id"]
        self.watcher = RiotWatcher(self.riot_id)
        self.region = "na1"


class SummonerProfile(RiotClient):
    def get_summoner(self, name):
        self.summoner_meta = self.watcher.summoner.by_name(self.region, name)
        stats = self.watcher.league.by_summoner(
            self.region, self.summoner_meta["id"]
        )
        self.ranked_stats = False
        for statidx in stats:
            if statidx["queueType"] == "RANKED_SOLO_5x5":
                self.ranked_stats = statidx
                break
            else:
                pass

def record_page(stream):
    page = open("/Users/cchang5/PycharmProjects/talent_program/page.txt", "w")
    page.write(stream.result["pagination"]["cursor"])
    page.close()


def open_page():
    pagedoc = open("/Users/cchang5/PycharmProjects/talent_program/page.txt", "r")
    for line in pagedoc:
        page = line
    return page


if __name__ == "__main__":
    # get LOL game_id @ twitch.tv
    gameprofile = GameProfile()
    gameprofile.get_game_metadata(name="League of Legends")
    # get english speaking streams from LOL
    # build streamer profile
    # do one stream at a time
    stream = Stream()
    stream.metadata(gameprofile.result["data"][0]["id"])
    try:
        page = open_page()
        stream.get_stream(
            first=1,
            game_id=gameprofile.result["data"][0]["id"],
            language="en",
            after=page,
        )
    except:
        stream.get_stream(
            first=1, game_id=gameprofile.result["data"][0]["id"], language="en"
        )
    record_page(stream)
    for page in range(500):
        stream.get_stream(
            first=1,
            game_id=gameprofile.result["data"][0]["id"],
            language="en",
            after=stream.result["pagination"]["cursor"],
        )
        record_page(stream)
        sleep(1)
    # summoner = SummonerProfile()
    # summoner.get_summoner("Froggen")
