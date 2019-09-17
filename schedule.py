import talent_program as tp
import chatlog
import numpy as np
import calendar
import datetime
import pytz
import matplotlib.pyplot as plt
import calmap
import pandas as pd


def timeofweek_convert(timeofweek):
    convert = dict()
    convert[0] = "Monday"
    convert[1] = "Tuesday"
    convert[2] = "Wednesday"
    convert[3] = "Thursday"
    convert[4] = "Friday"
    convert[5] = "Saturday"
    convert[6] = "Sunday"
    return convert[timeofweek]


def get_streamer_id(streamer):
    query = f"SELECT id FROM streamer WHERE display_name='{streamer}'"
    postgres = tp.Postgres()
    streamer_id = np.array(postgres.rawselect(query))[0, 0]
    postgres.close()
    return streamer_id


def get_schedule(streamer):
    # query = f"SELECT id FROM streamer WHERE display_name='{streamer}'"
    # postgres = tp.Postgres()
    # streamer_id = np.array(postgres.rawselect(query))[0, 0]
    # postgres.close()
    streamer_id = get_streamer_id(streamer)
    query = (
        f"SELECT id, created_at, duration FROM vod where streamer_id='{streamer_id}'"
    )
    postgres = tp.Postgres()
    vodtimes = np.array(postgres.rawselect(query))
    postgres.close()
    for vodtime in vodtimes:
        vod_id = vodtime[0]
        duration = chatlog.parse_duration(vodtime[2])
        vod_start, vod_end, dayofweek = get_session(vodtime, duration)
        postgres = tp.Postgres()
        query = f"INSERT INTO schedule (streamer_id, vod_id, vod_start, vod_end, duration, dayofweek) VALUES ('{streamer_id}', '{vod_id}', '{vod_start}', '{vod_end}', '{duration}', '{dayofweek}')"
        postgres.rawsql(query)
        postgres.close()


def get_session(vodtime, duration):
    YY, MM, DD = vodtime[1].split("T")[0].split("-")
    hh, mm, ss = vodtime[1][:-1].split("T")[1].split(":")
    endtime = datetime.datetime(
        int(YY), int(MM), int(DD), int(hh), int(mm), int(ss), tzinfo=pytz.utc
    )
    delta = datetime.timedelta(seconds=int(duration))
    starttime = endtime - delta
    dayofweek = starttime.weekday()
    return starttime, endtime, dayofweek


def visualize_calendar():
    print(calendar.month(theyear=2019, themonth=9))
    calhtml = calendar.HTMLCalendar()
    print(calhtml.formatmonth(theyear=2019, themonth=9))


def insert_schedule_to_db():
    streamers = chatlog.get_display_name()
    for streamer in streamers:
        get_schedule(streamer)


def visualize_schedule(streamer):
    streamer_id = get_streamer_id(streamer)
    query = (
        f"SELECT vod_start, duration FROM schedule WHERE streamer_id = {streamer_id};"
    )
    postgres = tp.Postgres()
    records = np.array(postgres.rawselect(query))
    postgres.close()
    days = [r[0].date() for r in records]
    years = np.unique([r[0].year for r in records])
    duration = [r[1] for r in records]
    events = pd.DataFrame(duration, columns=["duration"])
    events = events.set_index(pd.DatetimeIndex(pd.to_datetime(days)))["duration"]
    # for year in years:
    #    cplot = calmap.yearplot(events, year=year)
    #    plt.draw()
    #    plt.show()
    fig = calmap.calendarplot(
        events,
        yearascending=True,
        yearlabel_kws={"fontsize": 12, "color": "k"},
        monthticks=3,
        daylabels="MTWTFSS",
        dayticks=[0, 2, 4, 6],
        cmap="Reds",
        fillcolor="whitesmoke",
        linewidth=1,
        fig_kws=dict(figsize=(8, 6)),
        gridspec_kws={"hspace": 0.1}
    )
    plt.draw()
    plt.savefig(f"./annual_schedule/{streamer_id}.pdf", transparent=True)

def generate_visual_schedule():
    streamers = chatlog.get_display_name()
    for streamer in streamers[:5]:
        visualize_schedule(streamer)


if __name__ == "__main__":
    ### Process VOD information into schedule
    # insert_schedule_to_db()

    ### Pretty plots for streaming schedule
    generate_visual_schedule()
