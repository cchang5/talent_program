import talent_program as tp
import chatlog
import numpy as np
import calendar
import datetime
import pytz
import matplotlib.pyplot as plt
import calmap
import pandas as pd
from collections import Counter
from collections import OrderedDict


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


def annual_schedule(streamer):
    streamer_id = get_streamer_id(streamer)
    query = (
        f"SELECT vod_start, duration FROM schedule WHERE streamer_id = {streamer_id} AND vod_start > '2019-07-17T00:00:00Z';"
    )
    postgres = tp.Postgres()
    records = np.array(postgres.rawselect(query))
    postgres.close()
    days = [r[0].date() for r in records]
    years = np.unique([r[0].year for r in records])
    duration = [r[1] for r in records]
    events = pd.DataFrame(duration, columns=["duration"])
    events = events.set_index(pd.DatetimeIndex(pd.to_datetime(days)))["duration"]
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
        fig_kws=dict(figsize=(8, 2)),
        gridspec_kws={"hspace": 0.1},
    )
    plt.draw()
    #plt.savefig(f"./annual_schedule/{streamer_id}.pdf", transparent=True)
    plt.savefig(f"./flaskapp/flaskexample/static/annual_schedule/{streamer_id}.png", dpi=300, transparent=False)
    plt.close()

# count number of week days
# https://stackoverflow.com/questions/27391236/number-of-times-each-weekday-occurs-between-two-dates
def dates_between(start, end):
    while start <= end:
        yield start
        start += datetime.timedelta(1)


def count_weekday(start, end):
    counter = Counter()
    for date in dates_between(start, end):
        counter[date.weekday()] += 1
    return counter


# end


def weekly_schedule(streamer, plot=True):
    streamer_id = get_streamer_id(streamer)
    # only look in the past 60 days (VOD storage policy)
    query = f"SELECT vod_start, duration, dayofweek FROM schedule WHERE streamer_id = {streamer_id} and vod_start > '2019-07-17T00:00:00Z';"
    postgres = tp.Postgres()
    records = np.array(postgres.rawselect(query))
    postgres.close()
    mindate = min(records[:, 0]).date()
    maxdate = max(records[:, 0]).date()
    tdelta = (maxdate - mindate).days
    weekday_counts = count_weekday(mindate, maxdate)
    total_streamtime = {day: [] for day in range(7)}
    ### this block sums all stream times starting from same day
    clean_records = {
        record[0].date(): [record[0].date(), 0, record[2]] for record in records
    }
    for record in records:
        clean_records[record[0].date()][1] += record[1]
    records = []
    for key in clean_records:  # format to pgsql return
        records.append(clean_records[key])
    ###
    for record in records:
        total_streamtime[record[2]].append(int(record[1]))
    for day in total_streamtime:
        while len(total_streamtime[day]) < weekday_counts[day]:
            total_streamtime[day].append(0)
    if plot:
        position = range(7)
        x = [np.array(total_streamtime[day]) / 3600.0 for day in range(7)]
        fig = plt.figure(figsize=(7, 4))
        ax = plt.axes([0.15, 0.15, 0.8, 0.8])
        ax.axhline(8, ls="-", color="k", alpha=0.2)
        bp = ax.boxplot(x, sym="k+", positions=position)
        ax.set_ylabel("How long will I stream? (hours)", fontsize=12)
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
        ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_yticks([0, 8, 16, 24])
        ax.annotate(
            f"Past {tdelta} days (max up to 60 days)",
            xy=(1, -0.1),
            fontsize=8,
            ha="right",
            va="top",
            xycoords="axes fraction",
        )
        plt.draw()
        #plt.savefig(f"./weekly_schedule/{streamer_id}.pdf", transparent=True)
        plt.savefig(f"./flaskapp/flaskexample/static/weekly_schedule/{streamer_id}.png", dpi=300, transparent=False)
        plt.close()
    else:
        return total_streamtime


def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta


def make_freshday(minutes=1):
     #freshday = OrderedDict()
     #for dt in datetime_range(
     #    datetime.datetime(year=2019, month=1, day=1, hour=0, minute=0),
     #    datetime.datetime(year=2019, month=1, day=2, hour=0, minute=0),
     #    datetime.timedelta(minutes=minutes),
     #):
     #    freshday[dt.strftime("%H:%M")] = 0
     freshday = {
        dt.strftime("%H:%M"): 0
        for dt in datetime_range(
            datetime.datetime(year=2019, month=1, day=1, hour=0, minute=0),
            datetime.datetime(year=2019, month=1, day=1, hour=23, minute=59),
            datetime.timedelta(minutes=minutes),
        )
     }
     return freshday


def daily_schedule(streamer, plot=True):
    streamer_id = get_streamer_id(streamer)
    query = f"SELECT vod_start, vod_end FROM schedule WHERE streamer_id = {streamer_id} and vod_start > '2019-07-17T00:00:00Z';"
    postgres = tp.Postgres()
    records = np.array(postgres.rawselect(query))
    postgres.close()
    freshday = make_freshday()
    dschedule = pd.DataFrame.from_dict(data=freshday, orient="index", columns=["count"])
    for record in records:
        emptyday = pd.DataFrame.from_dict(
            data=freshday, orient="index", columns=["count"]
        )
        workday = {
            dt.strftime("%H:%M"): 1
            for dt in datetime_range(
                record[0], record[1], datetime.timedelta(minutes=1)
            )
        }
        workday = pd.DataFrame.from_dict(
            data=workday, orient="index", columns=["count"]
        )
        workday = (emptyday + workday).fillna(int(0))
        dschedule += workday
    mindate = min(records[:, 0]).date()
    maxdate = max(records[:, 0]).date()
    tdelta = (maxdate - mindate).days
    if plot:
        fig = plt.figure(figsize=(7, 4))
        ax = plt.axes([0.15, 0.15, 0.8, 0.8])
        Xlabel = dschedule.index
        X = range(len(Xlabel))
        y = dschedule["count"].values
        ax.bar(
            x=X, height=y, width=1.0, bottom=0, align="center", color="Red", alpha=0.7
        )
        ax.set_ylabel(f"Sessions in past {tdelta} days", fontsize=12)
        Xticks = X[:: 3 * 60]
        ax.set_xticks(Xticks)
        ax.set_xlim([X[0], X[-1]])
        ax.set_xticklabels(Xlabel[Xticks])
        # ax.set_yticks([0, 8, 16, 24])
        ax.annotate(
            f"Past {tdelta} days (max up to 60 days)",
            xy=(1, -0.1),
            fontsize=8,
            ha="right",
            va="top",
            xycoords="axes fraction",
        )
        plt.draw()
        #plt.savefig(f"./daily_schedule/{streamer_id}.pdf", transparent=True)
        plt.savefig(f"./flaskapp/flaskexample/static/daily_schedule/{streamer_id}.png", dpi=300, transparent=False)
        plt.close()
    else:
        return dschedule, tdelta


def generate_daily_schedule():
    streamers = chatlog.get_display_name()
    for streamer in streamers:
        try:
            daily_schedule(streamer)
        except:
            errorlog = open("./error.txt", "a")
            errorlog.write(f"daily: {streamer}")
            errorlog.close()
            pass


def generate_weekly_schedule():
    streamers = chatlog.get_display_name()
    for streamer in streamers:
        try:
            weekly_schedule(streamer)
        except:
            errorlog = open("./error.txt", "a")
            errorlog.write(f"weekly: {streamer}")
            errorlog.close()
            pass


def generate_annual_schedule():
    streamers = chatlog.get_display_name()
    for streamer in streamers:
        try:
            annual_schedule(streamer)
        except:
            errorlog = open("./error.txt", "a")
            errorlog.write(f"annual: {streamer}")
            errorlog.close()
            pass


if __name__ == "__main__":
    errorlog = open("./error.txt", "w")
    ### Process VOD information into schedule
    # insert_schedule_to_db()

    ### Pretty plots for streaming schedule
    generate_annual_schedule()

    ### Pretty weekly plots
    generate_weekly_schedule()

    ### Pretty daily plots
    generate_daily_schedule()
