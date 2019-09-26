# make graphic for streamer profile
import matplotlib.pyplot as plt
from features import *
from features_chatlog import *
import pandas as pd
import talent_program as tp


def count_dayofweek(Series):
    count = 0
    for day in Series:
        if day > 4:
            count += 1
        else:
            pass
    return count


def count_timeofday(Series):
    count = 0
    for time in Series:
        if time > 0.25:
            count += 1
        else:
            pass
    return count


def plot_pentagram(pent):
    figure = plt.figure(figsize=(8, 8))
    ax = figure.add_subplot(111, projection="polar")
    theta = 2 * np.pi * np.arange(len(pent) + 1) / len(pent)
    r = list(pent.values)
    r.append(r[0])
    colors = theta
    c = ax.errorbar(x=theta, y=r, color="#6441A4", ls="-")
    ax.fill_between(theta, 0, r, color="#6441A4", alpha=1)
    ax.set_ylim([0, 5.5])
    plt.xticks(
        theta, ("Strength", "Endurance", "Dexterity", "Charisma", "Vitality", "Infamy")
    )
    plt.yticks([1, 2, 3, 4, 5], ())
    plt.tick_params(axis="x", labelsize=16)
    plt.draw()
    #plt.show()
    postgres = tp.Postgres()
    query = f"SELECT id FROM STREAMER WHERE display_name='{pent.name}';"
    streamer_id = np.array(postgres.rawselect(query))[0, 0]
    plt.savefig(f"./flaskapp/flaskexample/static/hexagram/{streamer_id}.png", dpi=300, transparent=False)
    plt.close()


def main():
    norm = 5

    features_dict = make_empty_dict()
    features_dict = timeofweek(features_dict, "median", "hours")
    features_dict = timeofday(features_dict, bins=24)
    features_dict = tier_rank(features_dict)
    features_dict = tarray_feature(features_dict, binsize=60)
    features_dict = summonerlevel(features_dict)

    columns = features_dict["columns"]
    del features_dict["columns"]
    features = pd.DataFrame.from_dict(
        data=features_dict, orient="index", columns=columns
    ).fillna(0)

    # get day of week normalized
    dow_count = features[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]].apply(
        count_dayofweek, axis=1
    )
    dow_count = dow_count / max(dow_count) * norm

    # norm time of day
    tod_count = features[[f"day_part{idx}" for idx in range(24)]].apply(
        count_timeofday, axis=1
    )
    tod_count = tod_count / max(tod_count) * norm

    # norm skillz (reverse percentile)
    ranktier = (100 - features["ranktier"]) / max(100 - features["ranktier"]) * norm

    # norm chat excitement
    excite = features["excitement"] / max(features["excitement"]) * norm

    # norm summoner level
    level = features["summonerlevel"] / max(features["summonerlevel"]) * norm

    # popularity
    query = "SELECT display_name, view_count FROM streamer;"
    postgres = tp.Postgres()
    records = np.array(postgres.rawselect(query))
    postgres.close()
    view_count = {record[0]: int(record[1]) for record in records}
    vcdf = pd.DataFrame.from_dict(
        data=view_count, orient="index", columns=["view_count"]
    ).fillna(0)
    vc = np.sqrt(np.sqrt(vcdf["view_count"] / max(vcdf["view_count"]))) * norm

    # combine categories
    pent_input = (
        dow_count.to_frame(name="dow_count")
        .join(tod_count.to_frame(name="tod_count"))
        .join(ranktier.to_frame(name="ranktier"))
        .join(excite.to_frame(name="excite"))
        .join(level.to_frame(name="level"))
        .join(vc.to_frame(name="popularity"))
    )

    pent_input.apply(plot_pentagram, axis=1)


if __name__ == "__main__":
    main()
