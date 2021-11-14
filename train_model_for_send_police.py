import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def train_model_send_police():
    """
    trains the senPolice modele
    """
    path = 'Dataset_crimes.csv'
    data = pd.read_csv(path).dropna()
    date = pd.to_datetime(data['Date'])
    data['day'] = [int(date_time.strftime("%w")) for date_time in date]
    data['time'] = [int(date_time.strftime("%H")) + int(date_time.strftime("%M")) / 60 for date_time in date]
    data['X Coordinate'] = data['X Coordinate'] / 10000
    data['Y Coordinate'] = data['Y Coordinate'] / 10000

    X = data[['X Coordinate', 'Y Coordinate', 'time', 'day']].to_numpy()

    design_matrix_days = [X[X[:, 3] == day, :3] for day in range(7)]
    Kmins_for_day = [KMeans(n_clusters=800, random_state=0).fit(design_matrix_days[day]) for day in range(7)]
    model_for_day = []

    for i in range(7):
        center_for_day = np.concatenate(
            (Kmins_for_day[i].cluster_centers_, np.zeros((Kmins_for_day[i].cluster_centers_.shape[0], 2))), axis=1)

        def my_func(a):
            """
            This function updates the number of points that are around each center
            """
            if (np.linalg.norm(a[0:2] - center_for_day[int(a[3]), 0:2]) * 10000 <= 1640.42) and abs(
                    a[2] - center_for_day[int(a[3]), 2]) <= 0.5:
                center_for_day[int(a[3]), 3] = center_for_day[int(a[3]), 3] + 1
            else:
                center_for_day[int(a[3]), 4] = center_for_day[int(a[3]), 4] + 1

        np.apply_along_axis(my_func, 1,
                            np.concatenate((design_matrix_days[i], Kmins_for_day[i].labels_.reshape(-1, 1)), axis=1))

        model_for_day.append([(element[0] * 10000, element[1] * 10000,
                               time_ret(str(int(element[2])) + ":" + str((element[2] - int(element[2])) * 0.6)[2:4]))
                              for element in center_for_day[center_for_day[:, 3].argsort()[::-1]][0:30, [0, 1, 2]]])

    return model_for_day


def time_ret(time):
    hour = time.split(':')[0]
    if int(hour) > 12:
        hour = str(int(hour) - 12) + ":" + time.split(':')[1] + ":00"
        status = "PM"
    else:
        hour = time + ":00"
        status = "AM"
    return "{} {}".format(hour, status)


class SendPoliceCars:
    """
    Model for the send police car problem
    """

    def __init__(self, model):
        self.__model = model

    def predict(self, dates):
        data_frame_dates = pd.DataFrame(dates)
        dates_date_time = pd.to_datetime(data_frame_dates[0])
        dates_arr = [int(date_time.strftime("%w")) for date_time in dates_date_time]
        ret_pred = [self.__model[day] for day in dates_arr]
        for day in ret_pred:
            i = 0
            for j in range(30):
                day[j] = (day[j][0], day[j][1], dates[i][:10] + " " + day[j][2])
            i += 1
        return ret_pred
