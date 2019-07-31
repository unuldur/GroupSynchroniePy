import csv

import mpl_toolkits.axisartist as ax
from numpy import mean

import method
from who_speak_next import *
from read_features import *
import backchannel_prediction.borne_classifier as borne
import backchannel_prediction.plot_features_back as back_plot
import backchannel_prediction.neural_classifier as back_neural
import backchannel_prediction.filter_backchannels as back_filter
import backchannel_prediction.dna_best as back_dna
import backchannel_prediction.svm_classifier as svm
import pandas as pd
import tabulate as tab
from constante import *

use_range = (180, 300)
method_str = "avg"
feature_str = "pauseTime"
parameter = "isTurn"
use_method = False
# "backchannel", "shortLongTurnRatio"
plot = True
save = False
file_out = '../out/badSynchronie.csv'

save_plot = False
directory_plot = "../out/Bad/"

norm = 2
plot_type = 0


def invert_rc(datas):
    rows = list()
    for i in range(len(datas)):
        for j in range(len(datas[i])):
            if len(rows) <= j:
                rows.append(list())
            rows[j].append(datas[i][j])
    return rows


def to_csv(collumn_names, datas, file):
    with open(file, 'w+', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=collumn_names, delimiter=';')
        writer.writeheader()
        rows = list()
        for i in range(len(datas)):
            for j in range(len(datas[i])):
                if len(rows) <= j:
                    rows.append(dict())
                rows[j][collumn_names[i]] = datas[i][j]
        writer.writerows(rows)


def plot_method(datas, method_str):
    method_use = None
    if method_str == "avg":
        method_use = method.AvgMethod()
    if method_str == "min":
        method_use = method.MinMethod()
    if method_str == "max":
        method_use = method.MaxMethod()
    if method_str == "count":
        method_use = method.CategoricalMethod(lambda x: x == 1)
    if method_use is None:
        raise ValueError("Method " + method_str + " doesn't exist show -h for method type")
    for i in range(len(datas)):
        if parameter not in datas[i].keys():
            raise ValueError(
                "Header " + parameter + " doesn't exist in file " + filenames[i] + ". List of headers: " + str(
                    datas[i].keys()))
        data_to_print = method_use.compute(datas[i]["frameTime"], datas[i][parameter], size_frame, length_frame,
                                           use_range,
                                           offset[i])
        if offset[i] < use_range[0]:
            offset_print = use_range[0]
        else:
            offset_print = offset[i]
        plt.plot([(t * length_frame + offset_print) for t in range(len(data_to_print))], data_to_print,
                 label=filenames[i])
    plt.legend()
    plt.show()


def get_data_from_features(features):
    collums_name = list()
    datas = list()
    for f in features:
        if len(f.results) == 1:
            collums_name.append(f.title)
            datas.append(f.results[0])
        else:
            for i in range(len(f.results)):
                collums_name.append(f.title + " " + filenames[i])
                datas.append(f.results[i])
    return datas, collums_name


def main():
    datas = list()
    for file in filenames:
        with open(file, "r") as csvfile:
            datas.append(convert_dict_reader(csv.DictReader(csvfile, delimiter=';')))
    if use_method:
        plot_method(datas, method_str)
    else:
        features = plot_feature(datas, feature_str, use_range[0], use_range[1], offset, norm=norm, plot_type=plot_type)
        if save:
            datas, collums_name = get_data_from_features(features)
            to_csv(collums_name, datas, file_out)


def print_voice_activity():
    datas = list()
    for file in filenames:
        with open(file, "r") as csvfile:
            datas.append(convert_dict_reader(csv.DictReader(csvfile, delimiter=';')))
    j = 0
    names = filenames
    color = ['y', 'b', 'r']
    times = data_to_times(datas, offset, [0, None])

    fig = plt.figure(figsize=(4, 2.5))
    ax1 = fig.add_subplot(ax.Subplot(fig, "111"))
    ax1.axis["left"].set_visible(False)
    ax1.axis["left"].major_ticks.set_visible(False)
    ax1.axis["top"].set_visible(False)
    ax1.axis["right"].set_visible(False)
    for i, time in enumerate(times):
        for k, t in enumerate(time):
            if k > 0:
                plt.plot(t, [j, j], '-', color=color[i], linewidth=9)
            else:
                plt.plot(t, [j, j], '-', color=color[i], linewidth=9, label=names[i])
        plt.text(0, j, names[i])
        j += 0.1
    plt.legend()
    plt.show()


def who_speak_next():
    datas = list()
    for file in filenames:
        with open(file, "r") as csvfile:
            datas.append(convert_dict_reader(csv.DictReader(csvfile, delimiter=';')))
    features = ["pcm_LOGenergy", "pcm_intensity", "pcm_loudness", "F0final", "len"]
    _, times_red, _ = read_and_add_features(datas[2], features, [0, None], [-224, -224], [datas[0], datas[1]],
                                            frame_time)
    _, times_yellow, _ = read_and_add_features(datas[0], features, [0, None], [0, 224], [datas[1], datas[2]],
                                               frame_time)
    _, times_blue, _ = read_and_add_features(datas[1], features, [0, None], [0, 224], [datas[0], datas[2]], frame_time)
    decs = [[0, 0, 224], [0, 100, 224], [0, 100, 100]]
    collums = ['1_red', '1_yellow', '1_blue', '2_red', '2_yellow', '2_blue', '3_red', '3_yellow', '3_blue']
    res = dict()
    for c in collums:
        res[c] = list()
    index = ['mean', 'max', 'min', 'q0', 'q10', 'q20', 'q30', 'q40', 'q50', 'q60', 'q70', 'q80', 'q90']
    name = ['blue', 'yellow', 'red']
    is_turns = [data["isTurn"] for data in datas]
    end = len(is_turns[0]) * frame_time

    for k, dec in enumerate(decs):
        print("==========================================")
        print(dec)
        print("==========================================")
        feature = get_feature.GetOverlap(dec, frame_time, end)
        feature.calc(is_turns, 0, end)
        feature.normalize_2(0)
        print(feature.results)
        new_times_red = [0] * len(times_red)
        new_times_blue = [0] * len(times_blue)
        new_times_yellow = [0] * len(times_yellow)
        for i, (s, e) in enumerate(times_red):
            new_times_red[i] = (s + dec[2], e + dec[2])
        for i, (s, e) in enumerate(times_yellow):
            new_times_yellow[i] = (s + dec[1], e + dec[1])
        for i, (s, e) in enumerate(times_blue):
            new_times_blue[i] = (s + dec[0], e + dec[0])
        time_speak = who_speak_next_v3([new_times_blue, new_times_yellow, new_times_red], ("Bob", "Allan", "Linda"))
        synchronie = [[], [], []]
        end_last = None
        for id, (start, end) in time_speak:
            if end_last is not None:
                lat = start - end_last
                synchronie[id].append(lat)
            end_last = end
        resu = 0
        for j, l in enumerate(synchronie):
            su = 0
            coll = str(k + 1) + '_' + name[j]
            res[coll].append(mean(l))
            res[coll].append(max(l))
            res[coll].append(min(l))
            for i in range(0, 100, 10):
                res[coll].append(np.percentile(l, i))
            for v in l:
                if -0.6 < v <= 1:
                    su += 1
            val = su / len(l)
            print(val)
            resu += val
        print("resultat : " + str(resu / len(synchronie)))
    data_frame = pd.DataFrame(res, index=index, columns=collums)
    print(tab.tabulate(data_frame, headers="keys", tablefmt="fancy_grid"))
    print("end")


def write_in_file(datas, times, name):
    f = open(name, 'w+')
    for data, time in zip(datas, times):
        st = ""
        for d in data[0]:
            st += str(d) + ";"
        st += second_to_min_str(time[0], separator='|') + ";" + second_to_min_str(time[1], separator='|')
        f.write(st + "\n")


def write_datas_csv():
    features = ["pcm_LOGenergy", "pcm_intensity", "pcm_loudness", "F0final", "len"]
    datas = list()
    for file in filenames:
        with open(file, "r") as csvfile:
            datas.append(convert_dict_reader(csv.DictReader(csvfile, delimiter=';')))
    datas_blue, times_blue, _ = read_and_add_features(datas[1], features, [0, 1300], [0, 224],
                                                      [datas[0], datas[2]], frame_time)
    datas_yellow, times_yellow, features = read_and_add_features(datas[0], features, [0, 1300], [0, 224],
                                                                 [datas[1], datas[2]], frame_time)
    write_in_file(datas_blue, times_blue, '../out/bluetmp.csv')
    write_in_file(datas_yellow, times_yellow, '../out/yellowtmp.csv')


def main_backchannel_yellow():
    features = ["pcm_LOGenergy", "pcm_intensity", "pcm_loudness", "F0final", "len"]
    datas = list()
    for file in filenames:
        with open(file, "r") as csvfile:
            datas.append(convert_dict_reader(csv.DictReader(csvfile, delimiter=';')))
    datas_red, times_red, features = read_and_add_features(datas[2], features, [0, 1300], [-224, -224],
                                                           [datas[0], datas[1]], frame_time)

    datas_yellow, times_yellow = read_from_file('../out/yellow_back.csv')
    datas_blue, times_blue = read_from_file('../out/blue_back.csv')
    real_times_red = list()
    for t in times_red:
        real_times_red.append((t[0] + 224, t[1] + 224))
    time_speak = who_speak_next_v3([times_yellow, times_blue, real_times_red], ("Alan", "Bob", "Linda"))
    print(len(datas_yellow))
    _, min_max, _, _ = borne.borne_solution(features, datas_blue)
    data_filter_1, times_filter_1 = borne.classifie_backchannel(datas_yellow, features, min_max, times_yellow)
    back_plot.compare(times_filter_1, datas_yellow, times_yellow)
    datas_filter_2, times_filter_2 = back_filter.filter_in_main(time_speak, datas_yellow, times_yellow, 0)
    back_plot.compare(times_filter_2, datas_yellow, times_yellow)
    times_find_3, datas_find_3 = back_neural.load_neural('try_blue', datas_yellow, times_yellow, features)
    back_plot.compare(times_find_3, datas_yellow, times_yellow)

    datas_merge, times_merge, count_merge = back_filter.merge_result([data_filter_1, datas_filter_2, datas_find_3],
                                                                     [times_filter_1, times_filter_2, times_find_3])
    times_inter = list()
    for i, c in enumerate(count_merge):
        if c >= 1:
            times_inter.append(times_merge[i])
    back_plot.compare(times_inter, datas_yellow, times_yellow)
    back_dna.calculate_with_weights_neural('try_blue', datas_yellow, times_yellow, features)
    res = back_dna.calculate_with_weights(datas_yellow, times_yellow, min_max, features, datas_filter_2, times_filter_2)

    back_plot.plot_results_v2(res + [[]], datas_yellow, times_yellow,
                              [times_blue, real_times_red], ["inter", "filtre", "neural", ""])


def main_backchannel_blue():
    features = ["pcm_LOGenergy", "pcm_intensity", "pcm_loudness", "F0final", "len"]
    datas = list()
    for file in filenames:
        with open(file, "r") as csvfile:
            datas.append(convert_dict_reader(csv.DictReader(csvfile, delimiter=';')))
    datas_red, times_red, features = read_and_add_features(datas[2], features, [0, 1300], [-224, -224],
                                                           [datas[0], datas[1]], frame_time)

    datas_yellow, times_yellow = read_from_file('../out/yellow_back.csv')
    datas_blue, times_blue = read_from_file('../out/blue_back.csv')
    real_times_red = list()
    for t in times_red:
        real_times_red.append((t[0] + 224, t[1] + 224))
    res = svm.svc_learn(datas_yellow, datas_blue, times_blue)
    error = back_plot.compare(res, datas_blue, times_blue)
    res, weights = back_dna.train_svm('test.joblib', datas_blue, times_blue, res, error, 100000)
    print(len(times_blue))
    f = '['
    for w in weights:
        f += str(w) + ', '
    f = f[:-2]
    print(f)


def print_test():
    ploting_silence_time_file('../out/time_speak_back2.csv')


def print_test2():
    features = ["pcm_LOGenergy", "pcm_intensity", "pcm_loudness", "F0final", "len"]
    datas = list()
    for file in filenames:
        with open(file, "r") as csvfile:
            datas.append(convert_dict_reader(csv.DictReader(csvfile, delimiter=';')))
    datas_red, times_red, features = read_and_add_features(datas[2], features, [0, 1300], [-224, -224],
                                                           [datas[0], datas[1]], frame_time)
    datas_yellow, times_yellow = read_from_file('../out/yellow_back.csv')
    val = back_plot.correl(features, datas_yellow)
    back_plot.print_features(datas_yellow, features)


def write_datas_syncpy():
    dec = [0, 0, 224]
    nb = 3
    false_dec = [[0, 100, 100], [0, 100, 224]]
    for j in range(nb):
        rows = dict()
        rdec = dec
        if j > 0:
            rdec = false_dec[j - 1]
        for i, file in enumerate(filenames):
            with open(file, "r") as csvfile:
                data = csv.DictReader(csvfile, delimiter=';')
                for row in data:
                    time = float(row["frameTime"]) + rdec[i]
                    if time not in rows.keys():
                        rows[time] = dict()
                    rows[time][filenames[i]] = row['isTurn']

        name = 'session1'
        if j > 0:
            name += 'error' + str(j)
        print(name)
        file_to_write = open(name + '.csv', 'w+', newline='')
        writer = csv.writer(file_to_write, delimiter=',')
        writer.writerow(['time (s)'] + filenames)
        for key, value in rows.items():
            row = [key]
            for name in filenames:
                if name in value.keys():
                    row.append(value[name])
                else:
                    row.append(0)
            writer.writerow(row)


if __name__ == "__main__":
    main()
