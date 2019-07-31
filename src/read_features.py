import math
from matplotlib.widgets import Slider

from utils import *
import matplotlib.pyplot as plt


def read_features(datas, features, rng, offset, other_speaker_datas, frame_time):
    """
    Range les features par prise de paroles.
    :param datas: Dictionnaire ayant avec chaque clé la liste de valeur associé pour le speaker principal.
    :param features: La liste des features voulues
    :param rng: la range voulue dans toute les données
    :param offset: l'offset de chacun des autres speakers par rapport au principal
    :param other_speaker_datas: les dictionnnaires des personnes parlant en même temps que la personne principalle
    :return: (datas, temps)
        datas: liste de tuple -> (valeurs des features, valeurs des features des autres, boolean si c'est un backchannels)
        temps: liste de tuple -> (debut de la prise de parole, fin de la prise de parole)
    """
    if "isTurn" not in datas.keys() or "frameTime" not in datas.keys():
        raise ValueError("datas need to have isTurn and frameTime in keys")
    datas_sentence = list()
    times = list()
    for feature in features:
        if feature != "len":
            mx_speak = max(datas[feature])
            mn_speak = min(datas[feature])
            mx_other = list()
            mn_other = list()
            for other in other_speaker_datas:
                mx_other.append(max(other[feature]))
                mn_other.append(min(other[feature]))
        speak = list()
        other_speak = list()
        id = 0
        start = rng[0]
        pause_time = 0
        for i, turn in enumerate(datas["isTurn"]):
            if rng[1] is not None and datas["frameTime"][i] > rng[1]:
                break
            if datas["frameTime"][i] < rng[0]:
                continue
            if not turn and len(speak) > 0:
                """
                if is_pause_time(len(speak) * frame_time, pause_time):
                    pause_time += frame_time
                    continue
                """
                if feature != "len":
                    values = sum(speak) / len(speak)
                else:
                    values = datas["frameTime"][i] - pause_time - start
                other_speaker_values = sum(other_speak) / len(other_speak) if feature != "len" else sum(other_speak)
                if len(datas_sentence) <= id:
                    datas_sentence.append(([], [], False))
                    times.append((start, datas["frameTime"][i] - pause_time))
                datas_sentence[id][0].append(values)
                datas_sentence[id][1].append(other_speaker_values)
                id += 1
                speak = list()
                other_speak = list()
            if turn:
                pause_time = 0
                nb_other = 0
                for other in range(len(other_speaker_datas)):
                    if datas["frameTime"][i] > offset[other]:
                        nb_other += 1
                if feature == "len":
                    speak.append(frame_time)
                    val = list()
                    for other in range(len(other_speaker_datas)):
                        if datas["frameTime"][i] > offset[other]:
                            val.append(
                                frame_time if other_speaker_datas[other]["isTurn"][i - int(offset[other] / frame_time)] else 0)
                    other_speak.append(max(val))
                else:
                    speak.append((datas[feature][i] - mn_speak) / (mx_speak - mn_speak))
                    val = list()
                    for other in range(len(other_speaker_datas)):
                        if datas["frameTime"][i] > offset[other]:
                            val.append((other_speaker_datas[other][feature][i - int(offset[other] / frame_time)] - mn_other[
                                other]) / (mx_other[other] - mn_other[other]))
                    other_speak.append(max(val))
            else:
                start = datas["frameTime"][i]
    return datas_sentence, times



def read_from_file(path):
    """
    Lis les features à partir d'un fichier
    :param path: le chemin du fichier
    :return: (datas, temps)
        datas: liste de tuple -> (valeurs des features,  boolean si c'est un backchannels)
        temps: liste de tuple -> (debut de la prise de parole, fin de la prise de parole)
    """
    datas = list()
    times = list()
    f = open(path, 'r')
    for line in f.readlines():
        vals = line[:-1].split(';')

        val = list()
        for i, v in enumerate(vals[0:-3]):
            val.append(float(v))
        datas.append((val, bool(int(vals[-1]))))
        times.append((str_to_second(vals[-3]), str_to_second(vals[-2])))
    return datas, times


def merge_other(datas, features):
    """
        Récupéres les datas et ajoute les features, qui sont la différence entre la personne principale et les autres.
        :param datas: les données de base
        :param features: les features de base
        :return les nouvelles features, les nouvelles données liées aux features
    """
    new_datas = list()
    for (vals, other, back) in datas:
        new_datas.append((vals.copy(), back))
        for i, val in enumerate(vals):
            new_datas[-1][0].append(abs(other[i] - val))
    new_features = features.copy()
    for f in features:
        new_features.append("difference_" + f)
    return new_features, new_datas


def read_and_add_features(datas, features, rng, offset, other_speaker_datas, frame_time):
    """
    Lis les features et ajoute les features de différences.
    :param datas: Dictionnaire ayant avec chaque clé la liste de valeur associé pour le speaker principal.
    :param features: La liste des features voulues
    :param rng: la range voulue dans toute les données
    :param offset: l'offset de chacun des autres speakers par rapport au principal
    :param other_speaker_datas: les dictionnnaires des personnes parlant en même temps que la personne principalle
    :return: (datas, temps, features)
        datas: liste de tuple -> (valeurs des features, boolean si c'est un backchannels)
        temps: liste de tuple -> (debut de la prise de parole, fin de la prise de parole)
        features: la liste des nouvelle features
    """
    d, t = read_features(datas, features, rng, offset, other_speaker_datas, frame_time)
    f, nd = merge_other(d, features)
    return nd, t, f


def is_pause_time(size_last_speak, size_pause, val_0=0.7, val_1=47.):
    """
    Test si le temps de pause peux être considéré comme telle ou pas
    :param val_1:
    :param val_0:
    :param size_last_speak: temps de la dernière prise de parole
    :param size_pause: temps de la pause actuelle
    :return Boolean
    """
    return True if (math.log((math.exp(1) - math.exp(val_0)) / val_1 * size_last_speak + math.exp(val_0))) >= size_pause else False


def ploting_silence_time_file(path):
    fig, ax = plt.subplots()
    plt.subplots_adjust(right=0.5)
    times = list()
    f = open(path, 'r')
    val_0 = 0.50
    val_1 = 60
    id = list()
    for line in f.readlines():
        vals = line[:-1].split(';')
        id.append(vals[-1])
        vals = vals[:-1]
        times.append((str_to_second(vals[-3]), str_to_second(vals[-2])))

    vals = dict()
    time_k = dict()
    time_pause = list()
    all_time = list()
    size_speak = 0
    for i in range(1, len(times) - 1):
        val = times[i][0] - times[i - 1][1]
        id_a = '0' if id[i] != id[i - 1] else id[i]
        if id_a not in vals.keys():
            vals[id_a] = list()
            time_k[id_a] = list()
        vals[id_a].append(val)
        time_k[id_a].append(times[i - 1][1])
        size_speak += (times[i - 1][1] - times[i - 1][0])
        all_time.append(times[i - 1][1])
        time_pause.append((math.log((math.exp(1) - math.exp(val_0)) / val_1 * size_speak + math.exp(val_0))))
        if not is_pause_time(size_speak, val, val_0=val_0, val_1=val_1):
            size_speak = 0
        else:
            size_speak += val
    for k in vals.keys():
        plt.plot(time_k[k], vals[k], 'o')
    plot_line, = plt.plot(all_time, time_pause, '--')

    axcolor = 'lightgoldenrodyellow'
    axe = plt.axes([0.6, 0.80, 0.3, 0.03], facecolor=axcolor)
    slideval_0 = Slider(axe, "val_0", 0, 1, valinit=val_0, valstep=0.01)
    ax2 = plt.axes([0.6, 0.60, 0.3, 0.03], facecolor=axcolor)
    slideval_1 = Slider(ax2, "val_1", 0.1, 120, valinit=val_1, valstep=0.1)

    def update(val):
        val_0 = slideval_0.val
        val_1 = slideval_1.val
        size_speak = 0
        time_pause = list()
        for i in range(1, len(times) - 1):
            val = times[i][0] - times[i - 1][1]
            size_speak += (times[i - 1][1] - times[i - 1][0])
            time_pause.append((math.log((math.exp(1) - math.exp(val_0)) / val_1 * size_speak + math.exp(val_0))))
            if not is_pause_time(size_speak, val, val_0=val_0, val_1=val_1):
                size_speak = 0
            else:
                size_speak += val
        plot_line.set_ydata(time_pause)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
    slideval_0.on_changed(update)
    slideval_1.on_changed(update)
    plt.show()
