import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, MultiCursor
import numpy as np
import pandas as pd
from utils import *
from math import *

def print_features(datas_sentence, features):
    """
    Permet d'afficher les différentes features des différents en les associant à un poids.
    :param datas_sentence: les données de la piste audio sous la forme donnée par la méthode read_feature
    :param features: la liste des features disponibles
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(right=0.5)
    ids_back = list()
    backs = list()
    ids_nback = list()
    nback = list()
    weight = [0 for _ in range(len(features))]
    for j, (_, back) in enumerate(datas_sentence):
        if back:
            backs.append(0)
            ids_back.append(j)
        else:
            nback.append(0)
            ids_nback.append(j)
    backchannel_plot, = plt.plot(ids_back, backs, 'o', label="backchannel")
    no_backchannel_plot, = plt.plot(ids_nback, nback, 'o', label="not backchannel")
    plt.legend()
    sliderw = list()
    for i, w in enumerate(weight):
        axcolor = 'lightgoldenrodyellow'
        axe = plt.axes([0.6 , 0.80 - i * 0.05, 0.3, 0.03], facecolor=axcolor)
        sliderw.append(Slider(axe, features[i], 0, 1, valinit=weight[i], valstep=0.01))

    def update(val):
        nw1 = [slide.val for slide in sliderw]
        backs = list()
        nback = list()
        for j, (speak, back) in enumerate(datas_sentence):

            nb = 0
            for i, val in enumerate(speak):
                nb += val * nw1[i]
            if back:
                backs.append(nb)
            else:
                nback.append(nb)
        backchannel_plot.set_ydata(backs)
        no_backchannel_plot.set_ydata(nback)

        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    for slide in sliderw:
        slide.on_changed(update)

    plt.show()


def print_two_parameters(datas_sentence):
    """
        Affiche le premier paramètre de datas_sentence en fonction du deuxième paramètre de datas_sentence
    :param datas_sentence: les données de la piste audio sous la forme donnée par la méthode read_feature
    """
    parameters1 = list()
    parameters2 = list()
    parameters1_back = list()
    parameters2_back = list()
    for speak, back in datas_sentence:
        if back:
            parameters1_back.append(speak[0])
            parameters2_back.append(speak[1])
        else:
            parameters1.append(speak[0])
            parameters2.append(speak[1])
    plt.plot(parameters1_back, parameters2_back, 'o', label="backchannel")
    plt.plot(parameters1, parameters2, 'o', label="no backchannel")
    plt.legend()
    plt.show()


def correl(features, datas):
    """
        Affiche les correlations entre toutes les features.
        :param features: la liste des features
        :param datas: les données de la piste audio sous la forme donnée par la méthode read_feature
    """
    df = pd.DataFrame()
    feature_val = dict()

    for f in features + ["back"]:
        feature_val[f] = list()
    for (vals, back) in datas:
        for i, val in enumerate(vals):
            feature_val[features[i]].append(val)
        feature_val["back"].append(back)
    for f in features + ["back"]:
        df[f] = np.asarray(feature_val[f])
    salv = df.corr()
    return salv


def compare(time_find, datas, times):
    """
    Compare les bachchannels trouvé avec les vrai backchannels
    :param time_find: les temps des backchannels trouvé
    :param datas: les données générales pour comparer
    :param times:  les temps générales pour comparer
    """
    nb_good = 0
    nb_find_ngood = 0
    nb_nfind = 0
    for i, time_borne in enumerate(times):
        if time_borne in time_find:
            if datas[i][1]:
                nb_good += 1
            else:
                nb_find_ngood += 1
        elif datas[i][1]:
            nb_nfind += 1
    print("Backchannels trouvé : " + str(nb_good))
    print("Backchannels pas trouvé : " + str(nb_nfind))
    print("Pas backchannel trouvé : " + str(nb_find_ngood))
    return (nb_nfind + nb_find_ngood) / len(datas)


def plot_results(time_find, datas, times, others_times):
    """
    Compare les bachchannels trouvé avec les vrai backchannels
    :param others_times: le temps des autres
    :param time_find: les temps des backchannels trouvé
    :param datas: les données générales pour comparer
    :param times:  les temps générales pour comparer
    """
    y = 1
    linewidth = 10
    for i, time in enumerate(times):
        color = 'b'
        if time in time_find:
            if datas[i][1]:
                color = 'g'
            else:
                color = 'r'
        elif datas[i][1]:
            color = 'y'
        plt.plot(time, [y, y], color=color, linewidth=linewidth)
    for otimes in others_times:
        y += 1
        for time in otimes:
            plt.plot(time, [y, y], color='c', linewidth=linewidth)
    plt.show()


def plot_results_v2(time_finds, datas, times, others_times, names):
    """
    Compare les bachchannels trouvé avec les vrai backchannels
    :param others_times: le temps des autres
    :param time_finds: les temps des backchannels trouvé
    :param datas: les données générales pour comparer
    :param times:  les temps générales pour comparer
    :param names: Noms des différents graphiques
    """
    y = 1
    linewidth = 10
    max_x = ceil(sqrt(len(time_finds)))
    max_y = floor(sqrt(len(time_finds)))
    plot_x = 0
    plot_y = 0
    fig, axs = plt.subplots(max_x, max_y, sharex=True)
    for j, time_find in enumerate(time_finds):
        for i, time in enumerate(times):
            color = 'b'
            if time in time_find:
                if datas[i][1]:
                    color = 'g'
                else:
                    color = 'r'
            elif datas[i][1]:
                color = 'y'
            axs[plot_x, plot_y].plot(time, [y, y], color=color, linewidth=linewidth)
        for otimes in others_times:
            y += 1
            for time in otimes:
                axs[plot_x, plot_y].plot(time, [y, y], color='c', linewidth=linewidth)
        axs[plot_x, plot_y].set_title(names[j])
        plot_x += 1
        if plot_x >= max_x:
            plot_x = 0
            plot_y += 1
    MultiCursor(fig.canvas, axs.flat, color='r', lw=1, horizOn=True, vertOn=True)
    plt.show()






