from matplotlib.widgets import Slider, RadioButtons, CheckButtons

from utils import *
from graphviz import Digraph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def who_speak_next_v3(times, names):
    """
    Créer une liste de discussion avec la personne en main speaker et son temps de parole
    :param times: la liste des temps des frames de discution de chacune des personnes présente dans la conversation
    :param names: le nom de chacune des personnes dans la conversation
    :return: la liste des gens qui parle
    """
    times = list(filter(lambda x: len(x) > 0, times))
    indexs = [0 for _ in range(len(times))]
    dot = Digraph()
    min_start = min([times[i][0][0] for i in range(len(times))])
    indexs_speaking = [(i, indexs[i]) for i, x in enumerate(times) if x[0][0] == min_start][0]
    times_speak = list()
    sizes = list()
    for time in times:
        for t in time:
            sizes.append(t[1] - t[0])
    max_size = max(sizes)
    while not all([indexs[i] == (len(times[i]) - 1) for i in range(len(times))]):
        index, place = indexs_speaking
        val_suite = list()
        for t in range(len(times)):
            while times[t][indexs[t]][1] <= times[index][place][1]:
                if indexs[t] < len(times[t]) - 1:
                    indexs[t] += 1
                else:
                    break
            if indexs[t] >= len(times[t]) - 1:
                val_suite.append(999999999)
            else:
                val_suite.append(abs(times[t][indexs[t]][0] - times[index][place][1]) / 4 - (times[t][indexs[t]][1] - times[t][indexs[t]][0]) / max_size)
        val_min = min(val_suite)
        new_indexs = [(i, indexs[i]) for i, x in enumerate(val_suite) if x == val_min][0]
        if len(times_speak) == 0 or times_speak[-1][0] != new_indexs[0]:
            times_speak.append((new_indexs[0], times[new_indexs[0]][new_indexs[1]]))
        else:
            new_end = (new_indexs[0], (times_speak[-1][1][0], times[new_indexs[0]][new_indexs[1]][1]))
            del times_speak[-1]
            times_speak.append(new_end)
        indexs_speaking = new_indexs
    for j in range(1, len(times_speak) - 1):
        i, p = times_speak[j - 1]
        node_1 = names[i] + " " + second_to_min_str(p[0], separator='.') + "," + second_to_min_str(p[1], separator='.')
        i, p = times_speak[j]
        node_2 = names[i] + " " + second_to_min_str(p[0], separator='.') + "," + second_to_min_str(p[1], separator='.')
        dot.edge(node_1, node_2)
    dot.render()
    return times_speak[:-1]


def print_time_speak(time_speak, times):

    linewidth = 5
    for i, time in enumerate(times):
        for t in time:
            plt.plot(t, [i, i], color='c', linewidth=linewidth)
    last_i = -1
    last_time = -1
    for i, time in time_speak:
        if last_i != -1:
            plt.plot([last_time, time[0]], [last_i, i], color='y', linewidth=2)
        plt.plot(time, [i, i], color='y', linewidth=linewidth)
        last_i = i
        last_time = time[1]
    plt.show()


def who_speak_next_matrix(time_speak, size, rng=None, pourcentage=False):
    """
    Créer la who speak next matrix et la retourne
    :param time_speak: la liste des persoones qui parles
    :param size: le nombre de personne dans la conversation
    :param rng: la range de données voulues
    :param pourcentage: obtenir les données en pourcentage ou pas
    :return: la who speak next matrix
    """
    who_speak = np.zeros((size, size))
    last = -1
    for i, time in time_speak:
        if rng is not None and (time[1] < rng[0] or time[0] > rng[1]):
            continue
        if last != -1:
            who_speak[i, last] += 1
        last = i
    if pourcentage:
        who_speak /= np.sum(who_speak)
    return who_speak


def print_who_speak_next(time_speak, names):
    """
    Affiche un graphique permettant de changer les parametre de who speak next et d'observer le parametre
    :param time_speak: la liste des persoones qui parles
    :param names: le nom des personnes dans la conversation
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(right=0.8, bottom=0.5)

    val = 0
    wsms = list()
    xvalues = list()
    while val < time_speak[-1][1][1]:
        wsms.append(who_speak_next_matrix(time_speak, len(names), (val, val + 200)))
        xvalues.append(val)
        val += 200
    yvalues = list()
    for wsm in wsms:
        yvalues.append(wsm[0, 1])

    ax.bar(xvalues, yvalues, 180)



    axcolor = 'lightgoldenrodyellow'
    axframe = plt.axes([0.2 , 0.3, 0.5, 0.05], facecolor=axcolor)
    axminrng = plt.axes([0.2, 0.2, 0.5, 0.05], facecolor=axcolor)
    axmaxrng = plt.axes([0.2, 0.1, 0.5, 0.05], facecolor=axcolor)
    axlabel1 = plt.axes([0.85, 0.7, 0.1, 0.15])
    axLabel2 = plt.axes([0.85, 0.5, 0.1, 0.15])
    axcheck = plt.axes([0.85, 0.3, 0.15, 0.15])

    sframe = Slider(axframe, 'size window', 0.1, 500, valinit=200, valstep=0.1)
    sminrng = Slider(axminrng, 'Minimum', 0, time_speak[-1][1][1], valinit=0, valstep=1)
    smaxrng = Slider(axmaxrng, 'Maximum', 0, time_speak[-1][1][1], valinit=time_speak[-1][1][1], valstep=1)
    rad1 = RadioButtons(axlabel1, names)
    rad2 = RadioButtons(axLabel2, names, active=1)
    pourcent = CheckButtons(axcheck, ['Pourcentage'])

    def updateWsms(label):
        val = sminrng.val
        wsms.clear()
        xvalues.clear()
        while val < smaxrng.val:
            wsms.append(who_speak_next_matrix(time_speak, len(names), (val, val + sframe.val), pourcentage=pourcent.lines[0][0].get_visible()))
            xvalues.append(val)
            val += sframe.val
        update(rad1.value_selected, rad2.value_selected)

    def updateLabel1(label):
        update(label, rad2.value_selected)

    def updateLabel2(label):
        update(rad1.value_selected, label)

    def update(label1, label2):
        yvalues = list()
        for wsm in wsms:
            yvalues.append(wsm[names.index(label1), names.index(label2)])
        ax.clear()
        ax.bar(xvalues, yvalues, sframe.val - 2/10 * sframe.val)
        fig.canvas.draw_idle()

    sframe.on_changed(updateWsms)
    sminrng.on_changed(updateWsms)
    smaxrng.on_changed(updateWsms)
    rad1.on_clicked(updateLabel1)
    rad2.on_clicked(updateLabel2)
    pourcent.on_clicked(updateWsms)

    plt.show()


def print_who_speak_next_v2(time_speak, names):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.4)
    matrix = who_speak_next_matrix(time_speak, len(names), (0, time_speak[-1][1][1]))
    g = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    labels = dict()
    for i, name in enumerate(names):
        labels[i] = name
    red_edges = [(names[i - 1], names[i]) for i in range(1, len(names))]
    red_edges.append((names[-1], names[0]))
    blue_edges = [(y, x) for x, y in red_edges]
    nx.relabel_nodes(g, labels, False)

    edges_label_red = nx.get_edge_attributes(g, 'weight')
    for x, y in blue_edges:
        edges_label_red[(x, y)] = ''
    edges_label_blue = nx.get_edge_attributes(g, 'weight')
    for x, y in red_edges:
        edges_label_blue[(x, y)] = ''
    pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, ax=ax, pos=pos)
    nx.draw_networkx_labels(g, pos)
    nx.draw_networkx_edges(g, pos=pos, ax=ax, connectionstyle='arc3,rad=0.1', edgelist=red_edges, edge_color='r')
    nx.draw_networkx_edges(g, pos=pos, ax=ax, connectionstyle='arc3,rad=0.1', edgelist=blue_edges, edge_color='b')
    nx.draw_networkx_edge_labels(g, pos=pos, ax=ax, edge_labels=edges_label_red, label_pos=0.3, rotate=False, font_color='r')
    nx.draw_networkx_edge_labels(g, pos=pos, ax=ax, edge_labels=edges_label_blue, label_pos=0.3, rotate=False, font_color='b')

    ax.text(0.01, 0.9, 'Window selected: (0,' + str(int(time_speak[-1][1][1])) + ')', ha='left', va='center', transform=ax.transAxes)

    axcolor = 'lightgoldenrodyellow'
    axwindows = plt.axes([0.2, 0.3, 0.5, 0.05], facecolor=axcolor)
    wichframe = plt.axes([0.2, 0.2, 0.5, 0.05], facecolor=axcolor)
    axcheck = plt.axes([0.2, 0.05, 0.2, 0.10])
    sframe = Slider(axwindows, 'size window', 0.1, time_speak[-1][1][1], valinit=time_speak[-1][1][1], valstep=0.1)
    stime = Slider(wichframe, 'window time', 0, time_speak[-1][1][1], valinit=time_speak[-1][1][1]/2, valstep=1)
    pourcent = CheckButtons(axcheck, ['Pourcentage'])

    def update(val):
        matrix = who_speak_next_matrix(time_speak, len(names), (stime.val - sframe.val/2, stime.val + sframe.val/2),
                                       pourcentage=pourcent.lines[0][0].get_visible())
        g = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        nx.relabel_nodes(g, labels, False)
        ax.clear()
        nx.draw_networkx_nodes(g, ax=ax, pos=pos)
        nx.draw_networkx_labels(g, pos, ax=ax)
        nx.draw_networkx_edges(g, pos=pos, ax=ax, connectionstyle='arc3,rad=0.1', edgelist=red_edges, edge_color='r')
        nx.draw_networkx_edges(g, pos=pos, ax=ax, connectionstyle='arc3,rad=0.1', edgelist=blue_edges, edge_color='b')
        edges_label = nx.get_edge_attributes(g, 'weight')
        for k, v in edges_label.items():
            edges_label[k] = "{:.2f}".format(v)

        edges_label_red = edges_label.copy()
        for x, y in blue_edges:
            edges_label_red[(x, y)] = ''
        edges_label_blue = edges_label.copy()
        for x, y in red_edges:
            edges_label_blue[(x, y)] = ''

        nx.draw_networkx_edge_labels(g, pos=pos, ax=ax, edge_labels=edges_label_red, label_pos=0.3, rotate=False,
                                     font_color='r')
        nx.draw_networkx_edge_labels(g, pos=pos, ax=ax, edge_labels=edges_label_blue, label_pos=0.3, rotate=False,
                                     font_color='b')
        ax.text(0.01, 0.9, 'Window selected: (' + str(int(stime.val - sframe.val/2)) + ',' + str(int(stime.val + sframe.val/2)) +
                ')', ha='left', va='center', transform=ax.transAxes)

    sframe.on_changed(update)
    stime.on_changed(update)
    pourcent.on_clicked(update)
    plt.show()








