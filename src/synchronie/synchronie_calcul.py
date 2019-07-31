from numpy import mean

from who_speak_next import who_speak_next_v3
from utils import *


def turn_taking_method(peoples, names):
    time_speak = who_speak_next_v3(peoples, names)
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
        for v in l:
            if -0.6 < v <= 1:
                su += 1
        if len(l) != 0:
            val = su / len(l)
        else:
            val = 0
        resu += val
    return resu / len(synchronie)


def compare_method(peoples, windows_size, rng_tot_people, rag):
    peoples = list(filter(lambda x: x is not None, [people if off[0] - rag[0] <= 80 and off[1] - rag[1] >= -80 else None for (people, off) in zip(peoples, rng_tot_people)]))
    end_real = max([0 if len(speak) == 0 else speak[-1][1] for speak in peoples])
    start_real = min([0 if len(speak) == 0 else speak[0][0] for speak in peoples])
    lats = [[[[] for _ in range(len(peoples))] for _ in range(len(peoples))] for _ in
            range(int(start_real / windows_size) + 1, int(end_real / windows_size) + 1)]
    for i, speak_list in enumerate(peoples):
        for (start, end) in speak_list:
            if end - start < 1:
                continue
            for j, speak_list2 in enumerate(peoples):
                if len(speak_list2) == 0:
                    continue
                lat = min([abs(start2 - end) for (start2, end2) in speak_list2])
                lats[int(end / windows_size) - int(start_real / windows_size) - 1][i][j].append(lat)
    res = [[] for _ in range(len(peoples))]
    nb_0 = 0
    for time in lats:
        maxs = 0
        p = list()
        for i in time:
            for j, latences in enumerate(i):
                if len(latences) == 0:
                    continue
                s = len(list(filter(lambda x: x <= 1.5, latences))) / len(latences)
                if s >= maxs:
                    if s > maxs:
                        p = list()
                    maxs = s
                    p.append(j)

        if len(p) > 0:
            for pe in p:
                res[pe].append(maxs)
        else:
            nb_0 += 1
    val = mean([mean(r) if len(r) > 0 else 0 for r in res])
    return val


def other_ratio_synchrony_method(peoples, rng_tot_people, rag):
    peoples = list(filter(lambda x: x is not None, [people if off[0] - rag[0] <= 80 and off[1] - rag[1] >= -80 else None for (people, off) in zip(peoples, rng_tot_people)]))
    lats = [[] for _ in range(len(peoples))]
    for i, speak_list in enumerate(peoples):
        for (start, end) in speak_list:
            if end - start < 1:
                continue
            min_index = -1
            min_latence = 0
            for j, speak_list2 in enumerate(peoples):
                if len(speak_list2) == 0:
                    continue
                lat = min([abs(start2 - end) for (start2, end2) in speak_list2])
                if min_index == -1 or lat < min_latence:
                    min_latence = lat
                    min_index = j
            lats[min_index].append(min_latence)
    res = list()
    for lat in lats:
        if len(lat) > 0:
            res.append(len(list(filter(lambda x: x <= 1.5, lat))) / len(lat))
        else:
            res.append(0)
    return mean(res)


def synchronie_with_features(times, datas, windows_size, rag, offset, rng_tot_people, verbose=False):
    features_res = plot_feature(datas, ['pauseTime', 'speakTime', 'silence'], rag[0], rag[1], offset, plot=False,
                                length_frame=rag[1] - rag[0] if rag[1] is not None else 3600,
                                norm=0)
    mf = list()
    for i, feature in enumerate(features_res):
        mf.append(max([f[0] for f in feature.results]))
    syn = other_ratio_synchrony_method(times, rng_tot_people, rag)
    if verbose:
        print("features before = " + str(mf))
    mf[1] = abs(mf[1] - 1 / len(times)) * len(times) / (len(times) - 1)
    mf[2] = abs(mf[2] - mf[0])

    weights = [0.25, 0.25, 0.25, 0.25]
    final_val = sum([syn * weights[0]] + [(1 - val) * weights[i + 1] if i != -1 else val for i, val in enumerate(mf)])
    if verbose:
        print("synchronie = " + str(syn))
        print("methods = " + str(mf))
        print("final synchronie = " + str(final_val))
        print("=============")
    return final_val
