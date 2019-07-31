def filter_in_main(time_speak, datas, times, id):
    """
    Enlève toute les frames qui sont dans la discution principale pour avoir le plus de backchannels possible
    :param time_speak: le fil de la discution
    :param datas: les données à filter
    :param times: les temps à filter
    :param id: l'id de la personne à filtrer
    :return: les données filtrés, les temps filtrés
    """
    datas_filter = datas.copy()
    times_filter = times.copy()
    datas_not_ok = list()
    times_not_ok = list()
    for i, time in time_speak:
        if i != id:
            continue
        for j, (deb, end) in enumerate(times):
            if time[0] <= deb and time[1] >= end:
                times_not_ok.append(times[j])
                datas_not_ok.append(datas[j])
    for time, data in zip(times_not_ok, datas_not_ok):
        datas_filter.remove(data)
        times_filter.remove(time)
    return datas_filter, times_filter


def merge_result(datas, times):
    """
    merge les différents résultats dans une seule liste
    :param datas: la liste des données à merge
    :param times: la liste des temps à merge
    :return: les données merge, les temps merge, le nombre de fois qu'une données apparrait dans chacun des éléments
    """
    datas_merge = list()
    times_merge = list()
    count_merge = list()
    for i in range(len(datas)):
        for data, time in zip(datas[i], times[i]):
            if time in times_merge:
                count_merge[times_merge.index(time)] += 1
            else:
                datas_merge.append(data)
                times_merge.append(time)
                count_merge.append(1)
    return datas_merge, times_merge, count_merge


def merge_result_2(datas, times):
    """
    merge les différents résultats dans une seule liste
    :param datas: la liste des données à merge
    :param times: la liste des temps à merge
    :return: les données merge, les temps merge, le nombre de fois qu'une données apparrait dans chacun des éléments
    """
    datas_merge = list()
    times_merge = list()
    count_merge = list()
    for i in range(len(datas)):
        for data, time in zip(datas[i], times[i]):
            if time in times_merge:
                count_merge[times_merge.index(time)][i] = 1
            else:
                datas_merge.append(data)
                times_merge.append(time)
                count_merge.append([0 for _ in range(len(datas))])
                count_merge[-1][i] = 1
    return datas_merge, times_merge, count_merge
