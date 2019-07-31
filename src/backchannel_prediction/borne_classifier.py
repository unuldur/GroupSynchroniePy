from utils import *


def borne_solution(features, datas, verbose=False):
    """
        Sort les bornes pour avoir accer qu'aux backchannels
        edit: trés mauvais dès qu'on change de personne.

    :param verbose: Affiche les valeurs
    :param features: la liste des features disponible
    :param datas: les données de la piste audio sous la forme donnée par la méthode read_feature
    :return: la liste des données réduite, les bornes, le nombre de backchannel, le nombre de non-backchannel qui ont été classé en
    backchannel
    """
    feature_val = dict()
    min_max_features = dict()
    for f in features:
        feature_val[f] = list()
    for (vals, back) in datas:
        for i, val in enumerate(vals):
            if back:
                feature_val[features[i]].append(val)
    for f in features:
        min_max_features[f] = (min(feature_val[f]), max(feature_val[f]))
    ok = 0
    ok_but_fail = 0
    last = list()
    for j, (vals, back) in enumerate(datas):
        validate = True
        for i, val in enumerate(vals):
            if min_max_features[features[i]][0] > val or val > min_max_features[features[i]][1]:
                validate = False
                break
        if validate:
            last.append((vals, back))
            if back:
                ok += 1
            else:
                ok_but_fail += 1
    if verbose:
        print("ok: " + str(ok))
        print("not ok: " + str(ok_but_fail))
        print("total : " + str(len(datas)))
    return last, min_max_features, ok, ok_but_fail


def upgrade_solution(features, last, min_max):
    """
    Enleve un backchannel et réduit les bornes possible.
    :param features: la liste des features disponible
    :param last: les données déjà réduite
    :param min_max: les bornes existantes
    :return: la liste des données réduite, le nombre de backchannel, le nombre de non-backchannel qui ont été classé
    en backchannel
    """
    extrem = dict()
    m = (0, 0)
    for j, (vals, back) in enumerate(last):
        for i, val in enumerate(vals):
            if val == min_max[features[i]][0] or val == min_max[features[i]][1]:
                if j not in extrem.keys():
                    extrem[j] = 0
                extrem[j] += 1
                if extrem[j] > m[1]:
                    m = (j, extrem[j])
    del last[m[0]]
    return borne_solution(features, last)


def upgrade_while_not_good(features, last, min_max, ok, not_ok):
    """
    Enleve des backchannels tant que il est interressant d'en enlever un.
    :param features: la liste des features disponible
    :param last: les données déjà réduite
    :param min_max: les bornes existantes
    :param ok: le nombre de backchannel bien classé
    :param not_ok: le nombre de non-backchannel qui ont été classé en backchannel
    :return: les bornes améliorer
    """
    new_last, new_min_max, new_ok, new_not_good = upgrade_solution(features, last.copy(), min_max)
    while abs(new_ok - ok) < abs(not_ok - new_not_good):
        last = new_last
        min_max = new_min_max
        ok = new_ok
        not_ok = new_not_good
        new_last, new_min_max, new_ok, new_not_good = upgrade_solution(features, last.copy(), min_max)
    return min_max


def classifie_backchannel(datas, features, min_max_features, times):
    """
    Classifie les backchannels avec les bornes.
    :param datas: les données de la piste audio sous la forme donnée par la méthode read_feature
    :param features: la liste des features disponible
    :param min_max_features: les bornes
    :param times: la liste des temps de début et de fin de chaque donnée présente dans data
    :return: les datas des backchannels, les temps des backchannels trouvé.
    """
    backchannels = list()
    datas_find = list()
    for i, (vals, back) in enumerate(datas):
        validate = True
        for j, val in enumerate(vals):
            if min_max_features[features[j]][0] > val or val > min_max_features[features[j]][1]:
                validate = False
                break
        if validate:
            backchannels.append(times[i])
            datas_find.append((vals, back))
    return datas_find, backchannels


def main_borne(datas_train, features, data_test, times, verbose=True):
    """
    Effectue toute les manipulations nécessaire pour classer par borne. Et affiche les résultat.
    :param datas_train: les données pour 'entrainer' les bornes
    :param features: les features des données
    :param data_test: les données à retourner pour les résultats
    :param times:  les temps des données test
    :param verbose: Affiche les résultat ou pas
    :return: les temps des résultat
    """
    better, min_max, ok, not_ok = borne_solution(features, datas_train)
    min_max = upgrade_while_not_good(features, better, min_max, ok, not_ok)
    datas, res = classifie_backchannel(data_test, features, min_max, times)
    if not verbose:
        return datas, res
    for (deb, end) in res:
        print(second_to_min_str(deb) + " " + second_to_min_str(end))
    print("Nombre de résultats : " + str(len(res)))
    return datas, res

