from features import get_feature
from constante import *


def second_to_min_str(scd, separator=':'):
    """
    Convertie un nombre de seconde en string (min 'separator' sec)
    :param scd: le nombre de seconde
    :param separator: le separateur entre les minutes et les secondes
    :return: String
    """
    mn = int(scd / 60)
    sc = scd % 60
    return str(mn) + separator + str(int(sc))


def str_to_second(st, separator='|'):
    """
    Convertie un string contenant des mininutes et des secondes en nombres de secondes
    :param st: les string minute seconde
    :param separator: le sparateur entre les minutes et les secondes
    :return: un float du nombre de seconde
    """
    min_sec = st.split(separator)
    return float(min_sec[0]) * 60 + float(min_sec[1])


def convert_dict_reader(reader):
    data = dict()
    for r in reader:
        for str in reader.fieldnames:
            if str not in data.keys():
                data[str] = list()
            data[str].append(float(r[str]))
    return data


def plot_feature(datas, feature_str, start, end, offset, length_frame=3, plot=True, norm=1, plot_type=0,
                 save_plot=False, directory_plot="./plot.png"):
    """
    Permet de calculer les features et de les afficher au besoin
    :param datas: Dictionnaire contenant une collone "isTurn"
    :param feature_str: Le nom de la feature voulue ou une liste de feature voulue parmit celle ci: pauseTime,
                turnPauseRatio, silence, turnDuration, speakTime,
                overlap
    :param start: debut du calcul dans la piste
    :param end:  fin du calcul dans la piste
    :param offset: list contenant les decalage des differents pistes
    :param length_frame: taille d'une frame pour le calcul d'une valeur pour les features
    :param plot: Afficher à la fin les résultats
    :param norm: 0 -> normalizer la features
                 1 -> normalizer en fonction des autres pistes(ne marche pas sur certaines features)
                 2 -> ne pas normalizer
    :param plot_type: 0 -> histogramme
                      1 -> affichage avec valeurs des différentes pistes les une au dessus des autres
    :param save_plot: Sauvegarder ou pas l'affichage
    :param directory_plot:  Localisation de la sauvegarde './plot.png' de base
    :return: Un object GetFeature ou une liste d'object GetFeature
    """
    feature = None
    if feature_str == "pauseTime":
        feature = get_feature.GetPauseTime(offset, frame_time, length_frame)
    if feature_str == "turnPauseRatio":
        feature = get_feature.GetTurnPauseRatio(offset, frame_time, length_frame)
    if feature_str == "silence":
        feature = get_feature.GetSilence(offset, frame_time, length_frame)
    if feature_str == "turnDuration":
        feature = get_feature.GetTurnDuration(offset, frame_time, length_frame)
    if feature_str == "speakTime":
        feature = get_feature.GetSpeakingTime(offset, frame_time, length_frame)
    if feature_str == "backchannel":
        feature = get_feature.GetBackChannelNumber(offset, frame_time, length_frame, 1)
    if feature_str == "shortLongTurnRatio":
        feature = get_feature.GetLongTurnRatio(offset, frame_time, length_frame, 2)
    if feature_str == "overlap":
        feature = get_feature.GetOverlap(offset, frame_time, length_frame)
    if feature_str == "interruptionOverlap":
        feature = get_feature.GetInterruptionOverlap(offset, frame_time, length_frame)
    if feature_str == "interruption":
        feature = get_feature.GetInterruption(offset, frame_time, length_frame)
    if feature_str == "failedInterruptionOverlap":
        feature = get_feature.GetFailedInterruptionOverlap(offset, frame_time, length_frame)
    if feature_str == "failedInterruption":
        feature = get_feature.GetFailedInterruption(offset, frame_time, length_frame)
    if feature_str == "all":
        features = list()
        is_turns = [data["isTurn"] for data in datas]
        if end is None:
            end = len(is_turns[0]) * frame_time
        for f in feature_list:
            features += plot_feature(datas, f, start, end, offset, plot=False, length_frame=length_frame, norm=norm)
        if plot:
            get_feature.print_multiple(features, start, end, is_turns)
        return features
    if isinstance(feature_str, type([])):
        features = list()
        is_turns = [data["isTurn"] for data in datas]
        if end is None:
            end = len(is_turns[0]) * frame_time
        for f in feature_str:
            features += plot_feature(datas, f, start, end, offset, plot=False, length_frame=length_frame, norm=norm)
        if plot:
            get_feature.print_multiple(features, start, end, is_turns)
        return features
    if feature is None:
        raise ValueError("Feature doesn't exist " + feature_str + " doesn't exist show -h for method type")
    is_turns = [data["isTurn"] for data in datas]
    if end is None:
        end = len(is_turns[0]) * 0.01
    feature.calc(is_turns, start, end)
    if norm == 0:
        feature.normalize()
    elif norm == 1:
        feature.normalize_2(start)
    if plot:
        if plot_type == 0:
            feature.plot_histo(start, labels=filenames, save=save_plot, directory_path=directory_plot)
        else:
            feature.plot(start, labels=filenames, save=save_plot, directory_path=directory_plot)
    return [feature]


def data_to_times(datas, offset, rng):
    times = list()
    for data, off in zip(datas, offset):
        time = read_times(data, [rng[0] - off, rng[1] - off if rng[1] is not None else None])
        for i, (s, e) in enumerate(time):
            time[i] = (s + off, e + off)
        times.append(time)
    return times


def read_times(datas, rng):
    """
    Range les features par prise de paroles.
    :param datas: Dictionnaire ayant avec chaque clé la liste de valeur associé pour le speaker principal.
    :param rng: la range voulue dans toute les données
    :return: temps
        temps: liste de tuple -> (debut de la prise de parole, fin de la prise de parole)
    """
    if "isTurn" not in datas.keys() or "frameTime" not in datas.keys():
        raise ValueError("datas need to have isTurn and frameTime in keys")
    times = list()
    start = rng[0]
    last_turn = 0
    for i, turn in enumerate(datas["isTurn"]):
        if rng[1] is not None and datas["frameTime"][i] > rng[1]:
            break
        if datas["frameTime"][i] < rng[0]:
            continue
        if not turn and last_turn == 1:
            times.append((start, datas["frameTime"][i]))
        elif last_turn == 0:
            start = datas["frameTime"][i]
        last_turn = turn
    return times


def calculate_rng_peoples(datas, offset):
    times = data_to_times(datas, offset, [0, None])
    return [[min([t[0] for t in time]), max(t[1] for t in time)] for time in times]
