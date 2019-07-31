import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import *


def find_max_min_index_feature(datas, features, feature):
    """
    Cherche le max le min et l'index de feature d'une feature dans les données
    :param datas: les données
    :param features: l'ensemble des features
    :param feature: la feature à trouver
    :return: None si la feature n'existe pas sinon un triplet (index, min, max)
    """
    if feature in features:
        index_f = features.index(feature)
        f_list = [vals[index_f] for vals, _ in datas]
        return index_f, min(f_list), max(f_list)
    return None


def neural_solution(features, datas, datas_test, time_test):
    """
        Tentative de solution utilisant un réseau de neurone mais pas assez de données et résultat peu interressant.
        :param features: la liste des features disponibles
        :param datas: les données de la piste audio sous la forme donnée par la méthode read_feature
        :param datas_test: les données de test
        :param time_test: les temps de test
    """
    model = keras.Sequential([
        keras.layers.Dense(1000, activation=tf.nn.relu, input_shape=(len(features),)),
        keras.layers.Dense(3000, activation=tf.nn.relu),
        keras.layers.Dense(5000, activation=tf.nn.relu),
        keras.layers.Dense(3000, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    result = list()
    datas_for_tf = list()
    datas_try_tf = list()

    len_feature = find_max_min_index_feature(datas, features, "len")
    diff_len = find_max_min_index_feature(datas, features, "difference_len")

    for (vals, back) in datas:
        if len_feature is not None:
            vals[len_feature[0]] = (vals[len_feature[0]] - len_feature[1])/(len_feature[2] - len_feature[1])
        if diff_len is not None:
            vals[diff_len[0]] = (vals[diff_len[0]] - diff_len[1])/(diff_len[2] - diff_len[1])
        datas_for_tf.append(vals)
        result.append(int(back))

    len_feature_try = find_max_min_index_feature(datas_test, features, "len")
    diff_len_try = find_max_min_index_feature(datas_test, features, "difference_len")
    for (vals, _) in datas_test:
        if len_feature_try is not None:
            vals[len_feature_try[0]] = (vals[len_feature_try[0]] - len_feature_try[1])/(len_feature_try[2] - len_feature_try[1])
        if diff_len_try is not None:
            vals[diff_len_try[0]] = (vals[diff_len_try[0]] - diff_len_try[1])/(diff_len_try[2] - diff_len_try[1])
        datas_try_tf.append(vals)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(np.asarray(datas_for_tf), np.asarray(result), epochs=200)
    model.save('try_yellow')
    res = model.predict(np.asarray(datas_try_tf))
    end = list()
    for i, val in enumerate(res):
        if np.argmax(val):
            print(str(val) + " : " + second_to_min_str(time_test[i][0]) + ' ' + second_to_min_str(time_test[i][1]))
            end.append(time_test[i])
    return end


#model = keras.models.load_model('try_yellow')


def load_neural(path, datas_test, time_test, features):
    """
    Charge et utilise un réseau de neurone pour classifié les backchannels
    :param path: le chemin vers le réseau de neurone sauvegarder
    :param datas_test: les données à classifier
    :param time_test: les temps à classifier
    :param features:  la liste des différentes features
    :return: les données des backchannels trouvé, les temps des backchannels trouvés
    """
    model = keras.models.load_model(path)
    datas_try_tf = list()
    len_feature_try = find_max_min_index_feature(datas_test, features, "len")
    diff_len_try = find_max_min_index_feature(datas_test, features, "difference_len")
    for (vals, _) in datas_test:
        if len_feature_try is not None and (len_feature_try[2] - len_feature_try[1]) != 0:
            vals[len_feature_try[0]] = (vals[len_feature_try[0]] - len_feature_try[1]) / (
                        len_feature_try[2] - len_feature_try[1])
        if diff_len_try is not None and (diff_len_try[2] - diff_len_try[1]) != 0:
            vals[diff_len_try[0]] = (vals[diff_len_try[0]] - diff_len_try[1]) / (diff_len_try[2] - diff_len_try[1])
        datas_try_tf.append(vals)
    res = model.predict(np.asarray(datas_try_tf))
    end = list()
    back = list()
    for i, val in enumerate(res):
        if np.argmax(val):
            end.append(time_test[i])
            back.append(datas_test[i])
    return end, back


def neural_solution_2(features, datas, times, results, which_type):
    """
        Tentative de solution utilisant un réseau de neurone mais pas assez de données et résultat peu interressant.
        :param features: la liste des features disponibles
        :param datas: les données de la piste audio sous la forme donnée par la méthode read_feature
        :param datas_test: les données de test
        :param time_test: les temps de test
    """
    model = keras.Sequential([
        keras.layers.Dense(1000, activation=tf.nn.relu, input_shape=(len(features) + 4,)),
        keras.layers.Dense(3000, activation=tf.nn.relu),
        keras.layers.Dense(5000, activation=tf.nn.relu),
        keras.layers.Dense(3000, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    result = list()
    datas_for_tf = list()

    len_feature = find_max_min_index_feature(datas, features, "len")
    diff_len = find_max_min_index_feature(datas, features, "difference_len")

    for i, (vals, back) in enumerate(datas):
        if len_feature is not None:
            vals[len_feature[0]] = (vals[len_feature[0]] - len_feature[1])/(len_feature[2] - len_feature[1])
        if diff_len is not None:
            vals[diff_len[0]] = (vals[diff_len[0]] - diff_len[1])/(diff_len[2] - diff_len[1])
        if times[i] in results:
            vals += which_type[results.index(times[i])]
        else:
            vals += [0, 0, 0]
        vals.append(1 if times[i] in results else 0)

        datas_for_tf.append(vals)
        res = 0
        if back and times[i] in results or not back and times[i] not in results:
            res = 1
        result.append(res)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(np.asarray(datas_for_tf), np.asarray(result), epochs=100)
    print(model.evaluate(np.asarray(datas_for_tf), np.asarray(result)))
    model.save('next_step_2')


def load_neural_2(path, datas, times, results, which_type, features):
    """
    Charge et utilise un réseau de neurone pour classifié les backchannels
    :param path: le chemin vers le réseau de neurone sauvegarder
    :param datas: les données à classifier
    :param times: les temps à classifier
    :param features:  la liste des différentes features
    :return: les données des backchannels trouvé, les temps des backchannels trouvés
    """
    model = keras.models.load_model(path)
    datas_try_tf = list()
    len_feature_try = find_max_min_index_feature(datas, features, "len")
    diff_len_try = find_max_min_index_feature(datas, features, "difference_len")
    result = list()
    for i, (vals, back) in enumerate(datas):
        if len_feature_try is not None and (len_feature_try[2] - len_feature_try[1]) != 0:
            vals[len_feature_try[0]] = (vals[len_feature_try[0]] - len_feature_try[1]) / (
                        len_feature_try[2] - len_feature_try[1])
        if diff_len_try is not None and (diff_len_try[2] - diff_len_try[1]) != 0:
            vals[diff_len_try[0]] = (vals[diff_len_try[0]] - diff_len_try[1]) / (diff_len_try[2] - diff_len_try[1])
        datas_try_tf.append(vals)
        if times[i] in results:
            vals += which_type[results.index(times[i])]
        else:
            vals += [0, 0, 0]
        vals.append(1 if times[i] in results else 0)
        res = 0
        if back and times[i] in results or not back and times[i] not in results:
            res = 1
        result.append(res)
    print(model.evaluate(np.asarray(datas_try_tf), np.asarray(result)))
    res = model.predict(np.asarray(datas_try_tf))
    end = list()
    back = list()
    for i, val in enumerate(res):
        if np.argmax(val):
            if times[i] in results:
                end.append(times[i])
                back.append(datas[i])
        else:
            if times[i] not in results:
                end.append(times[i])
                back.append(datas[i])
    return end, back