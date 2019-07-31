from backchannel_prediction.plot_features_back import *
import backchannel_prediction.borne_classifier as borne
import backchannel_prediction.neural_classifier as n
import backchannel_prediction.filter_backchannels as f
import backchannel_prediction.svm_classifier as svm
import numpy as np
import random

better = [[0.79987647, 0.85024171, 1., 0.9428066, 0.85138648, 0.624043, 0.78696186, 0.20108513, 0.34255127, 0.05938907],
          [1., 0.78299778, 0.42927491, 0.49750315, 1., 0.96091522, 0.93070511, 1., 1., 1.]]

#best_neural = [1, 0.9284014748471305, 0.7235082824305507, 0.871756534070256, 0.11067959843577836, 1, 0.37495832427651177, 0.38473058027389684, 0.5118249184705974, 0]
#best_neural = [1, 0.20812572980632096, 0.5572515906004044, 0.47034559883200544, 0.4198362589089909, 0.8917727896280557, 0, 1, 1, 0]
best_neural = [-0.13323096301287496, -0.46302519712850954, 1, 0.07739456117115696, -0.41321747463249064, 0.14248021386152745, 0.05122264399430678, 0.3486952902101154, -0.10290224755006835, 0.8880913826052075]


def change_weights(weights):
    new_weights = list()
    for w in weights:
        x = np.random.normal(w, 0.2)
        if x < -1:
            x = -1
        if x > 1:
            x = 1
        if np.random.uniform() >= 0.5:
            new_weights.append(x)
        else:
            new_weights.append(w)
    return new_weights


def train(datas, times, results, error, size, min_max, features, datas_filter, times_filter):
    min_error = [error]
    min_times = results
    best_weights = [np.ones((2, len(datas[0][0])))]
    for k in range(size):
        print(k)
        datas_to_merge = [datas_filter]
        times_to_merge = [times_filter]

        use_weight = random.choice(best_weights).copy()
        for i in range(2):
            new_datas = list()
            use_weight[i] = change_weights(use_weight[i])
            for d, b in datas:
                new_datas.append(([use_weight[i, j] * d[j] for j in range(len(d))], b))
            if i == 0:
                data, time = borne.classifie_backchannel(new_datas, features, min_max, times)
            if i == 1:
                time, data = n.load_neural('try_yellow', new_datas, times, features)
            datas_to_merge.append(data)
            times_to_merge.append(time)
        datas_merge, times_merge, count_merge = f.merge_result(datas_to_merge, times_to_merge)
        times_inter = list()
        for i, c in enumerate(count_merge):
            if c == 3 or c == 2:
                times_inter.append(times_merge[i])
        res = compare(times_inter, datas, times)
        if res <= min(min_error):
            if len(min_error) >= 10:
                index = min_error.index(max(min_error))
                del min_error[index]
                del best_weights[index]
            min_error.append(res)
            best_weights.append(use_weight)
            min_times = times_inter
            print("New better results !!!")
            print(res)
    return min_times, best_weights[min_error.index(min(min_error))]


def train_neural(datas, times, results, error, size, features):
    min_error = [error]
    min_times = results
    best_weights = [np.zeros((len(datas[0][0])))]
    for k in range(size):
        print(k)
        use_weight = random.choice(best_weights).copy()
        new_datas = list()
        use_weight = change_weights(use_weight)
        for d, b in datas:
            new_vals = list()
            for j in range(len(d)):
                val_add = use_weight[j] * d[j]
                if d[j] + val_add < 0:
                    new_vals.append(0)
                    continue
                if d[j] + val_add > 1 and j != 3 and j != 7:
                    new_vals.append(1)
                    continue
                new_vals.append(d[j] + val_add)
            new_datas.append((new_vals, b))
        time, data = n.load_neural('try_blue', new_datas, times, features)
        res = compare(time, datas, times)
        if res <= min(min_error):
            if len(min_error) >= 10:
                index = min_error.index(max(min_error))
                del min_error[index]
                del best_weights[index]
            min_error.append(res)
            best_weights.append(use_weight)
            min_times = time
            print("New better results !!!")
            print(res)
    return min_times, best_weights[min_error.index(min(min_error))]


def calculate_with_weights(datas, times, min_max, features, datas_filter, times_filter, weights=None):
    if weights is None:
        weights = np.asarray(better)
    datas_to_merge = [datas_filter]
    times_to_merge = [times_filter]
    new_datas = list()
    for d, b in datas:
        new_vals = list()
        for j in range(len(d)):
            val_add = best_neural[j] * d[j]
            if d[j] + val_add < 0:
                new_vals.append(0)
                continue
            if d[j] + val_add > 1 and j != 3 and j != 7:
                new_vals.append(1)
                continue
            new_vals.append(d[j] + val_add)
        new_datas.append((new_vals, b))
    time, data = n.load_neural('try_blue', new_datas, times, features)
    datas_to_merge.append(data)
    times_to_merge.append(time)
    print("neural")
    compare(time, datas, times)
    datas_merge, times_merge, count_merge = f.merge_result(datas_to_merge, times_to_merge)
    times_inter = list()
    for i, c in enumerate(count_merge):
        if c == 2:
            times_inter.append(times_merge[i])
    res = compare(times_inter, datas, times)
    print(res)
    return [times_inter] + times_to_merge


def calculate_with_weights_neural(neural_name, datas, times, features, weights=None, verbose=False):
    if weights is None:
        weights = np.asarray(better)
    new_datas = list()
    for d, b in datas:
        new_vals = list()
        for j in range(len(d)):
            val_add = best_neural[j] * d[j]
            if d[j] + val_add < 0:
                new_vals.append(0)
                continue
            if d[j] + val_add > 1 and j != 3 and j != 7:
                new_vals.append(1)
                continue
            new_vals.append(d[j] + val_add)
        new_datas.append((new_vals, b))
    time, data = n.load_neural(neural_name, new_datas, times, features)
    if verbose:
        for a, f in time:
            print(second_to_min_str(a - 224) + " " + second_to_min_str(f - 224))
    compare(time, datas, times)
    res = compare(time, datas, times)
    print(res)
    return time


def calculate_with_weights_svm(svn_name, datas, times, features, weights=None, verbose=False):
    if weights is None:
        weights = np.asarray(better)
    new_datas = list()
    for d, b in datas:
        new_vals = list()
        for j in range(len(d)):
            val_add = best_neural[j] * d[j]
            if d[j] + val_add < 0:
                new_vals.append(0)
                continue
            if d[j] + val_add > 1 and j != 3 and j != 7:
                new_vals.append(1)
                continue
            new_vals.append(d[j] + val_add)
        new_datas.append((new_vals, b))
    time = svm.logreg_load(svn_name, new_datas, times)
    if verbose:
        for a, f in time:
            print(second_to_min_str(a - 224) + " " + second_to_min_str(f - 224))
    compare(time, datas, times)
    res = compare(time, datas, times)
    print(res)
    return time


def train_svm(filename, datas, times, results, error, size):
    min_error = [error]
    min_times = results
    best_weights = [np.zeros((len(datas[0][0])))]
    for k in range(size):
        print(k)
        use_weight = random.choice(best_weights).copy()
        new_datas = list()
        use_weight = change_weights(use_weight)
        for d, b in datas:
            new_vals = list()
            for j in range(len(d)):
                val_add = use_weight[j] * d[j]
                if d[j] + val_add < 0:
                    new_vals.append(0)
                    continue
                if d[j] + val_add > 1 and j != 3 and j != 7:
                    new_vals.append(1)
                    continue
                new_vals.append(d[j] + val_add)
            new_datas.append((new_vals, b))
        time = svm.logreg_load(filename, new_datas, times)
        res = compare(time, datas, times)
        if res <= min(min_error):
            if len(min_error) >= 10:
                index = min_error.index(max(min_error))
                del min_error[index]
                del best_weights[index]
            min_error.append(res)
            best_weights.append(use_weight)
            min_times = time
            print("New better results !!!")
            print(res)
    return min_times, best_weights[min_error.index(min(min_error))]