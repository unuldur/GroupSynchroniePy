from sklearn import svm
from sklearn.linear_model import LogisticRegression
import numpy as np
from joblib import dump, load


def svc_learn(datas, datas_try, time_try):
    result = list()
    datas_for_tf = list()
    datas_try_tf = list()

    for (vals, back) in datas:
        datas_for_tf.append(vals)
        result.append(int(back))

    for (vals, _) in datas_try:
        datas_try_tf.append(vals)

    logreg = LogisticRegression(C=1e10, solver='liblinear', multi_class='ovr', penalty='l2', fit_intercept=True,
                                intercept_scaling=1.5, class_weight='balanced',   verbose=True)
    logreg.fit(np.asarray(datas_for_tf), np.asarray(result))
    dump(logreg, 'test.joblib')

    res = logreg.predict(np.asarray(datas_try_tf))

    end = list()
    for i, r in enumerate(res):
        if r:
            end.append(time_try[i])
    return end


def logreg_load(filename, datas_try, time_try):
    datas = list()
    for (vals, _) in datas_try:
        datas.append(vals)
    logreg = load(filename)
    res = logreg.predict(np.asarray(datas))

    end = list()
    for i, r in enumerate(res):
        if r:
            end.append(time_try[i])
    return end
