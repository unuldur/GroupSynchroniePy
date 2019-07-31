import csv

import pandas as pd

from read_features import *
from synchronie.synchronie_calcul import *
from utils import *
import numpy as np
import statsmodels.formula.api as smf
import pandas as pandas


def question_to_cohesion(vals, weighs=None, rng=None):
    if weighs is not None:
        new_val = list()
        for val in vals:
            su = 0
            if rng is None:
                for i, v in enumerate(val):
                    su += v * weighs[i]
            else:
                for i in range(rng[0], rng[1]):
                    su += val[i - 1] * weighs[i - rng[0]]
            new_val.append(su)
    return [mean(val) for val in vals] if weighs is None else new_val


def print_cohesion():
    print("Start")
    questions = [str(i) for i in range(1, 11)]
    segments = ["A", "B", "C", "D", "E"]
    cohesion_questions_values = [[list() for _ in range(len(questions))] for _ in range(len(groups) * len(segments))]
    with open(cohesion_file, 'r') as csvfile:
        csv_dict = csv.DictReader(csvfile, delimiter=',')
        for r in csv_dict:
            if r["Group"] not in groups:
                continue
            j = groups.index(r['Group'])
            for i, segment in enumerate(segments):
                index = i + j * len(segments)
                for question in questions:
                    cohesion_questions_values[index][int(question) - 1].append(int(r[question + segment]))
    del cohesion_questions_values[end_groups:]
    del cohesion_questions_values[:start_groups]
    time = [i * 120 + 120 + 60 for i in range(len(cohesion_questions_values))]
    mean_cohesion_value = [[mean(question) for question in window] for window in cohesion_questions_values]
    cohesion_values = question_to_cohesion(mean_cohesion_value, None)
    plt.plot(time, cohesion_values)


def correlation_synchronie_cohesion(filenames, offset, start_groups, end_groups, groups, dec_correlattion, plot=True):
    # get cohesion values
    print("Start")
    start_question = 1
    end_question = 11
    questions = [str(i) for i in range(start_question, end_question)]
    segments = ["A", "B", "C", "D", "E"]
    cohesion_questions_values = [[list() for _ in range(len(questions))] for _ in range(len(groups) * len(segments))]
    with open(cohesion_file, 'r') as csvfile:
        csv_dict = csv.DictReader(csvfile, delimiter=',')
        for r in csv_dict:
            if r["Group"] not in groups:
                continue
            j = groups.index(r['Group'])
            for i, segment in enumerate(segments):
                index = i + j * len(segments)
                for question in questions:
                    cohesion_questions_values[index][int(question) - start_question].append(int(r[question + segment]))
    del cohesion_questions_values[end_groups:]
    del cohesion_questions_values[:start_groups]
    print("Cohesion OK")
    # Get synchrony values
    datas = list()
    for file in filenames:
        with open(file, "r") as csvfile:
            datas.append(convert_dict_reader(csv.DictReader(csvfile, delimiter=';')))
    dec = dec_correlattion
    size = 120
    rags = []
    for i in range(len(cohesion_questions_values)):
        rags.append([i * size + dec, i * size + size + dec])
    synchronie_results = list()
    time = list()
    rng_tot = calculate_rng_peoples(datas, offset)
    for k, rag in enumerate(rags):
        print(k)
        times = data_to_times(datas, offset, rag)
        synchronie_results.append(synchronie_with_features(times, datas, 10, rag, offset, rng_tot, True))
        time.append((rag[0] + rag[1]) / 2)
    weights = None
    mean_cohesion_value = [[mean(question) for question in window] for window in cohesion_questions_values]
    cohesion_values = question_to_cohesion(mean_cohesion_value, weights)
    print(mean_cohesion_value)
    if plot:
        print_correlation(synchronie_results, cohesion_values)
    return synchronie_results, cohesion_values, mean_cohesion_value


def print_correlation(synchronies, cohesions, s=session):
    corr = np.corrcoef(synchronies, cohesions)[0, 1]
    print("Correlation = " + str(corr))
    plt.plot(synchronies, cohesions, 'o')
    plt.title("Session " + str(s) + ", corrélation=" + str(corr))
    plt.xlabel('synchronie')
    plt.ylabel('cohesion')
    plt.show()


def mixtlelm_cohesion(filenames, offset, start_groups, end_groups, groups, dec_correlattion, sessions=range(1, 5)):
    data = list()
    synchronies = list()
    cohesions = list()
    for session in sessions:
        if session == 1:
            filenames = ["../data/session1/yellow.csv", "../data/session1/blue.csv", "../data/session1/red.csv"]
            offset = [0, 0, 224]
            start_groups = 0
            end_groups = -3
            groups = ["1", "2", "3"]
            dec_correlattion = 8
        if session == 2:
            filenames = ["../data/session2/yellow.csv", "../data/session2/blue.csv", "../data/session2/red.csv"]
            offset = [0, 0, 185]
            start_groups = 2
            end_groups = -3
            groups = ["3", "4", "5"]
            dec_correlattion = 2
        if session == 3:
            filenames = ["../data/session3/yellow.csv", "../data/session3/blue2.csv", "../data/session3/red.csv"]
            offset = [0, 0, 110]
            start_groups = 2
            end_groups = -4
            groups = ["5", "6", "7"]
            dec_correlattion = 0
        if session == 4:
            filenames = ["../data/session4Clean/yellow.csv", "../data/session4Clean/blue.csv",
                         "../data/session4Clean/red.csv"]
            offset = [0, 5, 128]
            start_groups = 1
            end_groups = -4
            groups = ["7", "8", "9"]
            dec_correlattion = 0
        synchronie, _, mean_cohesion = correlation_synchronie_cohesion(filenames, offset, start_groups, end_groups,
                                                                       groups,
                                                                       dec_correlattion, plot=False)
        synchronies.append(synchronie)
        cohesions.append(mean_cohesion)
        for i, (syn, cohesion) in enumerate(zip(synchronie, mean_cohesion)):
            val = {'synchronie': syn, 'session': session}
            for j, coh in enumerate(cohesion):
                val['Q' + str(j + 1)] = coh
            data.append(pandas.Series(val))
    data = pandas.DataFrame(data)

    for questions_range in [[1, 11]]:
        questions = ["Q" + str(i) for i in range(questions_range[0], questions_range[1])]
        md = smf.mixedlm("synchronie ~ " + " + ".join(questions), data, groups='session')
        mdf = md.fit()
        print(mdf.summary())
        synchronie_res = list()
        cohesion_res = list()
        for i, (coh, syn) in enumerate(zip(cohesions, synchronies)):
            synchronie_res += syn
            cohesion_res += question_to_cohesion(coh, [mdf.params[q] for q in questions], questions_range)
            print_correlation(syn, question_to_cohesion(coh, [mdf.params[q] for q in questions], questions_range),
                              sessions[i])
        print_correlation(synchronie_res, cohesion_res, sessions)


def multiple_correlation():
    cohesions = list()
    synchronie = list()
    for session in range(1, 5):
        if session == 1:
            filenames = ["../data/session1/yellow.csv", "../data/session1/blue.csv", "../data/session1/red.csv"]
            offset = [0, 0, 224]
            start_groups = 0
            end_groups = -3
            groups = ["1", "2", "3"]
            dec_correlattion = 8
        if session == 2:
            filenames = ["../data/session2/yellow.csv", "../data/session2/blue.csv", "../data/session2/red.csv"]
            offset = [0, 0, 185]
            start_groups = 2
            end_groups = -3
            groups = ["3", "4", "5"]
            dec_correlattion = 2
        if session == 3:
            filenames = ["../data/session3/yellow.csv", "../data/session3/blue2.csv", "../data/session3/red.csv"]
            offset = [0, 0, 110]
            start_groups = 2
            end_groups = -4
            groups = ["5", "6", "7"]
            dec_correlattion = 0
        if session == 4:
            filenames = ["../data/session4Clean/yellow.csv", "../data/session4Clean/blue.csv",
                         "../data/session4Clean/red.csv"]
            offset = [0, 5, 128]
            start_groups = 1
            end_groups = -4
            groups = ["7", "8", "9"]
            dec_correlattion = 0
        s, c, _ = correlation_synchronie_cohesion(filenames, offset, start_groups, end_groups, groups, dec_correlattion,
                                                  plot=False)
        cohesions += c
        synchronie += s

        plt.plot(s, c, 'o', label=str(session))
    print("Correlation = " + str(np.corrcoef(synchronie, cohesions)[0, 1]))
    # plt.plot(time, cohesion_values)
    plt.legend()
    plt.xlabel('synchronie')
    plt.ylabel('cohesion')
    plt.show()


def histogramme_cohesion():
    start_question = 1
    end_question = 11
    questions = [str(i) for i in range(start_question, end_question)]
    segments = ["A", "B", "C", "D", "E"]
    groups = [str(i) for i in range(1, 10)]
    cohesion_questions_values = [[list() for _ in range(len(questions))] for _ in range(len(groups) * len(segments))]
    with open(cohesion_file, 'r') as csvfile:
        csv_dict = csv.DictReader(csvfile, delimiter=',')
        for r in csv_dict:
            if r["Group"] not in groups:
                continue
            j = groups.index(r['Group'])
            for i, segment in enumerate(segments):
                index = i + j * len(segments)
                for question in questions:
                    cohesion_questions_values[index][int(question) - start_question].append(int(r[question + segment]))
    del cohesion_questions_values[-4:]
    mean_cohesion_value = [[mean(question) for question in window] for window in cohesion_questions_values]
    questions = list()
    for cohesions in mean_cohesion_value:
        val = {}
        for i, c in enumerate(cohesions):
            val['Q' + str(i)] = c
        questions.append(val)
    res = question_to_cohesion(mean_cohesion_value)
    plt.hist(res, range=(1, 7), bins=7)
    plt.title("Average")
    plt.xlabel("cohesion")
    plt.show()
    weight = [0.0913, 0.0722, 0.1027, 0.1483, 0.0989, 0.2053, 0.0456, 0.1065, 0, 0.1293]
    res = question_to_cohesion(mean_cohesion_value, weighs=weight)
    plt.hist(res, range=(1, 7), bins=7)
    plt.title("Cohesion with weight")
    plt.xlabel("cohesion")
    plt.show()
    print("end")


if __name__ == "__main__":
    if session == 1:
        filenames = ["../data/session1/yellow.csv", "../data/session1/blue.csv", "../data/session1/red.csv"]
        offset = [0, 0, 224]
        start_groups = 0
        end_groups = -3
        groups = ["1", "2", "3"]
        dec_correlattion = 8
    if session == 2:
        filenames = ["../data/session2/yellow.csv", "../data/session2/blue.csv", "../data/session2/red.csv"]
        offset = [0, 0, 185]
        start_groups = 2
        end_groups = -3
        groups = ["3", "4", "5"]
        dec_correlattion = 2
    if session == 3:
        filenames = ["../data/session3/yellow.csv", "../data/session3/blue2.csv", "../data/session3/red.csv"]
        offset = [0, 0, 110]
        start_groups = 2
        end_groups = -4
        groups = ["5", "6", "7"]
        dec_correlattion = 0
    if session == 4:
        filenames = ["../data/session4Clean/yellow.csv", "../data/session4Clean/blue.csv",
                     "../data/session4Clean/red.csv"]
        offset = [0, 5, 128]
        start_groups = 1
        end_groups = -4
        groups = ["7", "8", "9"]
        dec_correlattion = 0
    correlation_synchronie_cohesion(filenames, offset, start_groups, end_groups, groups, dec_correlattion)
    # multiple_correlation()
    # mixtlelm_cohesion(filenames, offset, start_groups, end_groups, groups, dec_correlattion, sessions=[1, 2, 3, 4])
    # histogramme_cohesion()
