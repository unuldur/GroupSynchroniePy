import csv

from read_features import *
from synchronie.random_generator_voice_activity import *
from synchronie.synchronie_calcul import *


def test_size_synchronie():
    datas = list()
    for file in filenames:
        with open(file, "r") as csvfile:
            datas.append(convert_dict_reader(csv.DictReader(csvfile, delimiter=';')))
    rags = []
    for size in range(60, 1350, 60):
        for i in range(30):
            if i * 50 + size + 300 < max_len:
                rags.append([i * 50 + 300, i * 50 + size + 300])
    result = dict()
    print(len(rags))
    rng_tot = calculate_rng_peoples(datas, offset)
    for k, rag in enumerate(rags):
        print(k)
        print(rag[1] - rag[0])
        times = data_to_times(datas, offset, rag)
        key = rag[1] - rag[0]
        if key not in result.keys():
            result[key] = list()
        val = synchronie_with_features(times, datas, 10, rag, offset, rng_tot, verbose=True)
        result[key].append(val)
    print_time = list()
    print_mean = list()
    print_time_mean = list()
    print_mean_mean = list()
    for k, v in result.items():
        print("===" + str(k) + "====")
        print("Min = " + str(min(v)))
        print("Max = " + str(max(v)))
        print("Mean = " + str(mean(v)))
        print_time_mean.append(k)
        print_mean_mean.append(mean(v))
        for val in v:
            print_time.append(k)
            print_mean.append(val)
    plt.plot(print_time, print_mean, 'o')
    plt.plot(print_time_mean, print_mean_mean, color='r')
    plt.xlabel("window size")
    plt.ylabel("synchronie")
    plt.show()


def test_synchronie_features(decalage=True, feature='all', method=max, norm=0):
    datas = list()
    for file in filenames:
        with open(file, "r") as csvfile:
            datas.append(convert_dict_reader(csv.DictReader(csvfile, delimiter=';')))
    features = ["pcm_LOGenergy", "pcm_intensity", "pcm_loudness", "F0final", "len"]
    decs = [[0, 0, 224]]

    if decalage:
        for i in range(1000):
            decs.append([rnd.randint(0, 500), rnd.randint(0, 500), rnd.randint(0, 500)])
        decs = [[0, 0, 224]] + list(filter(lambda x: not (x[0] < 20 and x[1] < 20 and 200 < x[2] < 240), decs))
    else:
        for i in range(1000):
            decs.append(1)

    print(decs)
    print(len(decs))

    res = dict()
    for k, dec in enumerate(decs):
        print(k)
        new_datas = list()
        if isinstance(dec, type([])):
            new_datas = datas
        else:
            for data in datas:
                _, new_data = generate_random_from_existante(data["isTurn"], [300, 1800])
                new_datas.append({"isTurn": new_data})
        features = plot_feature(new_datas, feature, 300, 1800, dec if isinstance(dec, type([])) else offset, plot=False, length_frame=1300, norm=norm)
        for f in features:
            if f.title not in res.keys():
                res[f.title] = list()
            val = method([val[0] for val in f.results])
            res[f.title].append(abs(val - 1/3) * 3/2)

    for f in res.keys():
        q = len(list(filter(lambda x: x <= res[f][0], res[f]))) / len(res[f])
        print(q)
        plt.hist(res[f], np.arange(0, 1, 0.001))
        plt.plot([res[f][0], res[f][0]], [0, 50], color='r')
        plt.title(f)
        plt.xlabel("value")
        plt.ylabel("size")
        plt.show()

    print("end")


def synchronie_during_file(windows_time=120, start=300, jump=50):
    datas = list()
    for file in filenames:
        with open(file, "r") as csvfile:
            datas.append(convert_dict_reader(csv.DictReader(csvfile, delimiter=';')))
    rags = []
    for i in range(30):
        if i * jump + windows_time + start <= max_len:
            rags.append([i * jump + start, i * jump + windows_time + start])
    time = list()
    print(len(rags))
    moy = list()
    rng_tot = calculate_rng_peoples(datas, offset)
    for k, rag in enumerate(rags):
        print(k)
        times = data_to_times(datas, offset, rag)
        moy.append(synchronie_with_features(times, datas, 10, rag, offset, rng_tot, verbose=True))
        time.append((rag[0] + rag[1]) / 2)
    plt.plot(time, moy, label='mean', linewidth=2)
    plt.legend()
    plt.xlabel("Time in sec")
    plt.ylabel("Synchronie value")
    plt.title("Session " + str(session) + ", window size = " + str(windows_time))
    plt.show()


def synchronie_during_file_change_peoples(windows_time=120, peoples=[(1, 2, 0), (0, 1, 2 ,3), (3, 1 ,0), (1, 2), (2, 0)]):
    datas = list()
    for file in filenames:
        with open(file, "r") as csvfile:
            datas.append(convert_dict_reader(csv.DictReader(csvfile, delimiter=';')))
    rags = []
    for i in range(30):
        if i * 50 + windows_time + 300 < 1600:
            rags.append([i * 50 + 300, i * 50 + windows_time + 300])
    time_end = list()
    print(len(rags))
    moy = [list() for _ in range(len(peoples))]
    rng_tot = calculate_rng_peoples(datas, offset)
    for l, people in enumerate(peoples):
        for k, rag in enumerate(rags):
            print(k)
            times = list()
            new_datas = list()
            new_offset = list()
            for p in people:
                new_datas.append(datas[p])
                new_offset.append(offset[p])
                time = read_times(datas[p], [rag[0] - offset[p], rag[1] - offset[p]])
                for i, (s, e) in enumerate(time):
                    time[i] = (s + offset[p], e + offset[p])
                times.append(time)
            moy[l].append(synchronie_with_features(times, new_datas, 20, rag, new_offset, rng_tot, verbose=True))
            if l == 0:
                time_end.append((rag[0] + rag[1]) / 2)
    for i, m in enumerate(moy):
        plt.plot(time_end, m, label=str(peoples[i]), linewidth=2)
    plt.legend()
    plt.xlabel("Time in sec")
    plt.ylabel("Synchronie value")
    plt.show()


def test_synchronie_with_feature(decalage=True):
    datas = list()
    for file in filenames:
        with open(file, "r") as csvfile:
            datas.append(convert_dict_reader(csv.DictReader(csvfile, delimiter=';')))
    rag = [0, 1600]
    times = list()

    times.append(read_times(datas[0], rag))
    times.append(read_times(datas[1], rag))
    times.append(read_times(datas[2], [0 if rag[0] - 224 < 0 else rag[0] - 224,
                                       rag[1] - 224 if rag[1] is not None else None]))
    decs = [[0, 0, 224]]
    if decalage:
        for i in range(1000):
            decs.append([rnd.randint(0, 500), rnd.randint(0, 500), rnd.randint(0, 500)])
        decs = [[0, 0, 224]] + list(filter(lambda x: not (x[0] < 20 and x[1] < 20 and 200 < x[2] < 240), decs))
    else:
        for i in range(1000):
            decs.append(1)
    rng_tot = calculate_rng_peoples(datas, offset)
    print(decs)
    print(len(decs))
    res_syncro = list()
    for k, dec in enumerate(decs):
        print(k)
        new_times = list()
        new_datas = list()
        if isinstance(dec, type([])):
            for j, time in enumerate(times):
                new_time = [0] * len(time)
                for i, (s, e) in enumerate(time):
                    new_time[i] = (s + dec[j], e + dec[j])
                new_times.append(new_time)
            new_datas = datas
        else:
            for data in datas:
                new_time, new_data = generate_random_from_existante(data["isTurn"], rag)
                new_times.append(new_time)
                new_datas.append({"isTurn": new_data})
            new_time = [0] * len(times[2])
            for i, (s, e) in enumerate(times[2]):
                new_time[i] = (s + 224, e + 224)
            new_times[2] = new_time
        res_syncro.append(synchronie_with_features(new_times, new_datas, 10, rag,
                                                   [0, 0, 224] if isinstance(dec, int) else dec, rng_tot))

    print(res_syncro[0])
    q = len(list(filter(lambda x: x <= res_syncro[0], res_syncro))) / len(res_syncro)
    print(q)
    plt.hist(res_syncro, np.arange(0, 1, 0.01))
    plt.plot([res_syncro[0], res_syncro[0]], [0, 50], color='r')
    plt.xlabel("synchronie value")
    plt.show()
    print("end")


def test_taux_synchronie(decalage=True):
    datas = list()
    for file in filenames:
        with open(file, "r") as csvfile:
            datas.append(convert_dict_reader(csv.DictReader(csvfile, delimiter=';')))
    rag = [0, 1600]
    times = list()

    times.append(read_times(datas[0], rag))
    times.append(read_times(datas[1], rag))
    times.append(read_times(datas[2], [0 if rag[0] - 224 < 0 else rag[0] - 224,
                                       rag[1] - 224 if rag[1] is not None else None]))
    decs = [[0, 0, 224]]
    if decalage:
        for i in range(1000):
            decs.append([rnd.randint(0, 500), rnd.randint(0, 500), rnd.randint(0, 500)])
        decs = [[0, 0, 224]] + list(filter(lambda x: not (x[0] < 20 and x[1] < 20 and 200 < x[2] < 240), decs))
    else:
        for i in range(1000):
            decs.append(1)
    rng_tot = calculate_rng_peoples(datas, offset)
    print(decs)
    print(len(decs))
    res_syncro = list()
    for k, dec in enumerate(decs):
        print(k)
        new_times = list()
        new_datas = list()
        if isinstance(dec, type([])):
            for j, time in enumerate(times):
                new_time = [0] * len(time)
                for i, (s, e) in enumerate(time):
                    new_time[i] = (s + dec[j], e + dec[j])
                new_times.append(new_time)
            new_datas = datas
        else:
            for data in datas:
                new_time, new_data = generate_random_from_existante(data["isTurn"], rag)
                new_times.append(new_time)
                new_datas.append({"isTurn": new_data})
            new_time = [0] * len(times[2])
            for i, (s, e) in enumerate(times[2]):
                new_time[i] = (s + 224, e + 224)
            new_times[2] = new_time
        res_syncro.append(compare_method(new_times, 10, rng_tot, [300, 1300]))

    print(res_syncro[0])
    q = len(list(filter(lambda x: x <= res_syncro[0], res_syncro))) / len(res_syncro)
    print(q)
    plt.hist(res_syncro, np.arange(0, 1, 0.001))
    plt.plot([res_syncro[0], res_syncro[0]], [0, 50], color='r')
    plt.xlabel("taux de synchronie")
    plt.show()
    print("end")


if __name__ == "__main__":
    #test_taux_synchronie(decalage=False)
    #test_synchronie_features(feature='speakTime', decalage=False)
    #test_synchronie_with_feature(decalage=True)
    synchronie_during_file(120, start=0, jump=60)
    #synchronie_during_file_change_peoples(240, peoples=[(0, 1, 2), (0, 2)])
    #test_size_synchronie()
    #correlation_synchronie_cohesion()
