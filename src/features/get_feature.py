from functools import reduce
import matplotlib.pyplot as plt
import re

from matplotlib.widgets import RadioButtons, Slider

from constante import colors


def print_multiple(features, start, end, is_turns):
    fig, ax = plt.subplots()
    plt.subplots_adjust(right=0.5, bottom=0.3)
    names = list()
    features[0].plot_histo(start, ax=ax, show=False)
    for i, f in enumerate(features):
        names.append(f.title)
    a = plt.axes([0.6, 0.3, 0.3, 0.6])
    a2 = plt.axes([0.6, 0.1, 0.15, 0.15])
    a3 = plt.axes([0.1, 0.1, 0.4, 0.1])
    a4 = plt.axes([0.8, 0.1, 0.15, 0.15])
    values = [names[0], "affichage 1", 1]
    rads = RadioButtons(a, names)
    check = RadioButtons(a2, ["affichage 1", "affichage 2"])
    rb_norm = RadioButtons(a4, ["normalisation 1", "normalisation 2"], active=1)
    sframe = Slider(a3, 'frame size', 0.1, 20, valinit=2, valstep=0.1)

    def update():
        ax.clear()
        if values[1] == "affichage 1":
            features[names.index(values[0])].plot_histo(start, ax=ax, show=False)
        if values[1] == "affichage 2":
            features[names.index(values[0])].plot(start, ax=ax, show=False)
        fig.canvas.draw_idle()

    def radio_update(name):
        values[0] = name
        update()

    def affichage_update(type):
        values[1] = type
        update()

    def norm_update(type):
        values[2] = type
        update_values(None)

    def update_values(_):
        frame_size = sframe.val
        for f in features:
            f.length_frame = float(frame_size)
            f.calc(is_turns, start, end)
            if values[2] == "normalisation 1" and values[1] != "affichage 2":
                f.normalize()
            if values[2] == "normalisation 2" or values[1] == "affichage 2":
                f.normalize_2(start)
        update()

    sframe.on_changed(update_values)
    rads.on_clicked(radio_update)
    check.on_clicked(affichage_update)
    rb_norm.on_clicked(norm_update)

    plt.show()


class GetFeature:

    def __init__(self, offset, frame_time, length_frame):
        self.title = ' '.join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__)[1:])
        self.results = list()
        self.x_label = "time(s)"
        self.y_label = ""
        self.offset = offset
        self.frame_time = frame_time
        self.length_frame = length_frame

    def get_range(self, is_turns, start, end):
        res = list()
        for turn in range(len(is_turns)):
            supplements = list()
            good_offset = 0

            if self.offset[turn] < end:
                offset_end = end - self.offset[turn]
            else:
                res.append([0 for _ in range(int((end - start) / self.frame_time))])
                continue

            if self.offset[turn] < start:
                good_offset = start - self.offset[turn]
            else:
                supplements = [0 for _ in range(int((self.offset[turn] - start) / self.frame_time))]

            if good_offset > end:
                res.append(list())
                continue
            res.append(
                supplements + is_turns[turn][int(good_offset / self.frame_time):int(offset_end / self.frame_time) + 1])
        return res

    def get(self, is_turns, start, end):
        pass

    def calc(self, is_turns, start, end):
        time = start
        self.results = list()
        while time < end:
            val = self.get(is_turns, time, time + self.length_frame)
            for i in range(len(val)):
                if len(self.results) <= i:
                    self.results.append(list())
                self.results[i].append(val[i])
            time += self.length_frame
        return self.results

    def normalize(self):
        for i in range(len(self.results)):
            self.results[i] = [v / self.length_frame for v in self.results[i]]

    def sum_offset(self, i, j, start, size):
        s = 0
        off = 0
        if start > self.offset[i]:
            off = start
        for k in range(size):
            off2 = 0
            if start > self.offset[k]:
                off2 = start
            dec = (off2 - off) / self.length_frame
            if j >= dec and j - round(dec) < len(self.results[k]):
                s += self.results[k][j - round(dec)]
        return s

    def calc_sum(self, start):
        l = 0
        for i in range(len(self.offset)):
            val = int(self.offset[i] / self.length_frame) + len(self.results[i])
            if val > l:
                l += val
        l_sum = list()
        for j in range(l):
            s = 0
            for i in range(len(self.offset)):
                if start > self.offset[i]:
                    off = start / self.length_frame
                else:
                    off = self.offset[i] / self.length_frame
                if j >= off and j - int(off) < len(self.results[i]):
                    s += self.results[i][j - int(off)]
            l_sum.append(s)
        return l_sum

    def normalize_2(self, start):
        new_res = [r.copy() for r in self.results]
        for i in range(len(self.results)):
            for j in range(len(self.results[i])):
                s = self.sum_offset(i, j, start, len(self.results))
                if s > 0:
                    new_res[i][j] /= s
        self.results = new_res

    def plot(self, start, labels=None, save=False, directory_path="", show=True, ax=None):
        if ax is None:
            ax = plt.gca()
        for i in range(len(self.results)):
            if labels is None or i >= len(labels) or len(self.results) == 1:
                label = None
            else:
                label = labels[i]
            if self.offset[i] <= start:
                offset_print = start
            else:
                offset_print = self.offset[i]
            offset_print = int(offset_print / self.length_frame) * self.length_frame
            size = [self.sum_offset(i, j, start, i) for j in range(len(self.results[i]))]
            ax.bar([(t * self.length_frame + offset_print) for t in range(len(self.results[i]))]
                   , self.results[i], self.length_frame, size, label=label, color=colors[i])
        ax.set_title(self.title)
        if labels is not None and len(self.results) != 1:
            ax.legend()
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        if save:
            ax.savefig(directory_path + self.title)
        if show:
            plt.show()

    def plot_histo(self, start, labels=None, save=False, directory_path="", show=True, ax=None):
        if ax is None:
            ax = plt.gca()
        size = self.length_frame / len(self.results)
        for i in range(len(self.results)):
            if labels is None or i >= len(labels) or len(self.results) == 1:
                label = None
            else:
                label = labels[i]
            if self.offset[i] <= start:
                offset_print = start
            else:
                offset_print = self.offset[i]
            offset_print = int(offset_print / self.length_frame) * self.length_frame
            ax.bar([(t * self.length_frame + offset_print - self.length_frame / 2 + i * size) for t in
                    range(len(self.results[i]))]
                   , self.results[i], size, label=label, color=colors[i])
            # if len(self.results) > 1:
            # ax.plot([(t * self.length_frame + offset_print) for t in range(len(self.results[i]))],
            #        [1/len(self.results) for _ in range(len(self.results[i]))], "c--")
        ax.set_title(self.title)
        if labels is not None and len(self.results) != 1:
            ax.legend()
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        if save:
            ax.savefig(directory_path + self.title)
        if show:
            plt.show()


class GetPauseTime(GetFeature):

    def __init__(self, offset, frame_time, length_frame):
        super().__init__(offset, frame_time, length_frame)
        self.y_label = "total time(s)"
        self.size = 1

    def get(self, is_turns, start, end):
        is_turn_range = self.get_range(is_turns, start, end)
        res = list()
        self.size = len(is_turns)
        for is_turn in is_turn_range:
            if len(is_turn) == 0:
                res.append(0)
                continue
            pause = 0
            tmp = 0
            for speak in is_turn:
                if not speak:
                    tmp += self.frame_time
                if speak:
                    if tmp <= 4:
                        pause += tmp
                    tmp = 0
            res.append(pause)
        return res


class GetTurnDuration(GetFeature):

    def __init__(self, offset, frame_time, length_frame):
        super().__init__(offset, frame_time, length_frame)
        self.y_label = "time(s)"

    def get(self, is_turns, start, end):
        is_turn_range = self.get_range(is_turns, start, end)
        res = list()
        for is_turn in is_turn_range:
            if len(is_turn) == 0:
                continue
            turns = list()
            time = 0
            for speak in is_turn:
                if speak:
                    time += self.frame_time
                elif time != 0:
                    turns.append(time)
                    time = 0
            turns.append(time)
            res.append(sum(turns) / len(turns))
        return res


class GetTurnPauseRatio(GetFeature):

    def __init__(self, offset, frame_time, length_frame):
        super().__init__(offset, frame_time, length_frame)
        self.y_label = "speak(s)/pause(s)"

    def get(self, is_turns, start, end):
        is_turn_range = self.get_range(is_turns, start, end)
        res = list()
        for is_turn in is_turn_range:
            if len(is_turn) == 0:
                continue
            pause = 0
            speak_val = 0
            for speak in is_turn:
                if not speak:
                    pause += self.frame_time
                    speak_val += self.frame_time
            res.append(speak_val / (pause + 1))
        return res

    def normalize(self):
        pass


class GetSpeakingTime(GetFeature):

    def __init__(self, offset, frame_time, length_frame):
        super().__init__(offset, frame_time, length_frame)
        self.size = 1
        self.y_label = "time(s)"

    def get(self, is_turns, start, end):
        is_turn_range = self.get_range(is_turns, start, end)
        speak_val = 0
        self.size = len(is_turns)
        for is_turn in is_turn_range:
            for speak in is_turn:
                if speak:
                    speak_val += self.frame_time
        return [speak_val]

    def normalize(self):
        for i in range(len(self.results)):
            self.results[i] = [v / (self.length_frame * self.size) for v in self.results[i]]

    def normalize_2(self, start):
        self.normalize()


class GetBackChannelNumber(GetFeature):

    def __init__(self, offset, frame_time, length_frame, time_back_channel):
        super().__init__(offset, frame_time, length_frame)
        self.time_back_channel = time_back_channel
        self.y_label = "number of back channel"

    def get(self, is_turns, start, end):
        is_turn_range = self.get_range(is_turns, start, end)
        res = list()
        for is_turn in is_turn_range:
            if len(is_turn) == 0:
                continue
            time = 0
            back_channel = 0
            for speak in is_turn:
                if speak:
                    time += self.frame_time
                elif time != 0:
                    if time <= self.time_back_channel:
                        back_channel += 1
                    time = 0
            res.append(back_channel)
        return res

    def normalize(self):
        pass


class GetLongTurnRatio(GetFeature):

    def __init__(self, offset, frame_time, length_frame, time_short_turn):
        super().__init__(offset, frame_time, length_frame)
        self.time_short_turn = time_short_turn
        self.y_label = "number of long turn/number of short turn"

    def get(self, is_turns, start, end):
        is_turn_range = self.get_range(is_turns, start, end)
        res = list()
        for is_turn in is_turn_range:
            if len(is_turn) == 0:
                continue
            time = 0
            short_turn = 0
            long_turn = 1
            for speak in is_turn:
                if speak:
                    time += self.frame_time
                elif time != 0:
                    if time <= self.time_short_turn:
                        short_turn += 1
                    else:
                        long_turn += 1
                    time = 0
            res.append(short_turn / long_turn)
        return res

    def normalize(self):
        pass


class GetSilence(GetFeature):

    def __init__(self, offset, frame_time, length_frame):
        super().__init__(offset, frame_time, length_frame)
        self.y_label = "time(s)"

    def get(self, is_turns, start, end):
        is_turn_range = self.get_range(is_turns, start, end)
        silences_list = list()
        for is_turn in is_turn_range:
            for i in range(len(is_turn)):
                if i >= len(silences_list):
                    silences_list.append(is_turn[i])
                else:
                    silences_list[i] = silences_list[i] or is_turn[i]
        return [reduce(lambda act, n: act + 1 if not n else act, silences_list, 0) * self.frame_time]

    def normalize_2(self, start):
        self.normalize()


class GetOverlap(GetFeature):

    def __init__(self, offset, frame_time, length_frame):
        super().__init__(offset, frame_time, length_frame)
        self.y_label = "time(s)"
        self.size = 1

    def get(self, is_turns, start, end):
        is_turn_range = self.get_range(is_turns, start, end)
        speak = list()
        self.size = len(is_turns)
        for is_turn in is_turn_range:
            for i in range(len(is_turn)):
                if i >= len(speak):
                    speak.append(0)
                if is_turn[i]:
                    speak[i] += 1
        return [reduce(lambda act, n: act + 1 if n >= 2 else act, speak, 0) * self.frame_time]

    def normalize_2(self, start):
        self.normalize()

    def normalize(self):
        for i in range(len(self.results)):
            self.results[i] = [v / (self.length_frame * self.size) for v in self.results[i]]


class GetInterruption(GetFeature):

    def __init__(self, offset, frame_time, length_frame):
        super().__init__(offset, frame_time, length_frame)
        self.y_label = "number of interruptions"

    def another_speak(self, i, j, is_turns):
        for is_turn in range(len(is_turns)):
            if is_turn == i:
                continue
            if is_turns[is_turn][j]:
                return True
        return False

    def get(self, is_turns, start, end):
        is_turn_range = self.get_range(is_turns, start, end)
        res = list()
        max_size = max([len(t) for t in is_turn_range])
        for t in is_turn_range:
            while len(t) < max_size:
                t.append(0)
        for j in range(len(is_turn_range)):
            interruption = 0
            stade = 0
            for i in range(len(is_turn_range[j])):
                if not is_turn_range[j][i] and self.another_speak(j, i, is_turn_range) and stade == 0:
                    stade = 1
                if is_turn_range[j][i] and not self.another_speak(j, i, is_turn_range) and stade == 1:
                    stade = 0
                if is_turn_range[j][i] and self.another_speak(j, i, is_turn_range) and stade == 1:
                    stade = 2
                if is_turn_range[j][i] and not self.another_speak(j, i, is_turn_range) and stade == 2:
                    interruption += 1
                    stade = 0
                if not is_turn_range[j][i] and self.another_speak(j, i, is_turn_range) and stade == 2:
                    stade = 0
            res.append(interruption)
        return res


class GetInterruptionOverlap(GetFeature):

    def __init__(self, offset, frame_time, length_frame):
        super().__init__(offset, frame_time, length_frame)
        self.y_label = "time (s) of interruptions"

    def another_speak(self, i, j, is_turns):
        for is_turn in range(len(is_turns)):
            if is_turn == i:
                continue
            if is_turns[is_turn][j]:
                return True
        return False

    def get(self, is_turns, start, end):
        is_turn_range = self.get_range(is_turns, start, end)
        res = list()
        max_size = max([len(t) for t in is_turn_range])
        for t in is_turn_range:
            while len(t) < max_size:
                t.append(0)
        for j in range(len(is_turn_range)):
            total_time = 0
            stade = 0
            time = 0
            for i in range(len(is_turn_range[j])):
                if not is_turn_range[j][i] and self.another_speak(j, i, is_turn_range) and stade == 0:
                    stade = 1
                    time = 0
                if is_turn_range[j][i] and not self.another_speak(j, i, is_turn_range) and stade == 1:
                    stade = 0
                    time = 0
                if is_turn_range[j][i] and self.another_speak(j, i, is_turn_range) and stade == 1:
                    time += self.frame_time
                    stade = 2
                if is_turn_range[j][i] and not self.another_speak(j, i, is_turn_range) and stade == 2:
                    total_time += time
                    stade = 0
                    time = 0
                if not is_turn_range[j][i] and self.another_speak(j, i, is_turn_range) and stade == 2:
                    stade = 0
                    time = 0
            res.append(total_time)
        return res


class GetFailedInterruption(GetFeature):

    def __init__(self, offset, frame_time, length_frame):
        super().__init__(offset, frame_time, length_frame)
        self.y_label = "number of failed interruptions"

    def another_speak(self, i, j, is_turns):
        for is_turn in range(len(is_turns)):
            if is_turn == i:
                continue
            if is_turns[is_turn][j]:
                return True
        return False

    def get(self, is_turns, start, end):
        is_turn_range = self.get_range(is_turns, start, end)
        res = list()
        max_size = max([len(t) for t in is_turn_range])
        for t in is_turn_range:
            while len(t) < max_size:
                t.append(0)
        for j in range(len(is_turn_range)):
            interruption = 0
            stade = 0
            for i in range(len(is_turn_range[j])):
                if not is_turn_range[j][i] and self.another_speak(j, i, is_turn_range) and stade == 0:
                    stade = 1
                if is_turn_range[j][i] and not self.another_speak(j, i, is_turn_range) and stade == 1:
                    stade = 0
                if is_turn_range[j][i] and self.another_speak(j, i, is_turn_range) and stade == 1:
                    stade = 2
                if is_turn_range[j][i] and not self.another_speak(j, i, is_turn_range) and stade == 2:
                    stade = 0
                if not is_turn_range[j][i] and self.another_speak(j, i, is_turn_range) and stade == 2:
                    interruption += 1
                    stade = 0
            res.append(interruption)
        return res


class GetFailedInterruptionOverlap(GetFeature):

    def __init__(self, offset, frame_time, length_frame):
        super().__init__(offset, frame_time, length_frame)
        self.y_label = "time (s) of failed interruptions"

    def another_speak(self, i, j, is_turns):
        for is_turn in range(len(is_turns)):
            if is_turn == i:
                continue
            if is_turns[is_turn][j]:
                return True
        return False

    def get(self, is_turns, start, end):
        is_turn_range = self.get_range(is_turns, start, end)
        res = list()
        max_size = max([len(t) for t in is_turn_range])
        for t in is_turn_range:
            while len(t) < max_size:
                t.append(0)
        for j in range(len(is_turn_range)):
            total_time = 0
            stade = 0
            time = 0
            for i in range(len(is_turn_range[j])):
                if not is_turn_range[j][i] and self.another_speak(j, i, is_turn_range) and stade == 0:
                    stade = 1
                    time = 0
                if is_turn_range[j][i] and not self.another_speak(j, i, is_turn_range) and stade == 1:
                    stade = 0
                    time = 0
                if is_turn_range[j][i] and self.another_speak(j, i, is_turn_range) and stade == 1:
                    time += self.frame_time
                    stade = 2
                if is_turn_range[j][i] and not self.another_speak(j, i, is_turn_range) and stade == 2:
                    stade = 0
                    time = 0
                if not is_turn_range[j][i] and self.another_speak(j, i, is_turn_range) and stade == 2:
                    total_time += time
                    stade = 0
                    time = 0
            res.append(total_time)
        return res
