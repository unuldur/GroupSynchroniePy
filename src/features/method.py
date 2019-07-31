class Method:
    def compute(self,  frame_times, parameters, frame_size, lenght_frame, range_use, offset):
        pass

    def concat_in_range(self, frame_times, parameters, frame_size, lenght_frame, range_use, offset):
        res = list()
        act_parameters = list()
        last_time = 0
        for i in range(len(frame_times)):
            if frame_times[i] + offset < range_use[0]:
                continue
            if range_use[1] is not None and frame_times[i] + offset >= range_use[1]:
                return res
            act_parameters.append(parameters[i])
            if frame_times[i] + frame_size - last_time > lenght_frame:
                last_time = frame_times[i] + frame_size
                res.append(act_parameters)
                act_parameters = list()
        return res


class AvgMethod(Method):
    def compute(self, frame_times, parameters, frame_size, lenght_frame, range_use, offset):
        values = self.concat_in_range(frame_times, parameters, frame_size, lenght_frame, range_use, offset)
        avgs = list()
        for val in values:
            avgs.append(sum(val) / len(val))
        return avgs


class MaxMethod(Method):
    def compute(self, frame_times, parameters, frame_size, lenght_frame, range_use, offset):
        values = self.concat_in_range(frame_times, parameters, frame_size, lenght_frame, range_use, offset)
        avgs = list()
        for val in values:
            avgs.append(max(val))
        return avgs


class MinMethod(Method):
    def compute(self, frame_times, parameters, frame_size, lenght_frame, range_use, offset):
        values = self.concat_in_range(frame_times, parameters, frame_size, lenght_frame, range_use, offset)
        avgs = list()
        for val in values:
            avgs.append(min(val))
        return avgs


class CategoricalMethod(Method):
    def __init__(self, classifier):
        self.classifier = classifier

    def compute(self,  frame_times, parameters, frame_size, lenght_frame, range_use, offset):
        values = self.concat_in_range(frame_times, parameters, frame_size, lenght_frame, range_use, offset)
        avgs = list()
        for val in values:
            avgs.append(len(list(filter(self.classifier, val))))
        return avgs

