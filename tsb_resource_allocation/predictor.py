import model, itertools
import numpy as np
from sklearn.linear_model import LinearRegression
from FirstAllocation import FirstAllocation
from default_model import Default
from witt_task_model import WittTaskModel as Witt
from k_segments_model_original import KSegmentsModelOriginal as KSegments

default_num_segments = 4


class Segment:
    def __init__(self, peak, size=1, error=0):
        self.peak = peak
        self.size = size
        self.error = error

    def merge(self, other):
        self.peak = max([self.peak, other.peak])
        self.size += other.size
        self.error += other.error + self.get_merge_error(other)

    def get_merge_error(self, other):
        return (self.peak - other.peak) * other.size if self.peak >= other.peak else (other.peak - self.peak) * self.size


def get_segments(data, segcount):
    res = [Segment(data[0])]
    for next in [Segment(d) for d in data[1:]]:
        if res[-1].peak >= next.peak:
            res[-1].merge(next)
        else:
            res.append(next)
    while len(res) > segcount:
        idx = np.argmin([first.get_merge_error(second) for first, second in zip(res, res[1:])])
        res[idx].merge(res[idx + 1])
        res.pop(idx + 1)
    res = ([r.size for r in res], [r.peak for r in res], [r.error for r in res])
    #if (len(res[0]) != segcount):
    #raise Exception(f"Error: too few segments make sense. Got {len(res[0])} instead of {segcount}.")
    while (len(res[0]) < segcount):  # todo: better handling of less segments
        res[0].append(res[0][-1] + 10)
        res[1].append(res[1][-1])
        res[2].append(0)
    return res


class Predictor:
    def __init__(self, tasks, num_segments=default_num_segments):
        tasks = [t for t in tasks if len(t.mem) >= num_segments]  # only use tasks that ran long enough to learn
        self.instances = tasks
        self.segments = [get_segments(t.mem, num_segments) for t in tasks]
        segends = [[i for i in itertools.accumulate(starts)] for (starts, peaks, _) in self.segments]
        segpeaks = [peaks for (starts, peaks, _) in self.segments]
        self.endpred = [
            LinearRegression().fit(np.array([t.insize for t in tasks]).reshape(-1, 1), [se[i] for se in segends]) for i
            in range(num_segments)]
        self.peakpred = [
            LinearRegression().fit(np.array([t.insize for t in tasks]).reshape(-1, 1), [sp[i] for sp in segpeaks]) for i
            in range(num_segments)]

    def get_model(self, taskinstance):
        return model.Model([r.predict([[taskinstance.insize]])[0] * 0.85 for r in self.endpred],
                           [r.predict([[taskinstance.insize]])[0] * 1.1 for r in self.peakpred])


class Unipredictor:
    def __init__(self, tasks, num_segments=default_num_segments):
        tasks = [t for t in tasks if len(t.mem) >= num_segments]  # only use tasks that ran long enough to learn
        self.instances = tasks
        self.num_segments = num_segments
        self.segments = [[int(float(len(t.mem)) / num_segments) * i for i in range(1, num_segments + 1)] for t in tasks]
        for i in range(len(tasks)):
            self.segments[i][-1] = len(tasks[i].mem)
        segends = self.segments
        segpeaks = [[max(t.mem[start:end]) for start, end in zip(itertools.chain([0], segends[i]), segends[i])] for i, t
                    in enumerate(tasks)]
        self.timepred = LinearRegression().fit(np.array([t.insize for t in tasks]).reshape(-1, 1),
                                               [se[-1] for se in segends])
        self.peakpred = [
            LinearRegression().fit(np.array([t.insize for t in tasks]).reshape(-1, 1), [sp[i] for sp in segpeaks]) for i
            in range(num_segments)]
        self.offsets = [max([segpeaks[tidx][sidx] -
                             self.peakpred[sidx].predict(np.array([self.instances[tidx].insize]).reshape((-1, 1)))[0]
                             for tidx in range(len(segpeaks))]) for sidx in range(len(self.peakpred))]

    def get_model(self, taskinstance):
        time = self.timepred.predict([[taskinstance.insize]])[0]
        return model.Unimodel([int(float(time) / self.num_segments) * i for i in range(1, self.num_segments + 1)],
                              [r.predict([[taskinstance.insize]])[0] + off for r, off in
                               zip(self.peakpred, self.offsets)])


class Unipredictor2:
    def __init__(self, tasks, num_segments=default_num_segments):
        tasks = [t for t in tasks if len(t.mem) >= num_segments]  # only use tasks that ran long enough to learn
        self.kseg = KSegments(True, 100, num_segments, 1)
        self.kseg.x = [ti.insize for ti in tasks]
        self.kseg.mem = [ti.mem for ti in tasks]
        self.kseg.train_model()

    def get_model(self, taskinstance):
        (ends, peaks) = self.kseg.predict(taskinstance.insize)
        return model.Unimodel(ends, peaks)


class UnipredictorSelective(Unipredictor):
    def get_model(self, taskinstance):
        time = self.timepred.predict([[taskinstance.insize]])[0]
        return model.UnimodelSelective(
            [int(float(time) / self.num_segments) * i for i in range(1, self.num_segments + 1)],
            [r.predict([[taskinstance.insize]])[0] + off for r, off in zip(self.peakpred, self.offsets)])


class Unipredictor2Selective(Unipredictor2):
    def get_model(self, taskinstance):
        (ends, peaks) = self.kseg.predict(taskinstance.insize)
        return model.UnimodelSelective(ends, peaks)


class TovarPredictor:
    def __init__(self, tasks, num_segments):
        self.fa = FirstAllocation(name="my memory usage")
        for ti in tasks:
            self.fa.add_data_point(value=max(ti.mem), time=len(ti.mem))

    def get_model(self, taskinstance):
        return model.TovarModel([0], [self.fa.first_allocation(mode='waste')])


class TovarImprovedPredictor:
    def __init__(self, tasks, num_segments):
        self.fa = FirstAllocation(name="my memory usage")
        for ti in tasks:
            self.fa.add_data_point(value=max(ti.mem), time=len(ti.mem))

    def get_model(self, taskinstance):
        return model.Unimodel([0], [self.fa.first_allocation(mode='waste')])


class DefaultPredictor:

    def __init__(self, tasks, num_segments):
        self.pred = Default()

    def get_model(self, taskinstance):
        return model.DefaultModel([0], taskinstance.max_mem)


class WittPredictor:
    def __init__(self, tasks, num_segments):
        self.pred = Witt(mode="mean+-")
        self.pred.x = [ti.insize for ti in tasks]
        self.pred.y = [max(ti.mem) for ti in tasks]
        self.pred.train_model()

    def get_model(self, taskinstance):
        return model.Unimodel([0], [self.pred.predict(len(taskinstance.mem))])
