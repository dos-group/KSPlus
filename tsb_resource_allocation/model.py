import bisect, itertools
import matplotlib.pyplot as plt

class Model:
    def __init__(self, segends, segpeaks):
        self.segends = segends
        self.segpeaks = segpeaks

        lastend = self.segends[-1]
        for i in range(len(self.segends)-2, -1, -1):
            if self.segends[i] > lastend:
                self.segends[i] = lastend
            else:
                lastend = self.segends[i]

        #todo: sanity: what if first peak prediction == 0?
        lastpeak = 100.0
        for i in range(len(self.segpeaks)):
            if self.segpeaks[i] < lastpeak:
                self.segpeaks[i] = lastpeak
            else:
                lastpeak = self.segpeaks[i]
    def predict(self, val):
        idx = bisect.bisect_right(self.segends, val)
        idx = min(idx, len(self.segends) - 1)
        return self.segpeaks[idx]

    def get_model_failat(self, val):
        idx = bisect.bisect_right(self.segends, val)
        if idx >= len(self.segends) - 1:
            res = Model(self.segends, self.segpeaks)
            res.segpeaks[-1] *= 1.2
        else:
            prefactor = float(val) / self.segends[idx]
            res = Model([int(prefactor * se) if i >= idx else se for i,se in enumerate(self.segends)], self.segpeaks)
        return res

    def simulate(self, data):
        model = self
        executions = 1
        waste = 0.0
        time = len(data)
        while (True):
            prediction = [model.predict(x) for x in range(len(data))]
            for i,(y,pred) in enumerate(zip(data, prediction)):
                if pred < y:
                    executions += 1
                    model = model.get_model_failat(i)
                    time += i
                    break
                waste += pred - y
            else:
                break
        return (executions, waste, time)

    def plot(self, *args, **kwargs):
        xpts = []
        ypts = []
        for start, end, peak in zip(itertools.chain([0],self.segends[:-1]), self.segends, self.segpeaks):
            xpts.append(start)
            xpts.append(end)
            ypts.append(peak)
            ypts.append(peak)
        plt.plot(xpts, ypts, *args, **kwargs)

class Unimodel(Model):
    def get_model_failat(self, val):
        idx = bisect.bisect_right(self.segends, val)
        idx = min(idx, len(self.segends) - 1)
        res = Unimodel(self.segends, [p * 2.0 if i >= idx else p for i,p in enumerate(self.segpeaks)])
        return res

class UnimodelPartial(Model):
    def get_model_failat(self, val):
        idx = bisect.bisect_right(self.segends, val)
        idx = min(idx, len(self.segends) - 1)
        res = UnimodelPartial(self.segends, [p * 2.0 if i == idx else p for i,p in enumerate(self.segpeaks)])
        return res

class TovarModel(Model):
    defaultval = 128.0 * 1000.0
    def get_model_failat(self, val):
        return TovarModel([1], [self.defaultval])
