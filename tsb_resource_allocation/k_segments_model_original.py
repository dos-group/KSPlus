from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


# k-Segments Model
# Author: Nils Diedrich (nils.diedrich@aol.de)

class KSegmentsModelOriginal:

    def __init__(
            self,
            monotonically_increasing=True,
            default_value=100,
            k=2,
            time_mode=1,

    ):
        self.monotonically_increasing = monotonically_increasing
        self.default_value = default_value
        self.k = k
        self.time_mode = time_mode
        self.x = []
        self.y_t = []
        self.y_v = []

    def predict(self, new_x):
        rend = []
        rpeaks = []
        p_t = self.model_time.predict(np.array([new_x]).reshape((-1, 1))).astype(int)[0] // self.k

        for i,m in enumerate(self.model_value):
            p_v = m.predict(np.array([new_x]).reshape((-1, 1)))[0]
            if self.monotonically_increasing and len(rpeaks) > 0 and p_v < rpeaks[-1]:
                p_v = rpeaks[-1]
            if p_v <= 0:
                p_v = self.default_value
            if p_t <= 0:
                p_t = 1
            rend.append(p_t*(i+1))
            rpeaks.append(p_v)
        return (rend, rpeaks)

    def train_model(self):
        #if len(self.files) == 0:
        #    print("Training files are empty!")
        #    return

        # f:files = (0:value, 1:file_events, 2:total_input_file_size)

        #self.x = [f[2] for f in self.files]
        self.y_t = [len(m) for m in self.mem]
        self.y_v = [[] for _ in range(self.k)]

        for y in self.mem:
            i = len(y) // self.k
            for j in range(1, self.k):
                self.y_v[j - 1].append(max(y[((j - 1) * i):(j * i)]))
            self.y_v[self.k - 1].append(max(y[((self.k - 1) * i):len(y)]))

        self.model_value = list(map(lambda y: self.train_linear_model(self.x, y, 0), self.y_v))
        self.model_time = self.train_linear_model(self.x, self.y_t, self.time_mode)

    def train_linear_model(self, x, y, mode):
        lm = LinearRegression()
        lm.fit(np.asarray(x).reshape((-1, 1)), y)
        l_diff = 0
        for i, v in enumerate(x):
            pred = lm.predict(np.array([v]).reshape((-1, 1)))
            diff = y[i] - pred
            if mode == 0:
                l_diff = max(diff, l_diff)
            elif mode == -1:
                l_diff = 0
            else:
                l_diff = min(diff, l_diff)
        lm.intercept_ += l_diff
        return lm

    def plot_time_regression(self):
        plt.scatter(self.x, self.y_t, color='g')
        plt.plot(self.x, self.model_time.predict(np.asarray([self.x]).reshape((-1, 1))), color='r')

    def plot_ram_regression(self, num_cluster):
        self.plot_regression(num_cluster, self.y_v, self.model_value)

    def plot_regression(self, num_cluster, y, model):
        plt.scatter(self.x, y[num_cluster], color='g')
        plt.plot(self.x, model[num_cluster].predict(np.asarray([self.x]).reshape((-1, 1))), color='r')
