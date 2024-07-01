import ctypes
import numpy as np

from pathlib import Path

libpwreg = ctypes.CDLL(Path(__file__).parent / "libpwreg.so")
f32ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C")
sizeptr = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags="C")
f32ptr2d = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C")
f64ptr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")
f64ptr2d = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C")

libpwreg.pwreg_f32d2_init.argtypes = [ctypes.c_size_t, f32ptr2d]
libpwreg.pwreg_f32d2_init.restype = ctypes.c_void_p
libpwreg.pwreg_f32d2_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f32d2_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f32d2_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f32d2_delete.restype = None
libpwreg.pwreg_f32d2_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f32ptr2d, f32ptr]
libpwreg.pwreg_f32d2_reduce.restype = None

libpwreg.pwreg_f64d2_init.argtypes = [ctypes.c_size_t, f64ptr2d]
libpwreg.pwreg_f64d2_init.restype = ctypes.c_void_p
libpwreg.pwreg_f64d2_copy.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d2_copy.restype = ctypes.c_void_p
libpwreg.pwreg_f64d2_delete.argtypes = [ctypes.c_void_p]
libpwreg.pwreg_f64d2_delete.restype = None
libpwreg.pwreg_f64d2_reduce.argtypes = [ctypes.c_void_p, ctypes.c_size_t, sizeptr, f64ptr2d, f64ptr]
libpwreg.pwreg_f64d2_reduce.restype = None

class PwReg:
    def __init__(self, x, y=None, funcs=None):
        if isinstance(x, PwReg):
            self.reg = libpwreg.pwreg_f64d2_copy(x.reg)
            self.num_pieces = x.num_pieces
        elif len(x) != len(y):
            raise("Every x needs an y")
        else:
            data = np.stack((np.ones(len(x)), x, y), dtype=np.float64, axis=1)
            self.reg = libpwreg.pwreg_f64d2_init(data.shape[0], data)
            self.num_pieces = data.shape[0]/2

    def __copy__(self):
        return PwReg(self)

    def __del__(self):
        libpwreg.pwreg_f64d2_delete(self.reg)

    def reduce(self, num_pieces):
        if (num_pieces < self.num_pieces):
            self.num_pieces = num_pieces
        self.starts = np.zeros((self.num_pieces), dtype=np.uintp)
        self.models = np.zeros((self.num_pieces,2), dtype=np.float64)
        self.errors = np.zeros((self.num_pieces), dtype=np.float64)
        libpwreg.pwreg_f64d2_reduce(self.reg, self.num_pieces, self.starts, self.models, self.errors)
        return (self.starts, self.models, self.errors)

    def getData(self):
        return self.reduce(np.uintp.max)

def linfit(model, x):
    return model[0] + model[1]*x

def get_breakpoints(xs, starts, models):
    starts = starts[1:].tolist()+[len(xs[:-1])]
    curstart = 0
    x = []
    y = []
    for i in range(0,len(starts)):
        curend = starts[i]
        x = x+[xs[curstart], xs[curend]]
        y = y+[linfit(models[i], val) for val in [xs[curstart], xs[curend]]]
        curstart = curend
    return (x,y)

def get_ys(xs, starts, models):
    xs_per_model = np.split(xs, starts[1:])
    ys_per_model = []
    for model, model_xs in zip(models, xs_per_model):
        ys_per_model.append(linfit(model, model_xs))
    return np.concatenate(ys_per_model)
