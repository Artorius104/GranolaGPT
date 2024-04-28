import ctypes
import numpy as np
import numpy.ctypeslib
import random



rust_path = "my_lib/target/debug/deps/libmy_lib.so"

my_lib = ctypes.cdll.LoadLibrary(rust_path)


my_lib.create_MyMLP.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
my_lib.create_MyMLP.restype = ctypes.c_void_p

nlp = np.array([2, 3, 1],np.int32)
c_nlp = np.ctypeslib.as_ctypes(nlp)

model = my_lib.create_MyMLP(c_nlp, len(nlp))


my_lib.train_MyMLP.argtypes = [ctypes.c_void_p, 
                               
                               ctypes.POINTER(ctypes.c_double), 
                               ctypes.c_int32, 
                               
                               ctypes.POINTER(ctypes.c_double), 
                               ctypes.c_int32,
                               
                               ctypes.c_int32,
                               ctypes.c_double,
                               ctypes.c_int32,
                               ctypes.c_bool]
my_lib.train_MyMLP.restype = ctypes.POINTER(ctypes.c_double)

X = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]

y = [
    [-1.0],
    [1.0],
    [1.0],
    [-1.0]
]

n_samples = len(X)

X_chunk = len(X[0])
X = np.array(X, np.float64).flatten()

y_chunk = len(y[0])
y = np.array(y, np.float64).flatten()

c_X = np.ctypeslib.as_ctypes(X)
c_y = np.ctypeslib.as_ctypes(y)

epochs = 500000

p_losses = my_lib.train_MyMLP(model,
                    
                   c_X, 
                   X_chunk, 
                   
                   c_y,
                   y_chunk, 
                   
                   n_samples,
                   0.1,
                   epochs,
                   True)


my_lib.predict_MyMLP.argtypes = [ctypes.c_void_p,
                                 ctypes.c_bool,
                                 ctypes.POINTER(ctypes.c_double),
                                 ctypes.c_int32,
                                 ctypes.c_int32]
my_lib.predict_MyMLP.restype = ctypes.POINTER(ctypes.c_double)

p_predictions = my_lib.predict_MyMLP(model,
                     True,
                     c_X,
                     X_chunk,
                     n_samples)

predictions = np.ctypeslib.as_array(p_predictions, (y_chunk * n_samples,))
print(predictions)

losses = np.ctypeslib.as_array(p_losses, (epochs,))
print(losses)