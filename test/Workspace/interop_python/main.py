import ctypes
import numpy as np
import numpy.ctypeslib
import random

if __name__ == "__main__":

    rust_path = "../mylibrary/target/debug/deps/libmylibrary.so"

    my_lib = ctypes.cdll.LoadLibrary(rust_path)

    my_lib.create_linear_regression.argtypes = [ctypes.c_double, ctypes.c_double]
    my_lib.create_linear_regression.restype = ctypes.c_void_p
    
    my_lib.predict_linear_regression.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int32]
    my_lib.predict_linear_regression.restype = ctypes.POINTER(ctypes.c_double)
    
    my_lib.fit_linear_regression.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_double]
    my_lib.fit_linear_regression.restype = ctypes.c_void_p
     
    my_lib.history_get_losses.argtypes = [ctypes.c_void_p]
    my_lib.history_get_losses.restype = ctypes.POINTER(ctypes.c_double)
    
    
    
    test_x = np.array([x for x in range(100)], np.float64)
    test_y = np.array([(celsius * 9/5) + 32 for celsius in test_x], np.float64)
    c_X = np.ctypeslib.as_ctypes(test_x)
    c_y = np.ctypeslib.as_ctypes(test_y)
    
    model = my_lib.create_linear_regression(5,4)
    
    history = my_lib.fit_linear_regression(model, c_X, c_y, len(c_X), 1000000, 0.000001)
    
    data = np.array([1,2,3,4,5], dtype=np.float64)
    
    c_data = np.ctypeslib.as_ctypes(data)
    p_pred = my_lib.predict_linear_regression(model, c_data, len(data))
    
    pred = numpy.ctypeslib.as_array(p_pred, (len(data),))
    print(pred)
    
    p_losses = my_lib.history_get_losses(history)
    losses = np.ctypeslib.as_array(p_losses,(1000000,))
    
    print("COMPILE")
    
