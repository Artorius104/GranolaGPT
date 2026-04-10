import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

def load_my_lib(path):
    my_lib = ctypes.cdll.LoadLibrary(path)
    
    ############## MyMLP ##############

    my_lib.create_MyMLP.argtypes = [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_uint64]
    my_lib.create_MyMLP.restype = ctypes.c_void_p

    my_lib.destroy_MyMLP.argtypes = [ctypes.c_void_p]

    my_lib.train_MyMLP.argtypes = [
        ctypes.c_void_p,
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_size_t,
        ctypes.c_bool
    ]

    my_lib.predict_MyMLP.argtypes = [
        ctypes.c_void_p,
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_bool
    ]
    my_lib.predict_MyMLP.restype = ctypes.POINTER(ctypes.c_double)

    my_lib.save_MyMLP.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    my_lib.save_MyMLP.restype = ctypes.c_int

    my_lib.load_MyMLP.argtypes = [ctypes.c_char_p]
    my_lib.load_MyMLP.restype = ctypes.c_void_p
    
    my_lib.summary_MyMLP.argtypes = [ctypes.c_void_p]
    my_lib.summary_MyMLP.restype = None

    ############## MyCNN ##############

    my_lib.create_MyCNN.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    my_lib.create_MyCNN.restype = ctypes.c_void_p

    my_lib.destroy_MyCNN.argtypes = [ctypes.c_void_p]
    
    my_lib.add_conv_layer_MyCNN.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_double, ctypes.c_uint64]
    
    my_lib.add_maxpooling_layer_MyCNN.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

    my_lib.add_dense_layer_MyCNN.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_double, ctypes.c_uint64]
    
    my_lib.set_optimizer_SGD.argtypes = [ctypes.c_void_p, ctypes.c_double]
    
    my_lib.train_MyCNN.argtypes = [
        ctypes.c_void_p,
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_size_t, 
        ctypes.c_size_t,
        ctypes.c_size_t, 
        ctypes.c_size_t, 
        ctypes.c_size_t,
        ctypes.c_size_t, 
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint64
    ]
    
    my_lib.predict_MyCNN.argtypes = [
        ctypes.c_void_p,
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t
    ]
    my_lib.predict_MyCNN.restype = ctypes.POINTER(ctypes.c_double)
    
    my_lib.save_MyCNN.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    my_lib.save_MyCNN.restype = ctypes.c_int

    my_lib.load_MyCNN.argtypes = [ctypes.c_char_p]
    my_lib.load_MyCNN.restype = ctypes.c_void_p
    
    ############## MyLinearModel ##############

    my_lib.create_MyLinearModel.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint64]
    my_lib.create_MyLinearModel.restype = ctypes.c_void_p

    my_lib.destroy_MyLinearModel.argtypes = [ctypes.c_void_p]

    my_lib.train_MyLinearModel.argtypes = [
        ctypes.c_void_p,
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_size_t,
        ctypes.c_int
    ]

    my_lib.predict_MyLinearModel.argtypes = [
        ctypes.c_void_p,
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int
    ]
    my_lib.predict_MyLinearModel.restype = ctypes.POINTER(ctypes.c_double)

    my_lib.save_MyLinearModel.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    my_lib.save_MyLinearModel.restype = ctypes.c_bool

    my_lib.load_MyLinearModel.argtypes = [ctypes.c_char_p]
    my_lib.load_MyLinearModel.restype = ctypes.c_void_p
    
        ############## MyRBF ##############

    my_lib.create_MyRBF.argtypes = [
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_uint64
    ]
    my_lib.create_MyRBF.restype = ctypes.c_void_p

    my_lib.destroy_MyRBF.argtypes = [ctypes.c_void_p]

    my_lib.train_MyRBF.argtypes = [
        ctypes.c_void_p,
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_size_t,
        ctypes.c_bool
    ]

    my_lib.predict_MyRBF.argtypes = [
        ctypes.c_void_p,
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_bool
    ]
    my_lib.predict_MyRBF.restype = ctypes.POINTER(ctypes.c_double)

    my_lib.save_MyRBF.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    my_lib.save_MyRBF.restype = ctypes.c_int

    my_lib.load_MyRBF.argtypes = [ctypes.c_char_p]
    my_lib.load_MyRBF.restype = ctypes.c_void_p
    
        ############## MyLinearModel_Regression ##############

    my_lib.create_MyLinearModel_Regression.argtypes = []
    my_lib.create_MyLinearModel_Regression.restype = ctypes.c_void_p

    my_lib.destroy_MyLinearModel_Regression.argtypes = [ctypes.c_void_p]

    my_lib.train_MyLinearModel_Regression.argtypes = [
        ctypes.c_void_p,
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t
    ]

    my_lib.predict_MyLinearModel_Regression.argtypes = [
        ctypes.c_void_p,
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ctypes.c_size_t
    ]
    my_lib.predict_MyLinearModel_Regression.restype = ctypes.POINTER(ctypes.c_double)

    my_lib.save_MyLinearModel_Regression.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    my_lib.save_MyLinearModel_Regression.restype = ctypes.c_int

    my_lib.load_MyLinearModel_Regression.argtypes = [ctypes.c_char_p]
    my_lib.load_MyLinearModel_Regression.restype = ctypes.c_void_p

    return my_lib