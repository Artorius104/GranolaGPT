import ctypes
import numpy as np
import numpy.ctypeslib

def load_my_lib():
    # Chemin vers la librairie Rust compilée
    rust_path = "../../my_lib/target/release/deps/libmy_lib.so"

    # Chargement de la bibliothèque Rust
    my_lib = ctypes.cdll.LoadLibrary(rust_path)


    #####################################

    ############## MyRBF_rb ##############

    my_lib.create_MyRBF_rb.argtypes = [ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_int32, ctypes.c_int32,
                                    ctypes.c_int32,
                                    ctypes.c_double]
    my_lib.create_MyRBF_rb.restype = ctypes.c_void_p

    my_lib.train_MyRBF_rb.argtypes = [ctypes.c_void_p,
                                    
                                    ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_int32, ctypes.c_int32,
                                    
                                    ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_int32, ctypes.c_int32,

                                    ctypes.c_double,
                                    ctypes.c_int32,
                                    ctypes.c_bool
                                    ]

    my_lib.predict_MyRBF_rb.argtypes = [ctypes.c_void_p,

                                    ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_int32, ctypes.c_int32,

                                    ctypes.c_bool
                                    ]
    my_lib.predict_MyRBF_rb.restype = ctypes.POINTER(ctypes.c_double)

    #####################################

    ############## MyRBF ##############
    my_lib.create_MyRBF.argtypes = [ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_int32, ctypes.c_int32,
                                    
                                    ctypes.c_double]
    my_lib.create_MyRBF.restype = ctypes.c_void_p

    my_lib.train_MyRBF.argtypes = [ctypes.c_void_p,
                                    
                                    ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_int32, ctypes.c_int32,
                                    
                                    ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_int32, ctypes.c_int32]

    my_lib.predict_MyRBF.argtypes = [ctypes.c_void_p,

                                    ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_int32, ctypes.c_int32,

                                    ctypes.c_bool
                                    ]
    my_lib.predict_MyRBF.restype = ctypes.POINTER(ctypes.c_double)

    ############## MyLinearRegression_rb ##############
    my_lib.create_MyLinearRegression_rb.argtypes = [ctypes.c_int32, ctypes.c_int32]
    my_lib.create_MyLinearRegression_rb.restype = ctypes.c_void_p

    my_lib.train_MyLinearRegression_rb.argtypes = [ctypes.c_void_p,
                                                
                                                ctypes.POINTER(ctypes.c_double), 
                                                ctypes.c_int32, ctypes.c_int32,
                                                
                                                ctypes.POINTER(ctypes.c_double), 
                                                ctypes.c_int32, ctypes.c_int32,
                                                
                                                ctypes.c_double, 
                                                ctypes.c_int32
    ]


    my_lib.predict_MyLinearRegression_rb.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_int32  # Input data
    ]
    my_lib.predict_MyLinearRegression_rb.restype = ctypes.POINTER(ctypes.c_double)
    #####################################


    ############## MyLinearRegression ##############
    my_lib.create_MyLinearRegression.argtypes = []
    my_lib.create_MyLinearRegression.restype = ctypes.c_void_p

    my_lib.train_MyLinearRegression.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32, ctypes.c_int32
    ]

    my_lib.predict_MyLinearRegression.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_int32
    ]
    my_lib.predict_MyLinearRegression.restype = ctypes.POINTER(ctypes.c_double)
    #####################################

    ############## My_MLP ##############

    # Définition des types d'arguments et de retour pour la fonction create_MyMLP
    my_lib.create_MyMLP.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    my_lib.create_MyMLP.restype = ctypes.c_void_p

    # Définition des types d'arguments et de retour pour la fonction train_MyMLP
    my_lib.train_MyMLP.argtypes = [ctypes.c_void_p, 
                                
                                ctypes.POINTER(ctypes.c_double), #X_train 
                                ctypes.c_int32, ctypes.c_int32,
                                
                                ctypes.POINTER(ctypes.c_double), #y_train 
                                ctypes.c_int32, ctypes.c_int32,
                                
                                ctypes.POINTER(ctypes.c_double), #X_test 
                                ctypes.c_int32, ctypes.c_int32,
                                
                                ctypes.POINTER(ctypes.c_double), #y_test 
                                ctypes.c_int32, ctypes.c_int32,
                                
                                ctypes.c_int32,
                                ctypes.c_double,
                                ctypes.c_int32,
                                ctypes.c_bool]


    my_lib.predict_MyMLP.argtypes = [ctypes.c_void_p,
                                    
                                    ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_int32, ctypes.c_int32,
                                    
                                    ctypes.c_bool]

    my_lib.predict_MyMLP.restype = ctypes.POINTER(ctypes.c_double)


    my_lib.save_MyMLP.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]

    my_lib.load_MyMLP.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    my_lib.load_MyMLP.restype = ctypes.c_void_p


    my_lib.get_train_losses.argtypes = [ctypes.c_void_p]
    my_lib.get_train_losses.restype = ctypes.POINTER(ctypes.c_double)

    my_lib.get_test_losses.argtypes = [ctypes.c_void_p]
    my_lib.get_test_losses.restype = ctypes.POINTER(ctypes.c_double)
    
    print("Librairie!!!")
    return my_lib

my_lib = load_my_lib()

def create_MyMLP(npl):
    npl_len = len(npl)
    c_npl = (ctypes.c_int32 * npl_len)(*npl)
    model = my_lib.create_MyMLP(c_npl, npl_len)
    return model

def train_MyMLP(model, X_train, y_train, X_test, y_test, learning_rate, nb_iter, is_classification):
    X_train_shape_0 = len(X_train)
    X_train_shape_1 = len(X_train[0])
    X_train = np.array(X_train, np.float64).flatten()
    c_X_train = np.ctypeslib.as_ctypes(X_train)

    y_train_shape_0 = len(y_train)
    y_train_shape_1 = len(y_train[0])
    y_train = np.array(y_train, np.float64).flatten()
    c_y_train = np.ctypeslib.as_ctypes(y_train)

    X_test_shape_0 = len(X_test)
    X_test_shape_1 = len(X_test[0])
    X_test = np.array(X_test, np.float64).flatten()
    c_X_test = np.ctypeslib.as_ctypes(X_test)

    y_test_shape_0 = len(y_test)
    y_test_shape_1 = len(y_test[0])
    y_test = np.array(y_test, np.float64).flatten()
    c_y_test = np.ctypeslib.as_ctypes(y_test)

    my_lib.train_MyMLP(model,
                       c_X_train,
                       X_train_shape_0, X_train_shape_1,
                       
                       c_y_train,
                       y_train_shape_0, y_train_shape_1,
                       
                       c_X_test,
                       X_test_shape_0, X_test_shape_1,
                       
                       c_y_test,
                       y_test_shape_0, y_test_shape_1,
                       
                       learning_rate,
                       nb_iter,
                       is_classification)

def predict_MyMLP(model, input, pred_size, is_classification):
    input_shape_0 = len(input)
    input_shape_1 = len(input[0])
    input = np.array(input, np.float64).flatten()
    c_input = np.ctypeslib.as_ctypes(input)

    p_predictions = my_lib.predict_MyMLP(model, c_input,
                                         input_shape_0, input_shape_1,
                                         is_classification)

    predictions = np.ctypeslib.as_array(p_predictions, (input_shape_0 * pred_size,)).reshape(input_shape_0, pred_size)
    return predictions

def save_MyMLP(model, path):
    path_bytes = path.encode('utf-8')
    c_path = (ctypes.c_uint8 * len(path_bytes))(*path_bytes)
    my_lib.save_MyMLP(model, c_path, len(path_bytes))

def load_MyMLP(path):
    path_bytes = path.encode('utf-8')
    c_path = (ctypes.c_uint8 * len(path_bytes))(*path_bytes)
    model = my_lib.load_MyMLP(c_path, len(path_bytes))
    return model

def get_train_losses(model):
    p_losses = my_lib.get_train_losses(model)
    return np.ctypeslib.as_array(p_losses, shape=(nb_iter,))

def get_test_losses(model):
    p_losses = my_lib.get_test_losses(model)
    return np.ctypeslib.as_array(p_losses, shape=(nb_iter,))