import random
import numpy as np
from pyESN import ESN
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from mackeyglass import MackeyGlass as Mac

range_list = [100, 1000, 10000, 100000]
for data_range in range_list:
    mac = Mac(tau=17)
    offset = random.randint(0, 1000)
    half_range = data_range / 2
    X, Y = mac.generateData(offset, offset + data_range)
    X = np.asarray(X)
    Y = np.asarray(Y)
    trainX = X[:half_range]
    trainY = Y[:half_range]
    testX = X[half_range:]
    testY = Y[half_range:]

    ne_list = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for ne in ne_list:
        print("neurons num: %d" % (ne))
        esn = ESN(n_inputs = 5, n_outputs = 1, n_reservoir = ne, spectral_radius = 1.5, random_state=42)
        esn_train = esn.fit(trainX, trainY)
        esn_pred = esn.predict(testX)
        print("esn error: \n"+str(np.sqrt(np.mean((esn_pred.flatten() - testY)**2))))

        mlp = MLPRegressor(alpha=0.1, hidden_layer_sizes = (ne,), max_iter=1000,
            activation = 'logistic', learning_rate = 'constant', learning_rate_init = 0.01)
        mlp_train = mlp.fit(trainX, trainY)
        mlp_pred = mlp.predict(testX)
        print("mlp error: \n"+str(np.sqrt(np.mean((mlp_pred.flatten() - testY)**2))))
        
        sgd = MLPRegressor(alpha=0.1, hidden_layer_sizes = (ne,), max_iter=1000, solver = 'sgd',
            activation = 'logistic', learning_rate = 'constant', learning_rate_init = 0.01)
        sgd_train = sgd.fit(trainX, trainY)
        sgd_pred = sgd.predict(testX)
        print("mlp(sgd) error: \n"+str(np.sqrt(np.mean((sgd_pred.flatten() - testY)**2))))
    
        relu = MLPRegressor(alpha=0.1, hidden_layer_sizes = (ne,), max_iter=1000,
            activation = 'relu', learning_rate = 'constant', learning_rate_init = 0.01)
        relu_train = relu.fit(trainX, trainY)
        relu_pred = relu.predict(testX)
        print("mlp(relu) error: \n"+str(np.sqrt(np.mean((relu_pred.flatten() - testY)**2))))
