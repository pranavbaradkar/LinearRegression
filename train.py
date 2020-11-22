import numpy as np 
import csv
def get_data():
    X = np.genfromtxt("train_X_lr.csv",delimiter=",",dtype = np.float64,skip_header = 1)
    Y = np.genfromtxt("train_Y_lr.csv",delimiter=",",dtype = np.float64)
    return X,Y 
def compute_cost(X, Y, W):
    y_pred = np.dot(X,W)
    mse = np.sum(np.square(y_pred-Y))
    cost_value = mse/(2*len(X))
    print(cost_value)
    
    
def compute_gradient_of_cost_function(X, Y, W):
    m = len(X)
    Y_pred = np.dot(X, W)
    difference =  Y_pred - Y
    dW = (1/m) * (np.dot(difference.T, X))
    dW = dW.T

    return dW
def optimize_weights_using_gradient_descent(X, Y, W, num_iterations, learning_rate):
    for i in range(num_iterations):
        compute_cost(X,Y,W)
        W = W - learning_rate * compute_gradient_of_cost_function(X,Y,W)
    return W
def train_data(X,Y):
    X = np.insert(X,0,1,axis = 1)
    Y = Y.reshape(len(X),1)
    W = np.zeros((X.shape[1],1))
    inv = np.linalg.inv(np.dot(X.T,X))
    pro = np.dot(inv,X.T)
    weights = np.dot(pro,Y)
    #weights = optimize_weights_using_gradient_descent(X, Y, W, 10000000, 0.0002)
    #weights = np.round(weights,3).tolist()
    return weights
def save_model(w,weights_file_name):
    with open (weights_file_name,'w') as weights_file : 
        wr = csv.writer(weights_file)
        wr.writerows(w) 
        weights_file.close()




if __name__ == "__main__":
    X,Y = get_data()
    W = train_data(X,Y)
    save_model(W,"WEIGHTS_FILE.csv")

    

    