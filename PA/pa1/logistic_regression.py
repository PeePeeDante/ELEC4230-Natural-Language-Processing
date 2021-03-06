import numpy as np # linear algebra
from sklearn.model_selection import train_test_split
import time

from preprocess import data_preprocess, feature_extraction_bow, normalization

def sigmoid(z):
    """
    TODO: Implement sigmoid function. s is a vector with the size as z.
    """
    from math import e
    s = 1/(1+e**z)
    return s

def initialize_w_and_b(dimension):
    w = np.zeros((dimension, 1))
    b = 0
    return w, b

def forward_prop(X, w, b):
    """
    Do the forward computation and return f(x)= \sigma (Wx+b)
    """
    # w.T: (1,d)
    # X: (d,m)
    #print("forward_prop()")
    #print("w.shape", w.shape)
    return sigmoid(np.dot(w.T, X) + b) # (1,m)

def compute_loss(A, Y, m):
    """
    TODO:
        Compute the loss function based on the formula you derived.
        loss is a scalar
        Hint:
            1) The formula should be (-1.0 / m) * np.sum(...)
    """
    # A = sigmoid(np.dot(w.T, X) + b) size: (1,m)
    # X size: (d,m) Y size: (1,m)
    loss = -(1.0/m) * (Y @ np.log(A).T + (1-Y) @ np.log(1-A).T)
    
    #print("compute_loss")
    #print("loss: ", loss.shape)
    return loss

def back_prop(X, A, Y, m):
    """
    TODO:
        Compute the gradient based on the formula you derived.
        dw and db are two scalars.
        Hint:
            1) The formula of dw should be (1.0 / m) * np.dot(...)
            2) The formula of db should be (1.0 / m) * np.sum(...)
    """
    # A = sigmoid(np.dot(w.T, X) + b) size: (1,m)
    # X size: (d,m) Y size: (1,m)
    #print("back_prop()")
    #print("X: ", X.shape)
    #print("Y: ", Y.shape)
    #print("A: ", A.shape)
    
    db = (1.0 / m) * np.sum(Y-A)
    dw = (1.0 / m) * (np.dot(Y-A,X.T)).T
    
    #print("dw: ", dw.shape)
    #print("db shape: ", db.shape)
    #print("db: ", db)
    return {"dw": dw, "db": db}

def optimize(w, b, X, Y, X_dev, Y_dev, num_iterations, learning_rate, output_name):
    m = X.shape[1] # m is the number of the samples
    max_acc = 0
    max_w, max_b = w, b
    start_time = time.time()
    log = open(output_name + '.log', 'w')
    log.write('iteration, train acc, dev acc\n')

    for i in range(num_iterations):
        f_x = forward_prop(X, w, b)
        cost = compute_loss(f_x, Y, m)
        grads = back_prop(X, f_x, Y, m)

        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]

        Y_prediction_train = predict(w, b, X)
        Y_prediction_dev = predict(w, b, X_dev)
        train_acc = compare(Y_prediction_train, Y)
        dev_acc = compare(Y_prediction_dev, Y_dev)
        log.write('{},{},{}\n'.format(str(i + 1), str(train_acc), str(dev_acc)))

        if dev_acc > max_acc: # keep the best parameters
            max_acc = dev_acc
            max_w, max_b = w, b

        print('iteration:', i + 1, ", time {0:.2f}", time.time() - start_time)
        print("\tTraining accuracy: {0:.4f} %, cost: {0:.4f}".format(train_acc, cost))
        print("\tDev accuracy: {0:.4f} %".format(dev_acc))

    params = {"w": max_w, "b": max_b}
    return params

def predict(w, b, X):
    """
    TODO:
        Predict the sentiment class based on the f(x) value.
        if f(x) > 0.5, then pred value is 1, otherwise is 0.
        Y_prediction is a 2-D array with the size (1*nb_sentence)
    """
    m = X.shape[1] # X size (d,m)
    Y_prediction = np.zeros((1, m))
    
    i=0
    for vec in X.T:
        vec.reshape(1,-1)
        #print(vec.shape)
        #print(w.shape)
        #print(forward_prop(vec,w,b).shape)
        if (forward_prop(vec,w,b) > 0.5):
            Y_prediction[0][i]=1
        else:
            Y_prediction[0][i]=0
        i+=1
        
    '''
    # (m,d)
    # vec size (1,d)
    i = 0
    for vec in X.T:
		print(vec.shape)
		print(forward_prop(vec,w,b).shape)
        if (forward_prop(vec,w,b) > 0.5):
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0
        i += 1
    '''
    return Y_prediction
    

def compare(pred, gold):
    """
    TODO:
        Compute the accuracy based on two array, pred and gold, and return a scalar between [0, 100]
    """
    m = pred.shape[1]
    correct = 0
    for i in range(m):
        if (pred[0][i] == gold[0][i]):
            correct += 1
    acc = correct/m * 100
    
    return acc

def write_testset_prediction(parameters, test_data, file_name="myPrediction.csv"):
    Y_prediction_test = predict(parameters['w'], parameters['b'], test_data)
    f_pred = open(file_name, 'w')
    f_pred.write('ID\tSentiment')
    ID = 1
    for pred in Y_prediction_test[0]:
        sentiment_pred = 'pos' if pred == 1 else 'neg'
        f_pred.write(str(ID) + ',' + sentiment_pred + '\n')
        ID += 1
        
        
# model(X_train.T, Y_train.T, X_dev.T, Y_dev.T, args['file_name'], num_iterations=int(args['num_iter']), learning_rate=float(args['learning_rate']))
def model(X_train, Y_train, X_dev, Y_dev, output_name, num_iterations=100, learning_rate=0.005):
    w, b = initialize_w_and_b(X_train.shape[0])
    # w size: (d,1)
    # b size: (1)

    parameters = optimize(w, b, X_train, Y_train, X_dev, Y_dev, num_iterations, learning_rate, output_name)
    # X_train = X.T (d,m)
    # Y_train = Y.T (1,m)

    Y_prediction_dev = predict(parameters["w"], parameters["b"], X_dev)
    print("Best dev accuracy: {} %".format(compare(Y_prediction_dev, Y_dev)))

    np.save(output_name + '.npy', parameters)

    return parameters


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Building Interactive Intelligent Systems')
    parser.add_argument('-c', '--clean', help='True to do data cleaning, default is False', action='store_true')
    parser.add_argument('-mv', '--max_vocab', help='max vocab size predifined, no limit if set -1', required=False, default=-1)
    parser.add_argument('-lr', '--learning_rate', required=False, default=0.1)
    parser.add_argument('-i', '--num_iter', required=False, default=200)
    parser.add_argument('-fn', '--file_name', help='file name', required=False, default='myTest')
    args = vars(parser.parse_args())
    print(args)

    print('[Read the data from twitter-sentiment.csv...]')
    revs, word2idx = data_preprocess('./twitter-sentiment.csv', args['clean'], int(args['max_vocab']))

    print('[Extract features from the read data...]')
    data1, label = feature_extraction_bow(revs, word2idx)
    data = normalization(data1)

    # shuffle data
    shuffle_idx = np.arange(len(data))
    np.random.shuffle(shuffle_idx)
    data = data[shuffle_idx]
    label = label[shuffle_idx]
    
    print('[Start training...]')
    X_train, X_dev, Y_train, Y_dev = train_test_split(data, label, test_size=0.2, random_state=0)
    parameters = model(X_train.T, Y_train.T, X_dev.T, Y_dev.T, args['file_name'], num_iterations=int(args['num_iter']), learning_rate=float(args['learning_rate']))
    # added line
    print("[writing training_result.csv...]")
    write_testset_prediction(parameters, data1.T, 'training_result.csv')
    
    print('\n[Start evaluating on the official test set and dump as {}...]'.format(args['file_name'] + '.csv'))
    revs, _ = data_preprocess("./twitter-sentiment-testset.csv", args['clean'], int(args['max_vocab']))
    test_data, _ = feature_extraction_bow(revs, word2idx)
    write_testset_prediction(parameters, test_data.T, args['file_name'] + '.csv')
