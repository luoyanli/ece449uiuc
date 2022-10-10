from cmath import exp
import torch
import hw2_utils as utils
import matplotlib.pyplot as plt

'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

'''

# Problem Linear Regression
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float, default: 0.01): learning rate
        num_iter (int, default: 1000): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    ##add column of 1 to X
    X = torch.cat((torch.ones(X.shape[0]).reshape(-1,1), X),1)
    ## now X should be n* (d+1)
    ## initiate w
    w = torch.zeros(X.shape[1])
    N= X.shape[0]
    a = 0 
    for i in range(num_iter):
        a = torch.sum(2*(torch.matmul(X,w.unsqueeze(-1))-Y)*X,0)/N
        w = w - lrate*a 
    # print(w)
    # print(a)
    return w
    # pass

    

    


def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    X = torch.cat((torch.ones(X.shape[0]).reshape(-1,1), X),1)
    w = torch.pinverse(X)@Y
    # print(w)
    w = w.squeeze(-1)
    return w
    


def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    X,Y = utils.load_reg_data()
    w = linear_normal(X,Y)
    X_i = torch.cat((torch.ones(X.shape[0]).reshape(-1,1), X),1)
    plt.figure()
    plt.title('Visualization of the dataset and regression results')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(X,Y,'bo')
    plt.plot(X, torch.matmul(X_i,w.unsqueeze(-1)),'r')
    plt.legend()
    plt.savefig('plot_linear.png')
    return plt.gcf()
    




# Problem Logistic Regression
def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float, default: 0.01): learning rate
        num_iter (int, default: 1000): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    X = torch.cat((torch.ones(X.shape[0]).reshape(-1,1), X),1)
    ## now X should be n* (d+1)
    ## initiate w
    w = torch.zeros(X.shape[1])
    N= X.shape[0]
    # print(X.shape)
    a = 0 
    b = 0 
    # c = 0
    for i in range(num_iter):
        b =1/(1+torch.exp(-Y*(torch.matmul(X,w.unsqueeze(-1))))) * torch.exp(-Y*(torch.matmul(X,w.unsqueeze(-1))))
        a = torch.sum(b*X*-Y,0)/N
        ## loss function here
        # c = torch.sum(torch.log(1+torch.exp(-Y*(torch.matmul(X,w.unsqueeze(-1))))),0)/N
        w = w - lrate*a 
    # print(w)
    # print(a)
    
    return w


def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    X,Y = utils.load_logistic_data()
    w_lo = logistic(X,Y)
    w_li = linear_gd(X,Y)
    Y_flat = Y.flatten()
    X_pos = X[torch.where(Y_flat==1.)]
    X_neg = X[torch.where(Y_flat==-1.)]

    X2_lo = -(w_lo[0] + w_lo[1] * X[:,0])/w_lo[2]
    X2_li = -(w_li[0] + w_li[1] * X[:,0])/w_li[2]

    plt.figure()
    plt.title('Logistic_vs_linear')
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.scatter(X_pos[:,0], X_pos[:,1], label = 'y = 1')
    plt.scatter(X_neg[:,0], X_neg[:,1], label = 'y = -1')
    plt.plot(X[:,0],X2_li,label ='linear',color = 'r')
    plt.plot(X[:,0],X2_lo,label ='logistic',color = 'y')
    plt.legend()
    plt.savefig('plot_lo_vs_li.png')
    # plt.show()
    
    
    return plt.gcf()




plot_linear()
logistic_vs_ols()
plt.show()

# X,Y = utils.load_logistic_data()
# w = logistic(X,Y)