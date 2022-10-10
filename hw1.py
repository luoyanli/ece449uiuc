import torch
import hw1_utils as utils



# Problem Naive Bayes
def bayes_MAP(X, y):
    '''
    Arguments:
        X (N x d LongTensor): features of each object, X[i][j] = 0/1
        y (N LongTensor): label of each object, y[i] = 0/1

    Returns:
        theta (2 x d Float Tensor): MAP estimation of P(X_j=1|Y=i)

    '''
    return torch.stack([torch.matmul(torch.transpose(X,0,1),1-y)/torch.sum(1-y), torch.matmul(torch.transpose(X,0,1),y)/torch.sum(y)])

def bayes_MLE(y):
    '''
    Arguments:
        y (N LongTensor): label of each object

    Returns:
        p (float or scalar Float Tensor): MLE of P(Y=0)

    '''
    return 1-(torch.sum(y)/y.shape[0])

def bayes_classify(theta, p, X):
    '''
    Arguments:
        theta (2 x d Float Tensor): returned value of `bayes_MAP`
        p (float or scalar Float Tensor): returned value of `bayes_MLE`
        X (N x d LongTensor): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (N LongTensor): label of each object for classification, y[i] = 0/1
    
    '''
    pos_0= torch.sum(torch.log(X*theta[0]+(1-X)*(1-theta[0])),1)+torch.log(p)
    pos_1= torch.sum(torch.log(X*theta[1]+(1-X)*(1-theta[1])),1)+torch.log(1-p)
    # a= torch.sum(torch.log(X*theta[1]+(1-X)*(1-theta[1])),1)
    # b = torch.log(1-p)
    # print(a.shape,b.shape)

    y=torch.argmax(torch.concat([pos_0.unsqueeze(-1),pos_1.unsqueeze(-1)],dim=1),dim=1)
    # y = torch.tensor((pos_0<pos_1)+0)
    # print(y)
    # print(X.shape)
    return y
    # pass
    

# Problem Gaussian Naive Bayes
def gaussian_MAP(X, y):
    '''
    Arguments:
        X (N x d FloatTensor): features of each object
        y (N LongTensor): label of each object, y[i] = 0/1

    Returns:
        mu (2 x d Float Tensor): MAP estimation of mu in N(mu, sigma2)
        sigma2 (2 x d Float Tensor): MAP estimation of mu in N(mu, sigma2)

    '''
    ##class0 mu0, sigma0
    mu0 = torch.matmul(torch.transpose(X,0,1),1-y.float())/torch.sum(1-y.float())
    mu00 = mu0.unsqueeze(-1)

    sigma0 = torch.matmul((torch.transpose(X,0,1)-mu00)**2,1-y.float())/torch.sum(1-y.float())

    ## class1 mu1, sigma1
    mu1= torch.matmul(torch.transpose(X,0,1),y.float())/torch.sum(y.float())
    mu11 = mu1.unsqueeze(-1)
    sigma1= torch.matmul((torch.transpose(X,0,1)-mu11)**2,y.float())/torch.sum(y.float())
    return torch.stack([mu0,mu1]), torch.stack([sigma0, sigma1])

def gaussian_MLE(y):
    '''
    Arguments:
        y (N LongTensor): label of each object

    Returns:
        p (float or scalar Float Tensor): MLE of P(Y=0)

    '''
    return 1-(torch.sum(y)/y.shape[0])

def gaussian_classify(mu, sigma2, p, X):
    '''
    Arguments:
        mu (2 x d Float Tensor): returned value #1 of `gaussian_MAP`
        sigma2 (2 x d Float Tensor): returned value #2 of `gaussian_MAP`
        p (float or scalar Float Tensor): returned value of `bayes_MLE`
        X (N x d FloatTensor): features of each object

    Returns:
        y (N LongTensor): label of each object for classification, y[i] = 0/1
    
    '''
    ## pos_0, pos_1 are all N FloatTansor
    pos_0=  torch.sum(-0.5*torch.log(2*torch.pi*sigma2[0])-0.5*((X-mu[0])/sigma2[0]**0.5)**2,1) +torch.log(p)
    pos_1=  torch.sum(-0.5*torch.log(2*torch.pi*sigma2[1])-0.5*((X-mu[1])/sigma2[1]**0.5)**2,1) +torch.log(1-p)
    # pos_0= -0.5*torch.log(2*torch.pi*sigma2[0]) 
    # pos_0= torch.log(p)
    # print(pos_0.shape, pos_1.shape)
    y=torch.argmax(torch.concat([pos_0.unsqueeze(-1),pos_1.unsqueeze(-1)],dim=1),dim=1)
    # pos_1= torch.sum(torch.log(X*theta[1]+(1-X)*(1-theta[1])),1)+torch.log(1-p)
    # y=torch.argmax(torch.concat([pos_0.unsqueeze(-1),pos_1.unsqueeze(-1)],dim=1),dim=1)
    # y = torch.tensor((pos_0<pos_1)+0)
    # print(y)
    # print(X.shape)
    return y

