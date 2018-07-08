#%%
import numpy as np
import matplotlib.pyplot as plt

#We need to always use seed for fixing random state for reproducibility
#take any arbitrary seed
np.random.seed(123124235)

class perceptron:
    def __init__(self, filename,learn_rate=0.01, num_epochs=25):
        self._learn_rate = learn_rate
        self._epochs = num_epochs
        self._filename = filename
        self._boundary_lines = []
        self._wronglabel = 0
        self._data = { "input" : set(), "label" :set()}
    def _set_inputs_label(self):
        x = self._inputdata[:,:-1]
        y = self._inputdata[:,-1]
        self._data["input"] = x
        self._data["label"] = y

    def _get_data(self):
        self._inputdata = np.genfromtxt(self._filename, delimiter=',')
        self._set_inputs_label()
    
    def _stepFunction(self,t):
        if t >= 0:
            return 1
        return 0
    
    def _prediction(self,X, W, b):
        return self._stepFunction((np.matmul(X,W)+b)[0])
    
    def _update_weights(self,Xi,W,b,oper='+'):
        if oper == '+':
            W[0] += Xi[0]*self._learn_rate
            W[1] += Xi[1]*self._learn_rate
            b += self._learn_rate
            self._wronglabel += 1
        else:
            W[0] -= Xi[0]*self._learn_rate
            W[1] -= Xi[1]*self._learn_rate
            b -= self._learn_rate
            self._wronglabel += 1
        return W, b    


    def _perceptronStep(self,X, y, W, b):
        '''
              n 
           ∑  f(x) ∈ ℝ  where x1 ....xn are real numbers
              i = 1

    Step function
               0 if b + ∑ wi xi < 0
    f(x) = ｛   
               1 if b + ∑ wi xi >= 0

    w is weight and b is bias which is again weight where x0 = 1 so f(x) = ∑ wx ranging from 0 to n+1 

    We have data x1 and x2 and label 1 or 0
    Based on above formula if f(x) >= 0 than set it as 1 else 0
    If Label - f(x) == 0 than we don't need to do anything
    If Label - f(x) == 1 that would mean f(x) = 0 and Label = 1 so we need to increase weight where 
    wj = wj + xij * learning_rate where i ranging from 0 to n input rows 0 to j input columns bias b = w0 x0 = 1
    Else decrease weights 
        '''
        for i in range(len(X)):  
            label_predication = self._prediction(X[i],W,b)

        #if label is 1 i.e positive and prediction was 0 than add else subtract
            difference = y[i]-label_predication 
            if difference == 0: #if label and prediction matches than do nothing
                continue
            W,b = self._update_weights(X[i],W,b,'+') if difference == 1 else self._update_weights(X[i],W,b,'-')
        return W, b

    def _find_boundary_lines(self,W,b):
        '''
        Standard form for a line is Ax+By+C = 0
        Slope = -A/B y-intercept = -C /B

        Lets take first row as example and below can be represented in standard form
        w1x1 + w2x2 + bias = 0 here x1 = 0.780510 x2 = -0.063669  bias,w1,w2 are weights 
        which can be used to calculate slopes and y-intercept to draw boundary lines
        '''
        self._boundary_lines.append((-W[0]/W[1], -b/W[1]))

    def _train_perceptron(self,X, y):
        #For bias we can choose random + min or max of x or y coordinates
        x_min, x_max = min(X.T[0]), max(X.T[0])
        y_min, y_max = min(X.T[1]), max(X.T[1])
        xall_min,xall_max = np.amin(X[:,-1]), np.amax(X[:,-1])
        
        #If input array is m * n matrix than weight should be n * 1 to avoid broadcasting
        xdim, ydim = X.shape[0], X.shape[1]
        W = np.array(np.random.rand(ydim,1))
        b = np.random.rand(1)[0] + y_min #Y min gave least number of weight updates change to different value

        
        for i in range(self._epochs):
            W, b = self._perceptronStep(X, y, W, b,)
            self._find_boundary_lines(W, b)
    
    def plot_data(self,inputs,labels,boundary):
        # fig config
        plt.figure(figsize=(10,6))
        plt.grid(True)

        #plot input samples(2D data points) and i have two classes. 
        #one is 1 and second one is 0, so it red color for 1 and blue color for 0
        for input,label in zip(inputs,labels):
            plt.plot(input[0],input[1],'ro' if (label == 1.0) else 'bo')
        
        count = 0
        #for i in np.linspace(np.amin(inputs[:,:1]),np.amax(inputs[:,:1])):
        for i in np.linspace(np.amin(inputs[:,:1]),np.amax(inputs[:,:1]), num = len(boundary)):
            slope = boundary[count][0][0]
            intercept = boundary[count][1][0]

            #y =mx+c, m is slope and c is intercept
            y = (slope*i) + intercept
            plt.plot(i, y,'ko')
            count += 1
        
        print(count,len(boundary))
            

    def run(self):
        self._get_data()
        self._train_perceptron(self._data["input"],self._data["label"])
        return self._data,self._boundary_lines,self._wronglabel



if __name__ == "__main__":

    plt.style.use('seaborn-whitegrid')

    perceptron_training = perceptron('data.csv')
    data, boundary_lines, total_corrections = perceptron_training.run()
    
    #Total number of inputs which needed weight updates
    print(total_corrections)

    perceptron_training.plot_data(data["input"],data["label"],boundary_lines)


