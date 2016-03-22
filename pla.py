'''
Uncomment this part and the part at the ending to see performance

import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()
'''

import numpy as np
import matplotlib.pyplot as plt

class PLA:
    ''' Perceptron learning algorithm demonstration '''
    def __init__(self, training_size, sample_size):
        self.training_size = training_size
        self.sample_size = sample_size
        # TV is the target function: [A, B, C] of the line Ax + By + C = 0
        x1,y1,x2,y2 = np.random.uniform(-1,1,4)
        self.TV = np.array([x1-x2, y2-y1, x2*y1-x1*y2])
        # generate classified data
        self.gen_training_data()
        self.run_pla()
        self.cal_error_out()
        
    def gen_training_data(self):
        # each row is in form: [1,x,y,+1/-1]
        self.training_data = np.random.uniform(-1, 1, (self.training_size, 4))
        self.training_data[:, 0] = 1
        self.training_data[:, 3] = 1
        for d in self.training_data:
            d[3] = np.int(np.sign(np.dot(self.TV, d[1:4])))
    
    def run_pla(self):
        # PV is final hypothesis
        self.PV = np.zeros(3)
        self.iterations = 0
        while True:
            mis_pos = []
            # get indexs of misclassified points
            for idx, d in enumerate(self.training_data):
                if np.int(np.sign(np.dot(self.PV, d[0:3]))) != d[3]:
                    mis_pos.append(idx)
            # if all are correctly classified then quit while
            # else choose a randomly misclassified point and update PV [w0,w1,w2]
            if not mis_pos:
                break
            else:
                d = self.training_data[np.random.choice(mis_pos)]
                self.PV = self.PV + d[0:3] * d[3]
                self.iterations += 1
    
    def cal_error_out(self):
        # create a large sample data
        self.sample_data = np.random.uniform(-1, 1, (self.sample_size, 4))
        self.sample_data[:, 0] = 1
        self.sample_data[:, 3] = 1
        for d in self.sample_data:
            d[3] = np.int(np.sign(np.dot(self.TV, d[1:4])))
        # calculate P(g(x) != f(x)) on sample data
        count = 0
        for d in self.sample_data:
            if np.int(np.sign(np.dot(self.PV, d[0:3]))) != d[3]:
                count += 1
        self.err_out = count / self.sample_size

    def plot_data(self):
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        # plot the target function and final hypothesis
        xs = np.linspace(-1, 1, num=50)
        plt.plot(xs,(-xs*self.TV[0]-self.TV[2])/self.TV[1], c='r') # Ax + By + C = 0
        plt.plot(xs,(-self.PV[0]-xs*self.PV[1])/self.PV[2], c='m') # A + Bx + Cy = 0
        # plot training data
        plt.scatter(self.training_data[:,1], self.training_data[:,2], s=60, c=self.training_data[:,3], cmap='brg')
        # plot sample data
        plt.scatter(self.sample_data[:,1], self.sample_data[:,2], s=10, c=self.sample_data[:,3], cmap='brg')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

def plot():
    pla = PLA(10,1000)
    pla.plot_data()
plot()


def experiments():
    iters = []
    eout = []
    for i in range(1000):
        pla = PLA(100, 1000)
        iters.append(pla.iterations)
        eout.append(pla.err_out)
    print(np.average(iters))
    print(np.average(eout))
experiments()


'''
pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
'''