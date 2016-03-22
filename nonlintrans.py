import numpy as np
import matplotlib.pyplot as plt

class NonlinTrans(object):
    """ Nonlinear transformation """
    def __init__(self, training_size, sample_size):
        self.training_size = training_size
        self.sample_size = sample_size
        self.training_data = self.gen_ori_data(training_size)
        
    def gen_ori_data(self, size):
        # each row is in form: [1,x,y,+1/-1]
        data = np.random.uniform(-1, 1, (size, 4))
        data[:, 0] = 1
        # 90% data is correct, 10% is random
        v = int(0.9 * size)
        for i in range(0, v):
            d = data[i, :]
            d[3] = np.int(np.sign(d[1] * d[1] + d[2] * d[2] - 0.6))
        for i in range(v, size):
            d = data[i, :]
            d[3] = np.random.choice([-1, 1])
        return data

    def gen_trans_data(self, size):
        # each row is in form: [1,x1,x2,x1x2,x1^2,x2^2,+1/-1]
        data = np.random.uniform(-1, 1, (size, 7))
        data[:, 0] = 1
        # 90% data is correct, 10% is random
        v = int(0.9 * size)
        for i in range(0, v):
            d = data[i, :]
            d[3] = d[1] * d[2]
            d[4] = d[1] * d[1]
            d[5] = d[2] * d[2]
            d[6] = np.int(np.sign(d[1] * d[1] + d[2] * d[2] - 0.6))
        for i in range(v, size):
            d = data[i, :]
            d[3] = d[1] * d[2]
            d[4] = d[1] * d[1]
            d[5] = d[2] * d[2]
            d[6] = np.random.choice([-1, 1])
        return data

    def linreg_without_trans(self):
        # X, X dagger matrix and compute w
        X = self.training_data[:, 0:3]
        Xd = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose())
        self.w = np.dot(Xd, self.training_data[:, 3])
        # calculate Ein
        count = 0
        for d in self.training_data:
            if np.int(np.sign(np.dot(self.w, d[0:3]))) != d[3]:
                count += 1
        self.Ein = count / self.training_size

    def linreg_with_trans(self):
        # new training data
        trans_training_data = np.ones((self.training_size, 7))
        np.copyto(trans_training_data[:, 1:3], self.training_data[:, 1:3])
        np.copyto(trans_training_data[:, 6], self.training_data[:, 3])
        for d in trans_training_data:
            d[3] = d[1] * d[2]
            d[4] = d[1] * d[1]
            d[5] = d[2] * d[2]
        # Z, Z dagger matrix and compute wz
        Z = trans_training_data[:, 0:6]
        Zd = np.dot(np.linalg.inv(np.dot(Z.transpose(), Z)), Z.transpose())
        self.wz = np.dot(Zd, trans_training_data[:, 6])
        # calculate number of agreement with 5 hypothesises from the new training data
        h1 = np.array([-1, -0.05, 0.08, 0.13, 1.5, 1.5])
        h2 = np.array([-1, -0.05, 0.08, 0.13, 1.5, 15])
        h3 = np.array([-1, -0.05, 0.08, 0.13, 15, 1.5])
        h4 = np.array([-1, -1.5, 0.08, 0.13, 0.05, 0.05])
        h5 = np.array([-1, -0.05, 0.08, 1.5, 0.15, 0.15])
        self.agree_counts = np.zeros(5)
        for d in trans_training_data:
            if np.sign(np.dot(self.wz, d[0:6])) == np.sign(np.dot(h1, d[0:6])):
                self.agree_counts[0] += 1
            if np.sign(np.dot(self.wz, d[0:6])) == np.sign(np.dot(h2, d[0:6])):
                self.agree_counts[1] += 1
            if np.sign(np.dot(self.wz, d[0:6])) == np.sign(np.dot(h3, d[0:6])):
                self.agree_counts[2] += 1
            if np.sign(np.dot(self.wz, d[0:6])) == np.sign(np.dot(h4, d[0:6])):
                self.agree_counts[3] += 1
            if np.sign(np.dot(self.wz, d[0:6])) == np.sign(np.dot(h5, d[0:6])):
                self.agree_counts[4] += 1
        # generate new sample data to estimate Eout
        sample_data = self.gen_trans_data(self.sample_size)
        count = 0
        for d in sample_data:
            if np.int(np.sign(np.dot(self.wz, d[0:6]))) != d[6]:
                count += 1
        self.Eout = count / self.sample_size

    def plot_data(self):
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        # plot the target function: y = x1^2 + x2^2 - 0.6
        plt.axes()
        circle = plt.Circle((0, 0), np.sqrt(0.6), color='r')
        circle.set_facecolor('none')
        plt.gca().add_patch(circle)
        # plot initial linear regression
        xs = np.linspace(-1, 1, num=50)
        plt.plot(xs,(-self.w[0]-xs*self.w[1])/self.w[2], c='m') # A + Bx + Cy = 0
        # plot training data
        plt.scatter(self.training_data[:,1], self.training_data[:,2], s=60, c=self.training_data[:,3], cmap='brg')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

def plot_data():
    nonlin = NonlinTrans(1000, 1000)
    nonlin.linreg_without_trans()
    nonlin.plot_data()
plot_data()

def calculate_ein_without_trans():
    ein = []
    for x in range(1000):
        nonlin = NonlinTrans(1000, 1000)
        nonlin.linreg_without_trans()
        ein.append(nonlin.Ein)
    print(np.average(ein))
calculate_ein_without_trans()

def calculate_eout_with_trans():
    eout = []
    acounts = np.zeros(5)
    for x in range(1000):
        nonlin = NonlinTrans(1000, 1000)
        nonlin.linreg_with_trans()
        acounts = (acounts + nonlin.agree_counts) / 2
        eout.append(nonlin.Eout)
    print(acounts)
    print(np.average(eout))
calculate_eout_with_trans()

