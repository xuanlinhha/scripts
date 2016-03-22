import numpy as np
import matplotlib.pyplot as plt

class HeoffdingIneq:
    """ Experiment with coins """
    def __init__(self, coins, flips, experiments):
        # number of coins, flips and experiments
        self.coins = coins
        self.flips = flips
        self.experiments = experiments
        # run experiments
        self.run_experiments()

    def run_experiments(self):
        n1 = []
        n2 = []
        n3 = []
        for i in range(self.experiments):
            # Head = 1, Tail = 0
            # Each row is a data for each coin
            data = np.random.randint(2, size=(self.coins, self.flips))
            # Head frequency of each coin
            head_frac = np.sum(data, axis=1)
            c2 = np.random.randint(0, self.coins) # choose random coin
            c3 = np.argmin(head_frac) # the coin has minimum Head
            # Ein with coin c1, c2, c3 (c1 is the first coin)
            n1.append(head_frac[0] / self.flips)
            n2.append(head_frac[c2] / self.flips)
            n3.append(head_frac[c3] / self.flips)
        # save fraction of head of 3 selected coins
        self.nu1 = np.array(n1)
        self.nu2 = np.array(n2)
        self.nu3 = np.array(n3)

    def plot_histogram(self):
        plt.hist(self.nu1, histtype='stepfilled', color='r', label='First coin')
        plt.hist(self.nu2, histtype='stepfilled', color='g', label='Random coin')
        plt.hist(self.nu3, histtype='stepfilled', color='b', label='Minimum Fraction coin')
        plt.title("Fraction of heads")
        plt.xlabel("Fraction")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    def prob_bound_plot(self):
        eps = np.arange(0, 1.1, 0.1)
        t1 = []
        t2 = []
        t3 = []
        b = []
        for ep in eps:
            # probability of bad event = number of experiments in which bad event happens over the total
            # (miu is 0.5 because of fair coins)
            t1.append((abs(self.nu1 - 0.5) > ep).sum() / self.experiments)
            t2.append((abs(self.nu2 - 0.5) > ep).sum() / self.experiments)
            t3.append((abs(self.nu3 - 0.5) > ep).sum() / self.experiments)
            # inequality bound
            b.append(2 * np.exp(-2 * (ep ** 2) * self.flips))
        plt.plot(eps, t1, marker='.', color='g', label='Firt coin')
        plt.plot(eps, t2, marker='.', color='y', label='Random coin')
        plt.plot(eps, t3, marker='.', color='b', label='Minimum Fraction coin')
        plt.plot(eps, b, marker='.', color='r', label='Hoeffding bound (N=10)')
        plt.title("Epsilon vs P[|nu - mu| > epsilon]")
        plt.xlabel("epsilon")
        plt.ylabel("Propability")
        plt.legend()
        plt.show()

hieq = HeoffdingIneq(1000, 10, 100000)
print(np.average(hieq.nu3))
hieq.plot_histogram()
hieq.prob_bound_plot()
