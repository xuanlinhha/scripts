'''
import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()
'''
#

import numpy as np
import matplotlib.pyplot as plt

COINS = 1000
FLIPS = 10
EXPERIMENTS = 100000

def exp():
    nu1 = []
    nu2 = []
    nu3 = []
    for i in range(EXPERIMENTS):
        # Head = 1, Tail = 0
        # each row is a data for each coin
        data = np.random.randint(2, size=(COINS, FLIPS))
        fre = np.sum(data, axis=1)
        c2 = np.random.randint(0, COINS) # choose random coin
        c3 = np.argmin(fre) # the coin has minimum Head
        # Ein with coin c1, c2, c3
        nu1.append(fre[0] / FLIPS)
        nu2.append(fre[c2] / FLIPS)
        nu3.append(fre[c3] / FLIPS)
    return np.array(nu1), np.array(nu2), np.array(nu3)

# plot histogram data
def histogram_plot(nu1, nu2, nu3):
    plt.hist(nu1, histtype='stepfilled', color='r', label='First coin')
    plt.hist(nu2, histtype='stepfilled', color='g', label='Random coin')
    plt.hist(nu3, histtype='stepfilled', color='b', label='Minimum Fraction coin')
    plt.title("Fraction of heads")
    plt.xlabel("Fraction")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# plot epsilon with propability
def pro_bound_plot(nu1, nu2, nu3):
    es = np.arange(0, 1.1, 0.1)
    t1 = []
    t2 = []
    t3 = []
    b = []
    for e in es:
        # count number of nu in each nu array
        t1.append((abs(nu1 - 0.5) > e).sum() / EXPERIMENTS)
        t2.append((abs(nu2 - 0.5) > e).sum() / EXPERIMENTS)
        t3.append((abs(nu3 - 0.5) > e).sum() / EXPERIMENTS)
        b.append(2 * np.exp(-2 * (e ** 2) * FLIPS))

    plt.plot(es, t1, marker='.', color='g', label='Firt coin')
    plt.plot(es, t2, marker='.', color='y', label='Random coin')
    plt.plot(es, t3, marker='.', color='b', label='Minimum Fraction coin')
    plt.plot(es, b, marker='.', color='r', label='Hoeffding bound (N=10)')
    plt.title("Epsilon vs P[|nu - mu| > epsilon]")
    plt.xlabel("epsilon")
    plt.ylabel("Propability")
    plt.legend()
    plt.show()

nu1, nu2, nu3 = exp()
histogram_plot(nu1, nu2, nu3)
pro_bound_plot(nu1, nu2, nu3)

#
'''
pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
'''