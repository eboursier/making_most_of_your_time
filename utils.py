import numpy as np

# Define ride distributions


class distribution():
    """
    Superclass for different types of reward function
    """

    def __init__(self, name):
        self.name = name


class uniformRide(distribution):
    """
    Uniform distribution class. Initialize with range of distribution.
    minval: minimum possible value for the distribution (must be non negative)
    maxval: maximum possible value for the distribution (must be >= minval)
    discrete: whether we use a discrete or continuous distribution
    K: size of support if discrete
    """

    def __init__(self, minval=0, maxval=3, discrete=False, K=10):
        super().__init__('uniform')
        if minval < 0:
            raise ValueError('minval must be non negative')
        elif maxval < minval:
            raise ValueError('maxval must be >= minval')
        self.minval = minval
        self.maxval = maxval
        self.discrete = discrete
        if self.discrete:
            self.K = K
            self.support = np.linspace(minval, maxval, K)

    def simu(self, nsamples=1):
        """
        return nsamples samples of the distribution
        """
        if self.discrete:
            S = np.random.randint(0, self.K, size=nsamples)
            return self.support[S]
        else:
            return np.random.uniform(low=self.minval,
                                     high=self.maxval, size=nsamples)


class uniformNoise(distribution):
    """
    Uniform distribution class. Initialize with range of distribution.
    ran: distrib is uniform in [-ran, ran]
    """

    def __init__(self, ran):
        super().__init__('uniform')
        self.range = ran
        self.var = ran*ran/4 # subgaussian parameter

    def simu(self, nsamples=1):
        """
        return nsamples samples of the distribution
        """
        return np.random.uniform(low=-self.range,
                                 high=self.range, size=nsamples)


class gaussianNoise(distribution):
    """
    Uniform distribution class. Initialize with range of distribution.
    ran: distrib is uniform in [-ran, ran]
    """

    def __init__(self, var):
        super().__init__('gaussian')
        self.var = var # subgaussian parameter

    def simu(self, nsamples=1):
        """
        return nsamples samples of the distribution
        """
        return np.random.normal(scale=np.sqrt(self.var), size=nsamples)

# Define useful functions

def reverse_bisect_right(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted in descending order.

    The return value i is such that all e in a[:i] have e >= x, and all e in
    a[i:] have e < x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    Essentially, the function returns number of elements in a which are >= than x.
    >>> a = [8, 6, 5, 4, 2]
    >>> reverse_bisect_right(a, 5)
    3
    >>> a[:reverse_bisect_right(a, 5)]
    [8, 6, 5]
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x > a[mid]: hi = mid
        else: lo = mid+1
    return lo

def phi_function(rewards, samples, rate=1, weights=None):
    """
    Approximate phi(x) using Monte Carlo methods. For a vector X of n points, rewards is r(X) and samples are X.
    rewards: vector of rewards (Y_i)
    samples: vector of associated rides (X_i)
    rate: poisson rate (default=1)
    weights: used to consider weighted average (default=None gives a mean). See numpy.average for more.
    """
    return (lambda x: rate*np.average(np.maximum(rewards-x*samples, 0),
                                      weights=weights)-x)


def phi_function_vector(rewards, samples, rate=1, weights=None):
    """
    Approximate phi(x) where x is a vector of points.
    rewards: vector of rewards (Y_i)
    samples: vector of associated rides (X_i)
    rate: poisson rate (default=1)
    weights: used to consider weighted average (default=None gives a mean). See numpy.average for more.
    """
    return (lambda x: rate*np.average(np.maximum(
        rewards-x.reshape(-1, 1)*samples.reshape(1, -1), 0), axis=1,
        weights=weights)-x)


def profOpt(rewards, samples, rate=1, up=1, precision=1e-3, weights=None):
    """
    Return the optimal profitability threshold.
    reward: reward class considered
    distrib: distribution of rides (X)
    rate: rate of the poisson distribution for appearance of new rides (default 1)
    nsamples: number of samples used to estimate the profitability (Monte-Carlo method)
    up: parameter of initalization when looking for the root (default=1)
    precision: precision level of the returned root
    weights: useful for speed up when some (rewards, samples) are repeated (see phi_function)
    """
    if up <= 0:
        raise ValueError('up must be positive')
    phi = phi_function(rewards, samples, rate=rate, weights=weights)
    # the goal is now to find the root of phi, which is a decreasing function
    # we start by looking a value of up such that phi(up) < 0
    while phi(up) >= 0:
        up *= 2
    # now process to a binary search
    low = 0
    while up-low > precision:
        mid = (up+low)/2
        if phi(mid) > 0:
            low = mid
        else:
            up = mid
    return mid


def simu(rew, distrib, algo, noise=uniformNoise(1), rate=1,
         horizon=100, track_history=False):
    """
    Simulate a problem instance with given ride distribution, reward and strategy
    rew: reward function r
    distrib: distribution of rides
    algo: used strategy
    noise: distribution of noise added to observed rewards
    rate: poisson rate for ride propositions
    horizon: length T of the problem
    """
    total_reward = 0
    if track_history:
        proposed_rides = []
        accepts = []
        proposal_times = []
        cstars = []
    time = 0
    while time < horizon:

        S = np.random.exponential(rate)
        X = distrib.simu()
        time += S

        if time < horizon:  # final time not reached yet
            accepted = algo.accept(X)  # decision of algo

            if track_history:  # update history of observations
                proposed_rides.append(X)
                accepts.append(accepted)
                proposal_times.append(np.copy(time))
                cstars.append(algo.c)

            if accepted:
                total_reward += rew.eval(X)[0]
                time += X
                r = rew.eval(X) + noise.simu()
                algo.update(X, accepted, S, r)
            else:
                algo.update(X, accepted, S, 0)  # reward is not observed

    if track_history:
        return total_reward, proposed_rides, accepts, proposal_times, cstars
    else:
        return total_reward


if __name__ == '__main__':
    pass
