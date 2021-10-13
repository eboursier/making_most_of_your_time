import numpy as np
from utils import *
from RBT import *


class strategy():
    """
    Strategy super class
    """

    def __init__(self, name):
        self.name = name


class BaselineStrategy(strategy):
    """
    Baseline algorithm
    reward: reward class
    distribution: distribution of ride propositions X
    rate: poisson rate
    nsamples: number of samples to estimate the acceptance
         threshold (default=100000)
    """

    def __init__(self, reward, distribution, rate=1,
                 nsamples=100000, precision=1e-5):
        super().__init__('BaselineStrategy')
        samples = distribution.simu(nsamples)
        rewards = reward.eval(samples)
        self.c = profOpt(rewards, samples, rate=rate, precision=precision)
        self.reward = reward
        self.nsamples = nsamples
        self.precision = precision
        self.rate = rate
        self.distribution = distribution
        self.t = 0

    def accept(self, X):
        """
        return True if accept the ride X, False if reject
        """
        return (self.reward.eval(X) >= self.c*X)

    def update(self, X, accepted, S, r):
        """
        Only update the time for this strategy (no learning)
        """
        self.t += S + accepted*X  # update time state

    def reset(self):
        """
        reset algorithm to start
        """
        self.t = 0
        samples = self.distribution.simu(self.nsamples)
        rewards = self.reward.eval(samples)
        self.c = profOpt(rewards, samples, rate=self.rate,
                         precision=self.precision)


class NaiveStrategy(strategy):
    """
    Naive algorithm which accepts all rides.
    """

    def __init__(self):
        super().__init__('NaiveStrategy')
        self.t = 0

    def accept(self, X):
        """
        accept all rides
        """
        return True

    def update(self, X, accepted, S, r):
        """
        Only update the time for this strategy (no learning)
        """
        self.t += S + accepted*X  # update time state

    def reset(self):
        """
        reset algorithm
        """
        self.t = 0


class UnknownDistribution(strategy):
    """
    Algorithm 1 (require knowledge of reward function)
    reward: reward class
    rate: poisson rate
    compute: True if we use the trick described in Remark 3.3 to improve complexity
    delta: confidence level (useful if compute=True)
    maxrew: max of reward function
    minrew: min of reward function
    """

    def __init__(self, reward, rate=1, compute=True, delta=None,
                 maxrew=1, minrew=0):
        super().__init__('UnknownDistribution')
        self.reward = reward
        self.t = 0  # current timestep
        self.c = 0  # acceptance threshold
        self.rides = np.array([])  # history of received rides
        self.rewards = np.array([])  # history of rewards
        self.rate = rate  # poisson rate
        self.n = 0  # number of proposed rides
        self.compute = compute
        self.maxrew = maxrew
        self.minrew = minrew
        if compute:
            if delta is None:
                raise ValueError('Set a value for delta with compute=True.')
            self.delta = delta
            self.sum_rew_above = 0  # mean of r(X) for rides above c^+
            self.sum_ride_above = 0  # mean of X for rides above c^+
            self.n_above = 0  # number of r(X) for rides above c^+
            self.n_below = 0  # number of r(X) for rides below c^-

    def accept(self, X):
        """
        return True if accept the ride X, False if reject
        """
        return (self.reward.eval(X) >= self.c*X)

    def update(self, X, accepted, S, Y):
        """
        update algo after receiving proposal X after a waiting time S
        and taking decision accepted (True or False)
        """
        self.t += S + accepted*X  # update time state
        self.n += 1
        # update history
        self.rides = np.append(self.rides, X)
        self.rewards = np.append(self.rewards, self.reward.eval(X))
        if self.compute:
            try:
                cont_rew = np.concatenate(
                    (self.rewards, [0, self.sum_rew_above/self.n_above]))
                cont_rides = np.concatenate(
                    (self.rides, [0, self.sum_ride_above/self.n_above]))
            except ZeroDivisionError:
                cont_rew = np.concatenate((self.rewards, [0, 0]))
                cont_rides = np.concatenate((self.rides, [0, 0]))

            weights = np.concatenate(
                (np.ones_like(self.rides), [self.n_below, self.n_above]))
            self.c = profOpt(cont_rew, cont_rides, weights=weights,
                             rate=self.rate, precision=1e-4, up=self.c+1)

            # uncertainty on c^*
            eta = self.rate*(self.maxrew-self.minrew) * \
                np.sqrt(-np.log(self.delta)/(2*self.n))
            # remove all rides below c^- from history
            below = (self.rewards < (self.c - eta)*self.rides)
            if below.any():
                self.rides = self.rides[np.invert(below)]
                self.rewards = self.rewards[np.invert(below)]
                self.n_below += np.sum(below)

            # remove rides above c^+ from history
            above = (self.rewards > (self.c + eta)*self.rides)
            if above.any():
                self.sum_rew_above += np.sum(self.rewards[above])
                self.sum_ride_above += np.sum(self.rides[above])

                self.rides = self.rides[np.invert(above)]
                self.rewards = self.rewards[np.invert(above)]
                self.n_above += np.sum(above)
        else:
            # update acceptance threshold
            self.c = profOpt(self.rewards, self.rides,
                             rate=self.rate, precision=1e-4, up=self.c+1)

    def reset(self):
        """
        reset algorithm to restart simu
        """
        self.t = 0
        self.c = 0
        self.rides = np.array([])
        self.rewards = np.array([])
        self.n = 0
        if self.compute:
            self.sum_rew_above = 0
            self.sum_ride_above = 0
            self.n_above = 0
            self.n_below = 0


class UnknownDistribution_RBT(strategy):
    """
    BST implementation of Algorithm 1 (require knowledge of reward function)
    reward: reward class
    rate: poisson rate
    delta: confidence level (useful to use the trick described in Remark 3.3 )
    maxrew: max of reward function
    minrew: min of reward function
    """

    def __init__(self, reward, rate=1, delta=None, maxrew=1, minrew=0):
        super().__init__('UnknownDistribution')
        self.reward = reward
        self.t = 0  # current timestep
        self.c = 0  # acceptance threshold
        self.rate = rate  # poisson rate
        self.n = 0  # number of proposed rides
        self.sum_rew_above = 0  # mean of r(X) for rides above c^+
        self.sum_ride_above = 0  # mean of X for rides above c^+
        self.maxrew = maxrew
        self.minrew = minrew
        self.tree = RedBlackTree()  # history of observations
        if delta is None:
            raise ValueError('Set a value for delta to use computation trick of Remark 3.3.')
        self.delta = delta

    def accept(self, X):
        """
        return True if accept the ride X, False if reject
        """
        return (self.reward.eval(X) >= self.c*X)

    def update(self, X, accepted, S, Y):
        """
        update algo after receiving proposal X after a waiting time S
        and taking decision accepted (True or False)
        """
        self.t += S + accepted*X  # update time state
        self.n += 1

        # uncertainty on c^*
        eta = self.rate*(self.maxrew-self.minrew) * \
            np.sqrt(-np.log(self.delta)/(2*self.n))

        # update history
        r = self.reward.eval(X)
        if (r/X <= self.c + eta) and (r/X >= self.c-eta):
            self.tree.insert(key=r/X, info=np.array([r, X]))
        elif r > self.c + eta:
            self.sum_rew_above += r[0]
            self.sum_ride_above += X[0]

        self.c = np.clip(self.profOpt(), self.c - eta, self.c + eta)

        # remove all rides below c^- from history
        self.tree.deletebelow(self.c-eta)
        # remove rides above c^+ from hitory
        infosum = self.tree.deleteabove(self.c+eta)
        self.sum_rew_above += infosum[0]
        self.sum_ride_above += infosum[1]

    def profOpt(self):
        """
        compute 0 of phi_n
        """
        x = self.tree.root
        # sum of r(X_j) for j with profitability > than current node
        sumrew = np.copy(x.right_info[0] + self.sum_rew_above)
        sumrides = np.copy(x.right_info[1] + self.sum_ride_above)
        s1 = None
        s1 = None
        while (x != TNULL):
            c = x.data[0]
            # evalute phi_n(c)
            phi = (self.rate/self.n)*(sumrew-c*sumrides)-c

        # search the largest c_i=r(X_i)/X_i such that
        # phi_n(c_i) > 0
            if phi > 0:
                s1, s2 = np.copy(sumrew), np.copy(sumrides)
                x = x.right
                if x != TNULL:
                    sumrew -= x.left_info[0] + x.info[0]
                    sumrides -= x.left_info[1] + x.info[1]
            else:
                x = x.left
                if x != TNULL:
                    sumrew += x.right_info[0]
                    sumrides += x.right_info[1]
                    if x.data[0] < x.parent.data[0]:
                        # avoid case of equality
                        sumrew += x.parent.info[0]
                        sumrides += x.parent.info[1]
        if s1:
            return (self.rate*s1)/(self.n+self.rate*s2)
        else:
            return (self.rate*sumrew)/(self.n+self.rate*sumrides)

    def reset(self):
        """
        reset algorithm to restart simu
        """
        self.t = 0
        self.c = 0
        self.n = 0
        self.tree = RedBlackTree()
        self.sum_rew_above = 0  # mean of r(X) for rides above c^+
        self.sum_ride_above = 0


class Bandit(strategy):
    """
    Algorithm 2
    horizon: horizon T of the problem
    rate: Poisson rate (default=1)
    delta: confidence level (default 1/horizon^2)
    maxride: maximum possible value of ride (default=1)
    nbuckets: number of buckets when discretizing the interval (0, maxride)
    beta: first Holder constant of reward function
    L: second Holder constant of reward function
    nbuckets: number M of buckets (default as in Theorem 4.5)
    kappa: kappa used in xi (default=150)
    var: subgaussian constant of the noise in observation of reward (default 1/2)
    """

    def __init__(self, horizon, L=1, beta=1, rate=1, delta=None, minrew=0,
                 maxrew=1, maxride=1, nbuckets=None, kappa=150, var=1/2):
        super().__init__('Bandit')
        self.horizon = horizon
        if delta is None:
            self.delta = 1/horizon**2
        else:
            self.delta = delta
        if nbuckets is None:
            self.nbuckets = (int)(np.ceil(
                maxride*L**(2/(2*beta+1))*(rate*horizon+1)**(1/(2*beta+1))))
        else:
            self.nbuckets = nbuckets
        self.t = 0  # current timestep
        self.c = 0  # acceptance threshold
        self.rate = rate  # poisson rate
        self.maxride = maxride
        self.mean_rewards = np.zeros(self.nbuckets)  # mean reward per bucket
        self.tilde = np.ones(self.nbuckets)  # used for tilde{r}
        self.N = np.zeros(self.nbuckets)  # number of accepted rides per bucket
        self.eta = maxrew*np.ones(self.nbuckets)  # uncertainty on reward
        # discretize space for profitability function (max X in bucket here)
        self.X = np.linspace(0, maxride-maxride/self.nbuckets, self.nbuckets)
        self.L = L
        self.beta = beta
        self.bias = L*(maxride/self.nbuckets)**(beta)  # constant added to eta
        self.var = var + 0.
        self.sigma1 = np.sqrt(var + self.bias**2/4)
        self.sigma2 = np.sqrt(var + (maxrew-minrew)**2/4)
        self.maxrew = maxrew
        self.minrew = minrew
        self.kappa = kappa
        self.n = 0  # total number of proposed rides
        self.xi = np.inf

    def accept(self, X):
        """
        return True if accept the ride X, False if reject
        """
        b = (int)(np.floor(X*self.nbuckets/self.maxride))  # current bucket
        return self.tilde[b]

    def update(self, X, accepted, S, Y):
        """
        update algo after receiving proposal X after a waiting time S
        and taking decision accepted (True or False)
        """
        self.t += S + accepted*X  # update time state
        self.n += 1
        # corresponding bucket
        b = (int)(np.floor(X*self.nbuckets/self.maxride))
        self.N[b] += 1

        if accepted:
            self.mean_rewards[b] = (
                (self.N[b]-1)*self.mean_rewards[b]+Y)/(self.N[b])

            self.eta[b] = self.sigma1 * \
                np.sqrt(np.log(self.nbuckets/self.delta) /
                        (2*self.N[b])) + self.bias

        self.xi = 2*self.rate*self.sigma2*np.sqrt(np.log(1/self.delta)/self.n)
        self.xi += self.kappa*self.rate*np.max(((self.maxrew-self.minrew)/2,
                                                np.sqrt(self.var))) * \
            np.sqrt((np.log(self.n)+1)/(self.maxride*self.n/self.nbuckets))
        # the constant terms in xi are ignored
        # thanks to Remark 4.5

        self.tilde *= (self.mean_rewards + self.eta >=
                       np.maximum(((self.c-self.xi)*self.X), 0))

        # compute root of phi_n
        self.c = profOpt(self.mean_rewards*self.tilde, self.X,
                         rate=self.rate, precision=1e-4, up=self.c+1,
                         weights=self.N)

    def reset(self):
        """
        reset algorithm to start
        """
        self.t = 0
        self.c = 0
        self.mean_rewards = np.zeros(self.nbuckets)
        self.N = np.zeros(self.nbuckets)
        self.eta = self.maxrew*np.ones(self.nbuckets)
        self.n = 0
        self.tilde = np.ones(self.nbuckets)
        self.xi = np.inf


class Bandit_RBT(strategy):
    """
    Algorithm 2
    horizon: horizon T of the problem
    rate: Poisson rate (default=1)
    delta: confidence level (default 1/horizon^2)
    maxride: maximum possible value of ride (default=1)
    nbuckets: number of buckets when discretizing the interval (0, maxride)
    beta: first Holder constant of reward function
    L: second Holder constant of reward function
    nbuckets: number M of buckets (default as in Theorem 4.5)
    kappa: kappa used in xi (default=150)
    var: subgaussian constant of the noise in observation of reward (default 1/2)

    Important note: tilde[b] does not have to be updated at each time step for all b,
    but only the one that was pulled.
    """

    def __init__(self, horizon, L=1, beta=1, rate=1, delta=None, minrew=0,
                 maxrew=1, maxride=1, nbuckets=None, kappa=150, var=1/2):
        super().__init__('Bandit')
        self.horizon = horizon
        if delta is None:
            self.delta = 1/horizon**2
        else:
            self.delta = delta
        if nbuckets is None:
            self.nbuckets = (int)(np.ceil(
                maxride*L**(2/(2*beta+1))*(rate*horizon+1)**(1/(2*beta+1))))
        else:
            self.nbuckets = nbuckets
        self.t = 0  # current timestep
        self.c = 0  # acceptance threshold
        self.rate = rate  # poisson rate
        self.maxride = maxride
        self.sum_rewards = np.zeros(self.nbuckets)  # sum of rewards per bucket
        self.tilde = np.ones(self.nbuckets)  # used for tilde{r}
        self.N = np.zeros(self.nbuckets)  # number of accepted rides per bucket
        self.tree = RedBlackTree()
        self.eta = maxrew*np.ones(self.nbuckets)  # uncertainty on reward
        # discretize space for profitability function (max X in bucket here)
        self.X = np.linspace(0, maxride-maxride/self.nbuckets, self.nbuckets)
        self.L = L
        self.beta = beta
        self.bias = L*(maxride/self.nbuckets)**(beta)  # constant added to eta
        self.var = var + 0.
        self.sigma1 = np.sqrt(var + self.bias**2/4)
        self.sigma2 = np.sqrt(var + (maxrew-minrew)**2/4)
        self.maxrew = maxrew
        self.minrew = minrew
        self.kappa = kappa
        self.n = 0  # total number of proposed rides
        self.xi = np.inf

    def accept(self, X):
        """
        return True if accept the ride X, False if reject
        """
        b = (int)(np.floor(X*self.nbuckets/self.maxride))  # current bucket
        if b == self.nbuckets:
            # limit case of non continuous distributions
            b = self.nbuckets - 1
        if self.N[b] == 0:
            return True
        elif self.tilde[b] == 0:
            return False
        else:
            return (self.sum_rewards[b]/self.N[b] + self.eta[b] >= np.maximum(((self.c-self.xi)*self.X[b]), 0))

    def update(self, X, accepted, S, Y):
        """
        update algo after receiving proposal X after a waiting time S
        and taking decision accepted (True or False)
        """
        self.t += S + accepted*X  # update time state
        self.n += 1
        # corresponding bucket
        b = (int)(np.floor(X*self.nbuckets/self.maxride))
        if b == self.nbuckets:
            # limit case of non continuous distributions
            b = self.nbuckets - 1
        if accepted:
            self.N[b] += 1
            if self.N[b] > 1:
                self.tree.delete_node(key=(self.sum_rewards[b]/((self.N[b]-1)*self.X[b]), b),
                                      info=np.array([self.sum_rewards[b], (self.N[b]-1)*self.X[b]]))

            self.sum_rewards[b] += Y

            self.tree.insert(key=(self.sum_rewards[b]/(self.N[b]*self.X[b]), b),
                             info=np.array([self.sum_rewards[b], self.N[b]*self.X[b]]))

            self.eta[b] = self.sigma1 * \
                np.sqrt(np.log(self.nbuckets/self.delta) /
                        (2*self.N[b])) + self.bias

        elif self.tilde[b] == 1:
            # rejected b for the first time
            # replace the mean reward by 0 in the tree
            self.tilde[b] = 0
            self.tree.delete_node(key=(self.sum_rewards[b]/((self.N[b])*self.X[b]), b),
                                  info=np.array([self.sum_rewards[b], (self.N[b])*self.X[b]]))
            self.tree.insert(key=(0, b), info=np.array(
                [0, self.N[b]*self.X[b]]))

        self.xi = 2*self.rate*self.sigma2*np.sqrt(np.log(1/self.delta)/self.n)
        self.xi += self.kappa*self.rate*np.max(((self.maxrew-self.minrew)/2,
                                                np.sqrt(self.var))) * \
            np.sqrt((np.log(self.n)+1)/(self.maxride*self.n/self.nbuckets))
        # the constant terms in xi are ignored
        # thanks to Remark 4.5

        # compute root of phi_n
        self.c = self.profOpt()

    def profOpt(self):
        """
        compute 0 of phi_n
        """
        x = self.tree.root
        # sum of r(X_j) for j with profitability > than current node
        sumrew = np.copy(x.right_info[0])
        sumrides = np.copy(x.right_info[1])
        s1 = None
        while (x != TNULL):
            c = x.data[0]
            # evalute phi_n(c)
            phi = (self.rate/self.n)*(sumrew-c*sumrides)-c

        # search the largest c_i=r(X_i)/X_i such that
        # phi_n(c_i) > 0
            if phi > 0:
                s1, s2 = np.copy(sumrew), np.copy(sumrides)
                x = x.right
                if x != TNULL:
                    sumrew -= x.left_info[0] + x.info[0]
                    sumrides -= x.left_info[1] + x.info[1]
            else:
                x = x.left
                if x != TNULL:
                    sumrew += x.right_info[0]
                    sumrides += x.right_info[1]
                    if x.data[0] < x.parent.data[0]:
                        # avoid case of equality
                        sumrew += x.parent.info[0]
                        sumrides += x.parent.info[1]
        if s1:
            return (self.rate*s1)/(self.n+self.rate*s2)
        else:
            return (self.rate*sumrew)/(self.n+self.rate*sumrides)

    def reset(self):
        """
        reset algorithm to start
        """
        self.t = 0
        self.c = 0
        self.sum_rewards = np.zeros(self.nbuckets)
        self.N = np.zeros(self.nbuckets)
        self.eta = self.maxrew*np.ones(self.nbuckets)
        self.n = 0
        self.tilde = np.ones(self.nbuckets)
        self.xi = np.inf
        self.tree = RedBlackTree()


class FiniteBandit(strategy):
    """
    Algorithm in the finite support (of X) setting
    horizon: horizon T of the problem
    X: support of the distribution
    rate: Poisson rate (default=1)
    delta: confidence level (default 1/horizon^2)
    minrew: minimal possible value of reward (default=0)
    maxrew: maximal possible value of reward (default=1)
    """

    def __init__(self, horizon, X, rate=1, delta=None, minrew=0,
                 var=1, maxrew=1):
        super().__init__('Finite Bandit')
        self.horizon = horizon
        if delta is None:
            self.delta = 1/horizon**2
        else:
            self.delta = delta
        self.X = X
        self.K = len(X)
        self.t = 0  # current timestep
        self.c = 0  # acceptance threshold
        self.rate = rate  # poisson rate
        self.minrew = minrew
        self.maxrew = maxrew
        self.mean_rewards = np.zeros_like(X)  # mean reward per ride time X
        self.N = np.zeros_like(X)  # number of proposed rides per ride X
        self.eta = maxrew*np.ones_like(X)  # uncertainty on reward
        self.tilde = np.ones_like(X)
        self.sigma = np.sqrt(var)
        self.n = 0  # total number of proposed rides
        self.xi = np.inf

    def accept(self, X):
        """
        return True if accept the ride X, False if reject
        """
        b = np.argwhere(self.X == X)[0, 0]
        return self.tilde[b]

    def update(self, X, accepted, S, Y):
        """
        update algo after receiving proposal X after a waiting time S
        and taking decision accepted (True or False)
        """
        self.t += S + accepted*X  # update time state
        self.n += 1
        b = np.argwhere(self.X == X)[0, 0]

        if accepted:
            self.N[b] += 1  # update number of rides X
            # update mean rewards
            self.mean_rewards[b] = (
                (self.N[b]-1)*self.mean_rewards[b]+Y) / \
                (self.N[b])

            self.eta[b] = self.sigma * \
                np.sqrt(np.log(self.K/self.delta)/(2*self.N[b]))

        self.xi = 2*self.rate * \
            np.sqrt(self.sigma**2+(self.maxrew-self.minrew)**2/4) * \
            np.sqrt(-np.log(self.delta)/self.n)
        self.xi += self.rate*self.sigma*np.sqrt(self.K/(2*self.n))
        self.xi += 8*self.rate*self.K*(self.maxrew-self.minrew)/self.n

        # now update the threshold c
        # first define tilde r

        self.tilde *= (self.mean_rewards + self.eta >=
                       np.maximum(((self.c-self.xi)*self.X), 0))

        # compute root of phi_n
        self.c = profOpt(self.mean_rewards*self.tilde, self.X,
                         rate=self.rate, precision=1e-4, up=self.c+1,
                         weights=self.N)

    def reset(self):
        """
        reset algorithm to start
        """
        self.t = 0
        self.c = 0
        self.mean_rewards = np.zeros(self.K)
        self.N = np.zeros(self.K)
        self.eta = self.maxrew*np.ones(self.K)
        self.n = 0
        self.tilde = np.ones(self.K)
        self.xi = np.inf


class FiniteBandit_RBT(strategy):
    """
    Algorithm in the finite support (of X) setting
    horizon: horizon T of the problem
    X: support of the distribution
    rate: Poisson rate (default=1)
    delta: confidence level (default 1/horizon^2)
    minrew: minimal possible value of reward (default=0)
    maxrew: maximal possible value of reward (default=1)
    """

    def __init__(self, horizon, X, rate=1, delta=None, minrew=0,
                 var=1, maxrew=1):
        super().__init__('Finite Bandit')
        self.horizon = horizon
        if delta is None:
            self.delta = 1/horizon**2
        else:
            self.delta = delta
        self.X = X
        self.K = len(X)
        self.t = 0  # current timestep
        self.c = 0  # acceptance threshold
        self.rate = rate  # poisson rate
        self.minrew = minrew
        self.maxrew = maxrew
        self.sum_rewards = np.zeros_like(X)  # mean reward per ride time X
        self.N = np.zeros_like(X)  # number of proposed rides per ride X
        self.eta = maxrew*np.ones_like(X)  # uncertainty on reward
        self.tilde = np.ones_like(X)
        self.sigma = np.sqrt(var)
        self.n = 0  # total number of proposed rides
        self.tree = RedBlackTree()
        self.xi = np.inf

    def accept(self, X):
        """
        return True if accept the ride X, False if reject
        """
        b = np.argwhere(self.X == X)[0, 0]
        if self.N[b] == 0:
            return True
        elif self.tilde[b] == 0:
            return False
        else:
            return (self.sum_rewards[b]/self.N[b] + self.eta[b] >= np.maximum(((self.c-self.xi)*self.X[b]), 0))

    def update(self, X, accepted, S, Y):
        """
        update algo after receiving proposal X after a waiting time S
        and taking decision accepted (True or False)
        """
        self.t += S + accepted*X  # update time state
        self.n += 1
        # corresponding bucket
        b = np.argwhere(self.X == X)[0, 0]
        if accepted:
            self.N[b] += 1
            if self.N[b] > 1:
                self.tree.delete_node(key=(self.sum_rewards[b]/((self.N[b]-1)*self.X[b]), b),
                                      info=np.array([self.sum_rewards[b], (self.N[b]-1)*self.X[b]]))

            self.sum_rewards[b] += Y

            self.tree.insert(key=(self.sum_rewards[b]/(self.N[b]*self.X[b]), b),
                             info=np.array([self.sum_rewards[b], self.N[b]*self.X[b]]))

            self.eta[b] = self.sigma * \
                np.sqrt(np.log(self.K/self.delta)/(2*self.N[b]))

        elif self.tilde[b] == 1:
            # rejected b for the first time
            # replace the mean reward by 0 in the tree
            self.tilde[b] = 0
            self.tree.delete_node(key=(self.sum_rewards[b]/((self.N[b])*self.X[b]), b),
                                  info=np.array([self.sum_rewards[b], (self.N[b])*self.X[b]]))
            self.tree.insert(key=(0, b), info=np.array(
                [0, self.N[b]*self.X[b]]))

        self.xi = 2*self.rate * \
            np.sqrt(self.sigma**2+(self.maxrew-self.minrew)**2/4) * \
            np.sqrt(-np.log(self.delta)/self.n)
        self.xi += self.rate*self.sigma*np.sqrt(self.K/(2*self.n))
        self.xi += 8*self.rate*self.K*(self.maxrew-self.minrew)/self.n

        # compute root of phi_n
        self.c = self.profOpt()

    def profOpt(self):
        """
        compute 0 of phi_n
        """
        x = self.tree.root
        # sum of r(X_j) for j with profitability > than current node
        sumrew = np.copy(x.right_info[0])
        sumrides = np.copy(x.right_info[1])
        s1 = None
        while (x != TNULL):
            c = x.data[0]
            # evalute phi_n(c)
            phi = (self.rate/self.n)*(sumrew-c*sumrides)-c

        # search the largest c_i=r(X_i)/X_i such that
        # phi_n(c_i) > 0
            if phi > 0:
                s1, s2 = np.copy(sumrew), np.copy(sumrides)
                x = x.right
                if x != TNULL:
                    sumrew -= x.left_info[0] + x.info[0]
                    sumrides -= x.left_info[1] + x.info[1]
            else:
                x = x.left
                if x != TNULL:
                    sumrew += x.right_info[0]
                    sumrides += x.right_info[1]
                    if x.data[0] < x.parent.data[0]:
                        # avoid case of equality
                        sumrew += x.parent.info[0]
                        sumrides += x.parent.info[1]
        if s1:
            return (self.rate*s1)/(self.n+self.rate*s2)
        else:
            return (self.rate*sumrew)/(self.n+self.rate*sumrides)

    def reset(self):
        """
        reset algorithm to start
        """
        self.t = 0
        self.c = 0
        self.sum_rewards = np.zeros(self.K)
        self.N = np.zeros(self.K)
        self.eta = self.maxrew*np.ones(self.K)
        self.n = 0
        self.tilde = np.ones(self.K)
        self.xi = np.inf
        self.tree = RedBlackTree()


class Monotone(strategy):
    """
    Algorithm 3: non-decreasing profitability function
    horizon: horizon T of the problem
    maxride: maximal value of algorithm
    rate: Poisson rate (default=1)
    delta: confidence level (default 1/horizon^2)
    minrew: minimal possible value of reward (default=0)
    maxrew: maximal possible value of reward (default=1)
    var: subgaussian proxy of noise in reward
    """

    def __init__(self, horizon, maxride=1, rate=1, delta=None, minrew=0,
                 var=1, maxrew=1):
        super().__init__('Monotone')
        self.horizon = horizon
        if delta is None:
            self.delta = 1/horizon
        else:
            self.delta = delta
        self.t = 0  # current timestep
        self.c = None
        self.s = 0  # acceptance threshold (s_n)
        self.smax = maxride
        self.maxride = maxride
        self.rides = [0]  # sorted history of rides
        # first term is sum of rides above smax
        self.rewards = [0]  # history of rewards
        # first term is sum of rewards of rides above smax
        self.rate = rate  # poisson rate
        self.minrew = min(minrew, 0)
        self.maxrew = max(maxrew, 0)
        self.sigma = np.sqrt(var+(self.maxrew-self.minrew)**2/4) + \
            (self.maxrew-self.minrew)*(rate*maxride+2)/np.sqrt(2)
        self.S = 2*rate*horizon+1
        self.n = 0  # total number of proposed rides
        self.xi = np.inf

    def accept(self, X):
        """
        return True if accept the ride X, False if reject
        """
        return X[0] >= self.s

    def update(self, X, accepted, S, Y):
        """
        update algo after receiving proposal X after a waiting time S
        and taking decision accepted (True or False)
        """
        self.t += S + accepted*X  # update time state
        self.n += 1
        # update history
        if X >= self.smax:
            self.rewards[0] += Y[0]
            self.rides[0] += X[0]
        elif X >= self.s:
            # we maintain an ordered list of rides in [s_n, s_n^+]
            # for optimized computation
            k = reverse_bisect_right(self.rides, X)  # where to add X and Y
            if k == 0:
                # first place of rides and rewards are for
                # rides longer than smax
                k += 1
            self.rides.insert(k, X[0])
            self.rewards.insert(k, Y[0])

        self.xi = self.sigma * \
            np.sqrt(np.log(2*(self.S+1)*self.horizon)/(self.n-1) +
                    self.rate*(self.maxrew-self.minrew)/self.n)

        # define the function c_n(s^k) for all k
        ridesum = np.cumsum(self.rides)
        cvals = self.rate*np.cumsum(self.rewards) / \
            (self.n+self.rate*ridesum)
        cmax = np.max(cvals)  # == c_n(s_n^*)
        # compute largest and smallest s in S_n
        k1 = next(i for i in range(len(cvals)) if (
            cvals[i]-cmax)*((1/self.rate+1/self.n*ridesum[i])) >= -2*self.xi)
        k2 = next(i for i in reversed(range(len(cvals))) if (
            cvals[i]-cmax)*((1/self.rate+1/self.n*ridesum[i])) >= -2*self.xi)
        if k1 > 1:
            self.smax = self.rides[k1-1]
            # only store rides shorter than smax
            self.rewards[0] += np.sum(self.rewards[1:k1-1])
            self.rides[0] += np.sum(self.rides[1:k1-1])
            del self.rewards[1:k1-1]
            del self.rides[1:k1-1]
        if k2+1 < len(self.rides):
            self.s = self.rides[k2+1]
            # only store rides longer than s
            del self.rewards[k2+2:]
            del self.rides[k2+2:]

    def reset(self):
        """
        reset algorithm to start
        """
        self.t = 0  # current timestep
        self.s = 0  # acceptance threshold (s_n)
        self.smax = self.maxride
        self.rate = self.rate  # poisson rate
        self.n = 0  # total number of proposed rides
        self.rides = [0]  # sorted history of rides
        self.rewards = [0]  # history of rewards
        self.xi = np.inf
