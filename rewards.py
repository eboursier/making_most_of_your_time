import numpy as np

# for multiple thresholds strategies, here are a few examples I have in mind:
#   - central band = suboptimal: rew(x) = tanh(x) (1-Lipschitz)
#   - central band = optimal: rew(x) = x(X-1)^2


class reward():
    """
    Superclass for different types of reward function
    """

    def __init__(self, function):
        self.eval = function

    def opt_areas(self, cstar, precision=1e-3, maxride=1):
        """
        return two vectors x,l
        require the function thresholds to be define for the considered reward
        l_i is True if profitability is > cstar on [x_i, x_i+1]
            and False if profitability is < cstar on [x_i, x_i+1]
        """
        x = self.thresholds(y=cstar, precision=precision, maxride=maxride)
        x = np.concatenate((np.array(0).reshape(
            1), x, np.array(maxride).reshape(1)))
        l = np.zeros(len(x)-1)
        for i in range(len(x)-1):
            l[i] = (self.eval((x[i]+x[i+1])/2) >= cstar*(x[i]+x[i+1])/2)
        return x, l


class affineReward(reward):
    """
    Affine Reward class. Initialize with 'dirCoef' and 'originOrd'
    """

    def __init__(self, dirCoef=1, originOrd=-0.5):
        super().__init__(lambda x: dirCoef*x+originOrd)
        self.name = "affine"
        self.dirCoef = dirCoef
        self.originOrd = originOrd

    def holder(self, **kwargs):
        """
        return (beta, L) such that the reward is (beta, L)
        holder on the interval [0, maxride]
        """
        return (1, self.dirCoef)

    def thresholds(self, y, **kwargs):
        """
        return all x such that r(x)/x = y
        very useful to visualize optimality regions when y = c*
        """
        return [self.originOrd/(y-self.dirCoef)]  # closed form


class concaveReward(reward):
    """
    Concave reward class of the form r(x) = log(1+x)
    """

    def __init__(self):
        super().__init__(lambda x: np.log(1+x))
        self.name = "concave"

    def holder(self, **kwargs):
        """
        return (beta, L) such that the reward is
        (beta, L) holder on the interval [0, maxride]
        """
        return (1, 1)

    def thresholds(self, y, precision=1e-5, maxride=1):
        """
        return all x such that r(x)/x = y
        very useful to visualize optimality regions when y = c*
        use linesearch as r(x)/x is decreasing
        """
        def prof(t): return self.eval(t)/t - y
        # now process to a linesearch
        up = maxride
        low = 0
        while up-low > precision:
            mid = (up+low)/2
            if prof(mid) > 0:
                low = mid
            else:
                up = mid

        return [(up+low)/2]


class convexReward(reward):
    """
    Convex reward of the form r(x)=exp(x)-1
    """

    def __init__(self):
        super().__init__(lambda x: np.exp(x)-1)
        self.name = "convex"

    def holder(self, maxride=1):
        """
        return (beta, L) such that the reward is
        (beta, L) holder on the interval [0, maxride]
        """
        return (1, np.exp(maxride))

    def thresholds(self, y, precision=1e-5, maxride=1):
        """
        return all x such that r(x)/x = y
        very useful to visualize optimality regions when y = c*
        use linesearch as r(x)/x is increasing
        """
        def prof(t): return self.eval(t)/t - y
        up = maxride
        low = 0
        while up-low > precision:
            mid = (up+low)/2
            if prof(mid) < 0:
                low = mid
            else:
                up = mid

        return [(up+low)/2]


class concaveReward2(reward):
    """
    Reward class such that optimal rides are in [x_0, x_1]
    """
    def __init__(self, a = 1, b = 0.2, c = 0.3):
        f=lambda x: a*x-b-c*x*x
        self.a, self.b, self.c = a, b, c
        super().__init__(f)
        self.name = "concave2"

    def holder(self, maxride=1):
        """
        return (beta, L) such that the reward is
        (beta, L) holder on the interval [0, maxride]
        """
        return(1, max(self.a, 2*self.b*maxride-1))

    def thresholds(self, y, precision=1e-3, maxride=1):
        """
        return all x such that r(x)/x = y
        very useful to visualize optimality regions when y = c*
        """
        t= np.arange(0, maxride, precision)
        z= np.sign(self.eval(t)/t-y)  # detect the changes of sign
        return t[np.where(np.diff(z))]


class customReward(reward):
    """
    Custom reward class with different reward functions.
    """

    def __init__(self, number=1):
        self.number= number
        self.name= "custom {}".format(number)
        if number == 1:
            f= lambda x: np.exp(-x**4 + 4*x**3 - 8*x + 1)+10*x
        elif number == 2:
            f=lambda x: x**4-3*x**3-2*x**2+5
        elif number == 3:
            f=lambda x: -10*(x-1)**2+1
        else:
            raise ValueError("Incorrect choice of custom number.")
        super().__init__(f)

    def holder(self, maxride=1):
        """
        return (beta, L) such that the reward is
        (beta, L) holder on the interval [0, maxride]
        """
        if self.number == 1:
            return(1, 482)  # |f'(x)| is bounded by 482 on R_+
        elif self.number == 2:
            return(1, np.max(4*maxride**3, 9*maxride**2+4*maxride))
        elif self.number == 3:
            return(1, 20*maxride)
        elif self.number == 4:
            return(1, max(1, 0.6*maxride-1))

    def thresholds(self, y, precision=1e-3, maxride=1):
        """
        return all x such that r(x)/x = y
        very useful to visualize optimality regions when y = c*
        """
        t=np.arange(0, maxride, precision)
        z=np.sign(self.eval(t)/t-y)  # detect the changes of sign
        return t[np.where(np.diff(z))]


if __name__ == '__main__':
    pass
