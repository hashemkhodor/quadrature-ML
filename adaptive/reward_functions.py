import numpy as np


class Reward:
    def __call__(self, error, step_size):
        """
        Parameters
        ----------
        error : float
        step_size : float

        Returns
        -------
        float
            reward
        """
        return 0

    @staticmethod
    def linear_map(point1, point2):
        """
        Calculate parameters of linear map y = a * x + b through two points (x1, y1), (x2, y2).

        Parameters
        ----------
        point1 : tuple[float]
        point2 : tuple[float]

        Returns
        -------
        function
        """
        a = (point2[1] - point1[1]) / (point2[0] - point1[0])
        b = point1[1] - a * point1[0]

        def f(x):
            return a * x + b
        return f

    @staticmethod
    def log_map(point1, point2):
        """
        Calculate logarithmic map y = a * log(b * x) through two points (x1, y1), (x2, y2).

        Parameters
        ----------
        point1 : tuple[float]
        point2 : tuple[float]

        Returns
        -------
        function
        """
        a = (point1[1] - point2[1]) / np.log(point1[0] / point2[0])
        b = np.exp(point2[1] / a) / point2[0]

        def f(x):
            return a * np.log(b * x)
        return f

    @staticmethod
    def exp_map(tol, m, limit):
        """
        Calculate function f(x) = a * exp(-b * x) - L, such that
        1) f(tol) = 0
        2) f(m * tol) = -1
        3) f(inf) = -limit

        Parameters
        ----------
        tol : float
        m : float
        limit : float

        Returns
        -------
        function
        """
        L = limit
        a = (L ** m / (L - 1)) ** (1 / (m - 1))
        b = -np.log(L / a) / tol

        def f(x):
            return a * np.exp(-b * x) - L
        return f


class RewardLog10(Reward):
    def __init__(self, error_tol, step_size_range, reward_range):
        """
        Scale positive rewards logarithmically with step_size.
        Scale negative rewards logarithmically with error.

        Parameters
        ----------
        error_tol : float
        step_size_range : tuple[float]
            lower and upper bound of expected step sizes
        reward_range : tuple[float]
            lower and upper bound of desired reward
        """
        self.error_tol = error_tol
        self.pos_f = self.log_map(*list(zip(step_size_range, reward_range)))

    def __call__(self, error, step_size):
        """
        Parameters
        ----------
        error : float
        step_size : float

        Returns
        -------
        float
            reward
        """
        if error < self.error_tol:
            return self.pos_f(step_size)

        # if error = 10^m * tol, then reward = -m
        return np.log10(self.error_tol / error)


class RewardExp(Reward):
    def __init__(self, error_tol, step_size_range, reward_range):
        """
        Scale positive rewards linearly with step_size.
        Scale negative rewards via negative exponential function w.r.t. error.

        Parameters
        ----------
        error_tol : float
        step_size_range : tuple[float]
            lower and upper bound of expected step sizes
        reward_range : tuple[float]
            lower and upper bound of desired reward
        """
        self.error_tol = error_tol
        self.pos_f = self.linear_map(*list(zip(step_size_range, reward_range)))
        self.neg_f = self.exp_map(error_tol, 2, reward_range[1])

    def __call__(self, error, step_size):
        """
        Parameters
        ----------
        error : float
        step_size : float

        Returns
        -------
        float
            reward
        """
        if error < self.error_tol:
            return self.pos_f(step_size)

        return self.neg_f(error)


# 1. Piece-wise linear (simple reference baseline)
class RewardLinear(Reward):
    def __init__(self, error_tol, step_size_range, reward_range):
        self.error_tol = error_tol
        self.pos_f = self.linear_map(*list(zip(step_size_range, reward_range)))
        # negative range mirrors positive peak at −reward_range[1]
        self.neg_f = self.linear_map((error_tol, 0.0),
                                     (10*error_tol, -reward_range[1]))

    def __call__(self, error, step_size):
        return self.pos_f(step_size) if error < self.error_tol else self.neg_f(error)


# 2. Smooth logistic shaping
class RewardSigmoid(Reward):
    def __init__(self, error_tol, step_size_mid, s_scale=10.0, e_scale=5.0):
        self.error_tol = error_tol
        self.ss_mid = step_size_mid
        self.s_scale = s_scale
        self.e_scale = e_scale

    def _sigmoid(self, x):     # standard logistic in (−1, 1)
        return 2.0 / (1.0 + np.exp(-x)) - 1.0

    def __call__(self, error, step_size):
        if error < self.error_tol:
            return self._sigmoid(self.s_scale * (step_size / self.ss_mid - 1.0))
        return -self._sigmoid(self.e_scale * np.log10(error / self.error_tol))


# 3. Hyper-inverse penalty
class RewardInverse(Reward):
    def __init__(self, error_tol, step_size_norm=1.0, p=2.0):
        self.tol = error_tol
        self.s_norm = step_size_norm
        self.p = p

    def __call__(self, error, step_size):
        if error < self.tol:
            return step_size / self.s_norm
        return -((error / self.tol) ** self.p - 1.0)


# 4. Quadratic “band” around optimum
class RewardQuadratic(Reward):
    def __init__(self, error_tol, step_opt, w_step=1.0, w_err=1.0):
        self.tol = error_tol
        self.s_opt = step_opt
        self.ws = w_step
        self.we = w_err

    def __call__(self, error, step_size):
        # reward is highest when step_size == step_opt and error == tol
        r_step = -self.ws * ((step_size - self.s_opt) / self.s_opt) ** 2
        r_err  = -self.we * (np.log10(max(error, 1e-16) / self.tol)) ** 2
        return r_step + r_err


# 5. Asymmetric exponential penalty (harsh on overshoot)
class RewardAsymmetricExp(Reward):
    def __init__(self, error_tol, step_size_range, k=4.0):
        self.tol = error_tol
        self.pos_f = self.linear_map(*list(zip(step_size_range, (0.0, 1.0))))
        self.k = k

    def __call__(self, error, step_size):
        if error < self.tol:
            return self.pos_f(step_size)
        # overshoot penalty grows exponentially in error ratio
        return -np.exp(self.k * np.log(error / self.tol))