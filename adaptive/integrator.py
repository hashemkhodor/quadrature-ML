from typing import Sequence

from scipy.optimize import root
from sklearn.linear_model import LinearRegression
import numpy as np
from functions import Function, FunctionODE


class Integrator:
    def __call__(self, state):
        """
        Base Integrator call.

        Parameters
        ----------
        state : np.ndarray
            contains step_size and function evaluations

        Returns
        -------
        float
            integral
        """
        return 0

    @staticmethod
    def calc_state(a, h, f):
        """
        Calculate state consisting of step size and function evaluations.

        Parameters
        ----------
        a : float
        h : float
            step size
        f : Function

        Returns
        -------
        np.ndarray
        """
        return np.array([h, f(a)])


class Simpson(Integrator):
    def __init__(self):
        self.num_nodes = 3

    def __call__(self, state):
        """
        Parameters
        ----------
        state : np.ndarray
            contains step_size and function evaluations

        Returns
        -------
        float
            integral approximated using the simpson rule
        """

        step_size = state[0]
        fa, fm, fb = state[1:]
        return step_size / 6 * (fa + 4 * fm + fb)

    @staticmethod
    def calc_state(a, h, f):
        """
        Parameters
        ----------
        a : float
        h : float
            step size
        f : Function

        Returns
        -------
        np.ndarray
        """
        fa = f(a)
        fm = f((2 * a + h) / 2)
        fb = f(a + h)
        return np.array([h, fa, fm, fb])


class Boole(Integrator):
    def __init__(self):
        self.num_nodes = 5

    def __call__(self, state):
        """
        Parameters
        ----------
        state : np.ndarray
            contains step_size and function evaluations

        Returns
        -------
        float
            integral approximated using boole's rule
        """

        step_size = state[0]
        f1, f2, f3, f4, f5 = state[1:]
        return step_size / 90 * (7 * f1 + 32 * f2 + 12 * f3 + 32 * f4 + 7 * f5)

    @staticmethod
    def calc_state(a, h, f):
        """
        Parameters
        ----------
        a : float
        h : float
            step size
        f : Function

        Returns
        -------
        np.ndarray
        """
        b = a + h
        f1 = f(a)
        f2 = f(0.75 * a + 0.25 * b)
        f3 = f(0.5 * a + 0.5 * b)
        f4 = f(0.25 * a + 0.75 * b)
        f5 = f(b)
        return np.array([h, f1, f2, f3, f4, f5])


class Kronrod21(Integrator):
    def __init__(self):
        self.num_nodes = 21
        self.nodes = [1.488743389816312108848260011297200e-01,
                      2.943928627014601981311266031038656e-01,
                      4.333953941292471907992659431657842e-01,
                      5.627571346686046833390000992726941e-01,
                      6.794095682990244062343273651148736e-01,
                      7.808177265864168970637175783450424e-01,
                      8.650633666889845107320966884234930e-01,
                      9.301574913557082260012071800595083e-01,
                      9.739065285171717200779640120844521e-01,
                      9.956571630258080807355272806890028e-01]
        self.nodes = np.array(self.nodes)
        self.weights = [1.477391049013384913748415159720680e-01,
                        1.427759385770600807970942731387171e-01,
                        1.347092173114733259280540017717068e-01,
                        1.234919762620658510779581098310742e-01,
                        1.093871588022976418992105903258050e-01,
                        9.312545458369760553506546508336634e-02,
                        7.503967481091995276704314091619001e-02,
                        5.475589657435199603138130024458018e-02,
                        3.255816230796472747881897245938976e-02,
                        1.169463886737187427806439606219205e-02]
        self.weights = np.array(self.weights)
        self.nodes = np.concatenate((-np.flip(self.nodes), [0], self.nodes))
        self.weights = np.concatenate((np.flip(self.weights),
                                       [1.494455540029169056649364683898212e-01],
                                       self.weights))

    def __call__(self, state):
        """
        Parameters
        ----------
        state : np.ndarray
            contains step_size and function evaluations

        Returns
        -------
        float
            integral approximated using Kronrod21 rule
        """

        step_size = state[0]
        f_nodes = state[1:]
        return 0.5 * step_size * np.inner(self.weights, f_nodes)

    def calc_state(self, a, h, f):
        """
        Parameters
        ----------
        a : float
        h : float
            step size
        f : Function

        Returns
        -------
        np.ndarray
        """
        b = a + h
        f_nodes = [f(0.5 * h * self.nodes[i] + 0.5 * (a + b)) for i in range(21)]
        return np.array([h] + f_nodes)


class Gauss21(Integrator):
    def __init__(self):
        self.num_nodes = 21
        self.nodes = [0.1455618541608950909370309823386863,
                      0.2880213168024010966007925160646003,
                      0.4243421202074387835736688885437881,
                      0.551618835887219807059018796724313,
                      0.6671388041974123193059666699903392,
                      0.768439963475677908615877851306228,
                      0.8533633645833172836472506385875677,
                      0.9200993341504008287901871337149689,
                      0.9672268385663062943166222149076952,
                      0.9937521706203895002602420359379409]
        self.nodes = np.array(self.nodes)
        self.weights = [0.1445244039899700590638271665537525,
                        0.139887394791073154722133423867583,
                        0.1322689386333374617810525744967756,
                        0.121831416053728534195367177125734,
                        0.1087972991671483776634745780701056,
                        0.093444423456033861553289741113932,
                        0.07610011362837930201705165330018318,
                        0.057134425426857208283635826472448,
                        0.0369537897708524937999506682993297,
                        0.016017228257774333324224616858471]
        self.weights = np.array(self.weights)
        self.nodes = np.concatenate((-np.flip(self.nodes), [0], self.nodes))
        self.weights = np.concatenate((np.flip(self.weights),
                                       [0.1460811336496904271919851476833712],
                                       self.weights))

    def __call__(self, state):
        """
        Parameters
        ----------
        state : np.ndarray
            contains step_size and function evaluations

        Returns
        -------
        float
            integral approximated using Kronrod21 rule
        """

        step_size = state[0]
        f_nodes = state[1:]
        return 0.5 * step_size * np.inner(self.weights, f_nodes)

    def calc_state(self, a, h, f):
        """
        Parameters
        ----------
        a : float
        h : float
            step size
        f : Function

        Returns
        -------
        np.ndarray
        """
        b = a + h
        f_nodes = [f(0.5 * h * self.nodes[i] + 0.5 * (a + b)) for i in range(21)]
        return np.array([h] + f_nodes)


class IntegratorLinReg(Integrator):
    def __init__(self, step_sizes, models, base_integrator):
        """
        Integrator that uses a linear regression model for each step size.

        The nodes are the same as base_integrator, but the weights are optimized.

        Parameters
        ----------
        step_sizes : list[float]
        models : list[LinearRegression]
            input are the function evals [f_1,...,f_n] and output the integral
        base_integrator : Integrator
        """
        self.step_sizes = np.array(step_sizes)
        self.models = models
        self.base_integrator = base_integrator
        self.num_nodes = base_integrator.num_nodes

    def __call__(self, state):
        """
        Parameters
        ----------
        state : np.ndarray
            contains step_size and function evaluations

        Returns
        -------
        float
            approximated integral
        """
        step_size = state[0]
        f_evals = state[1:]

        if step_size in self.step_sizes:
            idx = np.argwhere(np.isclose(self.step_sizes, step_size))[0, 0]
            integral = self.models[idx].predict([f_evals])[0]
        else:
            integral = self.base_integrator(state)
        return integral

    def calc_state(self, a, h, f):
        """
        Parameters
        ----------
        a : float
        h : float
            step size
        f : Function

        Returns
        -------
        np.ndarray
        """
        return self.base_integrator.calc_state(a, h, f)


class StateODE:
    """ small container to handle the states in the ODE case """

    def __init__(self, step_size, f_evals, step_idx=None):
        """
        Parameters
        ----------
        step_size : float
        f_evals : list[np.ndarray]
        step_idx : int, optional
            index of step_size
        """
        self.f_evals = f_evals
        self.step_size = step_size
        self.step_idx = step_idx

    def flatten(self, use_idx=False):
        if not use_idx:
            return np.concatenate(([self.step_size], np.concatenate(self.f_evals)))
        return np.concatenate(([self.step_idx], np.concatenate(self.f_evals)))


class IntegratorODE:
    def __call__(self, state, x0):
        """
        Base Integrator call.

        Parameters
        ----------
        state : StateODE
            contains step_size and function evaluations f(t, x) for fitting inputs t and x
        x0 : np.ndarray
            initial state

        Returns
        -------
        np.ndarray
            state x after step_size
        """
        return 0

    @staticmethod
    def calc_state(t, x, h, f):
        """
        Parameters
        ----------
        t : float
            time
        x : np.ndarray
            state
        h : float
            step size
        f : FunctionODE

        Returns
        -------
        StateODE
            state
        """
        return StateODE(h, [f(x, t)])


class ClassicRungeKutta(IntegratorODE):
    def __call__(self, state, x0):
        """
        Classic Runge Kutta integration

        Parameters
        ----------
        state : StateODE
            contains step_size and function evaluations f(t, x) for fitting inputs t and x
        x0 : np.ndarray
            initial state

        Returns
        -------
        np.ndarray
            state after step_size
        """
        if len(state.f_evals) != 4:
            raise ValueError('Number of f_Evals in state has to be 4.')

        step_size = state.step_size
        k1, k2, k3, k4 = state.f_evals
        return x0 + step_size * (1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4)

    @staticmethod
    def calc_state(t, x, h, f):
        """
        Parameters
        ----------
        t : float
            time
        x : np.ndarray
            state
        h : float
            step size
        f : FunctionODE

        Returns
        -------
        StateODE
            state [h, k1, k2, k3, k4]
        """
        k1 = f(t, x)
        k2 = f(t + h / 2, x + h / 2 * k1)
        k3 = f(t + h / 2, x + h / 2 * k2)
        k4 = f(t + h, x + h * k3)
        return StateODE(h, [k1, k2, k3, k4])


class RKDP(IntegratorODE):
    """
    Dormand-Prince RK method of order 5 (6 stages).
    """

    def __init__(self):
        self.c = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
        self.b = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
        self.a = np.array([
            [0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0]
        ])

    def calc_state(self, t, x, h, f):
        """
        Parameters
        ----------
        t : float
            time
        x : np.ndarray
            state
        h : float
            step size
        f : FunctionODE

        Returns
        -------
        StateODE
            state [h, k1, k2, k3, k4]
        """
        k = []
        for i in range(len(self.c)):
            time = t + self.c[i] * h
            state = np.zeros(x.shape)
            for j in range(i):
                state += h * self.a[i, j] * k[j]
            k.append(f(time, x + state))

        return StateODE(h, k)

    def __call__(self, state, x0):
        """
        Dormand-Prince RK method (GoTo solver in ode45)

        Parameters
        ----------
        state : StateODE
            contains step_size and function evaluations f(t, x) for fitting inputs t and x
        x0 : np.ndarray
            initial state

        Returns
        -------
        np.ndarray
            state after step_size
        """
        if len(state.f_evals) != 6:
            raise ValueError('Number of f_Evals in state has to be 6.')

        step_size = state.step_size
        k = state.f_evals
        out = np.zeros(x0.shape)
        for idx in range(len(k)):
            out += step_size * self.b[idx] * k[idx]
        return x0 + out


class ExplicitRungeKutta(IntegratorODE):
    """Generic explicit Runge–Kutta single‑step integrator."""

    def __init__(self, c: Sequence[float], a: Sequence[Sequence[float]], b: Sequence[float]):
        self.c = np.asarray(c, dtype=float)
        self.b = np.asarray(b, dtype=float)
        a_matrix = np.zeros((len(self.c), len(self.c)))
        for i, row in enumerate(a):
            a_matrix[i, : len(row)] = row
        self.a = a_matrix

        if not (len(self.c) == len(self.b) == self.a.shape[0]):  # pragma: no cover
            raise ValueError("Inconsistent Butcher tableau dimensions.")

    def calc_state(self, t: float, x: np.ndarray, h: float, f) -> StateODE:  # type: ignore[override]
        k: list[np.ndarray] = []
        for i, ci in enumerate(self.c):
            ti = t + ci * h
            xi = x.astype(float).copy()
            if i:
                xi += h * sum(self.a[i, j] * k[j] for j in range(i))
            k.append(f(ti, xi))
        return StateODE(h, k)

    def __call__(self, state: StateODE, x0: np.ndarray) -> np.ndarray:  # type: ignore[override]
        h = state.step_size
        out = x0.astype(float).copy()
        for bi, ki in zip(self.b, state.f_evals):
            out += h * bi * ki
        return out


class CashKarp45(ExplicitRungeKutta):
    """Cash–Karp 5(4) explicit Runge–Kutta method (6 stages, order 5)."""

    def __init__(self):
        c = [0.0, 1 / 5, 3 / 10, 3 / 5, 1.0, 7 / 8]
        a = [
            [],
            [1 / 5],
            [3 / 40, 9 / 40],
            [3 / 10, -9 / 10, 6 / 5],
            [-11 / 54, 5 / 2, -70 / 27, 35 / 27],
            [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096],
        ]
        b = [37 / 378, 0.0, 250 / 621, 125 / 594, 0.0, 512 / 1771]
        super().__init__(c, a, b)


class Fehlberg78(ExplicitRungeKutta):
    """Fehlberg RKF 78 explicit Runge–Kutta method (13 stages, order 8)."""

    def __init__(self):
        c = [
            0.0,
            2 / 27,
            1 / 9,
            1 / 6,
            5 / 12,
            0.5,
            5 / 6,
            1 / 6,
            2 / 3,
            1 / 3,
            1.0,
            0.0,
            1.0,
        ]

        a = [
            [],
            [2 / 27],
            [1 / 36, 1 / 12],
            [1 / 24, 0.0, 1 / 8],
            [5 / 12, 0.0, -25 / 16, 25 / 16],
            [1 / 20, 0.0, 0.0, 1 / 4, 1 / 5],
            [-25 / 108, 0.0, 0.0, 125 / 108, -65 / 27, 125 / 54],
            [31 / 300, 0.0, 0.0, 0.0, 61 / 225, -2 / 9, 13 / 900],
            [2.0, 0.0, 0.0, -53 / 6, 704 / 45, -107 / 9, 67 / 90, 3.0],
            [
                -91 / 108,
                0.0,
                0.0,
                23 / 108,
                -976 / 135,
                3113 / 810,
                -19 / 60,
                17 / 6,
                -1 / 12,
            ],
            [
                2383 / 4100,
                0.0,
                0.0,
                -341 / 164,
                4496 / 1025,
                -301 / 82,
                2133 / 4100,
                45 / 82,
                45 / 82,
                18 / 41,
            ],
            [3 / 205, 0.0, 0.0, 0.0, 0.0, -6 / 41, -3 / 205, -3 / 41, 3 / 41, 6 / 41, 0.0],
            [
                -1777 / 4100,
                0.0,
                0.0,
                -341 / 164,
                4496 / 1025,
                -289 / 82,
                2193 / 4100,
                51 / 82,
                33 / 82,
                12 / 41,
                0.0,
                1.0,
            ],
        ]

        b = [
            41 / 840,
            0.0,
            0.0,
            0.0,
            34 / 105,
            9 / 35,
            9 / 35,
            9 / 280,
            9 / 280,
            41 / 840,
            0.0,
            0.0,
            0.0,
        ]

        super().__init__(c, a, b)


RKF78 = Fehlberg78


class GaussLegendre2Stage(IntegratorODE):
    """Implicit 2-stage Gauss–Legendre collocation (order 4)."""

    def __init__(self):
        # Butcher tableau for 2-stage Gauss–Legendre
        sqrt3 = np.sqrt(3)
        self.c = np.array([0.5 - sqrt3 / 6, 0.5 + sqrt3 / 6])
        self.A = np.array([
            [0.25, 0.25 - sqrt3 / 6],
            [0.25 + sqrt3 / 6, 0.25]
        ])
        self.b = np.array([0.5, 0.5])

    def calc_state(self, t, x, h, f):
        """
        Calculates implicit stage derivatives using root finding.

        Parameters
        ----------
        t : float
        x : np.ndarray
        h : float
        f : FunctionODE

        Returns
        -------
        StateODE
        """
        s = len(self.c)
        dim = x.shape[0]

        def residual(K_flat):
            K = K_flat.reshape((s, dim))
            R = []
            for i in range(s):
                ti = t + self.c[i] * h
                xi = x + h * sum(self.A[i, j] * K[j] for j in range(s))
                R.append(K[i] - f(ti, xi))
            return np.concatenate(R)

        # Initial guess: all zeros
        K0 = np.zeros((s, dim)).flatten()

        sol = root(residual, K0, method='hybr')

        if not sol.success:
            raise RuntimeError("Gauss–Legendre solver failed: " + sol.message)

        K = sol.x.reshape((s, dim))
        return StateODE(h, list(K))

    def __call__(self, state, x0):
        """
        Compute next step using implicit RK formula.

        Parameters
        ----------
        state : StateODE
        x0 : np.ndarray

        Returns
        -------
        np.ndarray
        """
        h = state.step_size
        K = state.f_evals
        x_next = x0 + h * sum(self.b[i] * K[i] for i in range(len(self.b)))
        return x_next
