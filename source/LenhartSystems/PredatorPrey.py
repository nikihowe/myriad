from ..systems import FiniteHorizonControlSystem
import gin

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@gin.configurable
class PredatorPrey(FiniteHorizonControlSystem):
    def __init__(self, d_1, d_2, A, B, guess_a, guess_b, M, x_0, T):
        """
        Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 22, Lab 13)
        The states evolution is base on a standard Lotka-Volterra model.
        This particular environment is inspired from Bean San Goh, George Leitmann, and Thomas L. Vincent.
        Optimal control of a prey-predator system. Mathematical Biosciences, 19, 1974.

        This environment model the evolution of a pest (prey) population (x_0(t)) and a predator population (x_1(t)) in
        the presence of a pesticide (u(t)) that affects both the pest and predator population. The objective in mind is
        to minimize the final pest population, while limiting the cost usage of the pesticide. Thus:

        .. math::

            \min_{u} \quad & x_0(T) + \frac{A}{2}\int_0^T u(t)^2 dt \\
            \mathrm{s.t.}\qquad & x_0'(t) = (1 - x_1(t))x_0(t) - d_1x_0(t)u(t) \\
            & x_1'(t) = (x_0(t) - 1)x_1(t) - d_2x_1(t)(t)u(t) \\
            & 0 \leq u(t) \leq M, \quad \int_0^T u(t) dt = B

        The particularity here is that the total amount of pesticide to be applied is fixed. To take into account this
        constraint, a virtual state variable (z(t)) is added where:

        .. math::

            z'(t) = u(t), \; z(0) = 0, \; z(T) = B

        Finally, note that guess_a and guess_b have been carefully chosen in the study cases to allow for fast iteration
        and ensure convergence.

        :param d_1: Impact of the pesticide on the pest population
        :param d_2: Impact of the pesticide on the prey population
        :param A: Weight parameter balancing the cost
        :param B: Total amount of pesticide to be applied
        :param guess_a: Node 2 at which the secant method begins its iteration (Newton's method)
        :param guess_b: Node 1 at which the secant method begins its iteration (Newton's method)
        :param M: Bound on pesticide application at a given time
        :param x_0: Initial density of the pest and prey population (x_0, x_1)
        :param T: Horizon
        """
        self.adj_T = np.array([1, 0, 0]) # Final condition over the adjoint, if any
        self.d_1 = d_1
        self.d_2 = d_2
        self.A = A
        self.guess_a = guess_a
        self.guess_b = guess_b

        super().__init__(
            x_0=np.array([
                x_0[0],
                x_0[1],
                x_0[2]
            ]),                     # Starting state
            x_T=[None, None, B],    # Terminal state, if any
            T=T,                    # Duration of experiment
            bounds=np.array([       # Bounds over the states (x_0, x_1 ...) are given first,
                [np.NINF, np.inf],      # followed by bounds over controls (u_0,u_1,...)
                [np.NINF, np.inf],
                [np.NINF, np.inf],
                [0, M]
            ]),
            terminal_cost=True,
            discrete=False,
        )

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray = None, t: np.ndarray = None) -> np.ndarray:
        x_0, x_1, x_2 = x_t
        if u_t.ndim > 0:
            u_t, = u_t

        d_x = np.array([
            (1 - x_1)*x_0 - self.d_1*x_0*u_t,
            (x_0 - 1)*x_1 - self.d_2*x_1*u_t,
            u_t,
            ])

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float:
        return self.A*0.5*u_t[0]**2

    def terminal_cost_fn(self, x_T: np.ndarray, u_T: np.ndarray, T: np.ndarray=None) -> float:
        return x_T[0]

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.array([
            adj_t[0]*(x_t[1] -1 + self.d_1*u_t[0]) - adj_t[1]*x_t[1],
            adj_t[0]*x_t[0] + adj_t[1]*(1 - x_t[0] + self.d_2*u_t[0]),
            0
        ])

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        char = (adj_t[:, 0]*self.d_1*x_t[:, 0] + adj_t[:, 1]*self.d_2*x_t[:, 1] - adj_t[:, 2])/self.A
        char = char.reshape(-1,1)

        return np.minimum(self.bounds[-1, 1], np.maximum(self.bounds[-1, 0], char))

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array = None) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))

        if adj is None:
            adj = u.copy()
            flag = False
        else:
            flag = True

        x, u, adj = x.T, u.T, adj.T

        ts_x = np.linspace(0, self.T, x[0].shape[0])
        ts_u = np.linspace(0, self.T, u[0].shape[0])
        ts_adj = np.linspace(0, self.T, adj[0].shape[0])

        labels = ["Prey Population", "Predator Population"]

        to_print = [0,1] #curves we want to print out

        plt.subplot(3, 1, 1)
        for idx, x_i in enumerate(x):
            if idx in to_print:
                plt.plot(ts_x, x_i, label=labels[idx])
        plt.legend()
        plt.title("Optimal state of dynamic system via forward-backward sweep")
        plt.ylabel("state (x)")

        plt.subplot(3, 1, 2)
        for idx, u_i in enumerate(u):
            plt.plot(ts_u, u_i, label='Pesticide level')
        plt.legend()
        plt.title("Optimal control of dynamic system via forward-backward sweep")
        plt.ylabel("control (u)")

        if flag:
            plt.subplot(3, 1, 3)
            for idx, adj_i in enumerate(adj):
                if idx in to_print:
                    plt.plot(ts_adj, adj_i)
            plt.title("Optimal adjoint of dynamic system via forward-backward sweep")
            plt.ylabel("adjoint (lambda)")

        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.show()