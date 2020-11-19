from ..systems import FiniteHorizonControlSystem
import gin

import jax.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@gin.configurable
class Glucose(FiniteHorizonControlSystem):
    def __init__(self, a, b, c, A, l, x_0, T):
        """
        Taken from: Optimal Control Applied to Biological Models, Lenhart & Workman (Chapter 16, Lab 10)
        Model is presented in more details in Martin Eisen. Mathematical Methods and Models in the Biological Sciences.
        Prentice Hall, Englewood Cliffs, New Jersey, 1988.

        This environment try to model the blood glucose (x_0(t0) level of a diabetic person in the presence of injected
        insulin (u(t)) and the net hormonal concentration (x_1(t)) of the insulin in the person system. In this model,
        the diabetic person is assumed to be unable to produce natural insulin via its pancreas.

        Note that the model was developed for regulating blood glucose levels over a short window of time. As thus, T
        should be kept under 0.45 for the model to make sense (T here is measured in day, 0.45 corresponds to ~11 hours)

        The goal of the control is to maintain the blood glucose level close to a desired level, l, while also taking
        into account that there is a cost associated to the treatment. Thus the objective is:

        .. math::

            \min_{u} \quad &\int_0^T A(x_0(t)-l)^2 + u_f(t)^2  dt \\
            \mathrm{s.t.}\qquad & x_0'(t) = -ax_0(t) - bx_1(t) ,\; x_0(0) > 0 \\
            & x_1'(t) = -cx_1(t) + u(t) ,\; x_1(0)=0 \\
            & a,b,c > 0 \; A \geq 0

        :param a: Rate of decrease in glucose level resulting of its use by the body
        :param b: Rate of decrease in glucose level resulting from its degradation provoked by insulin
        :param c: Rate of degradation of the insulin
        :param A: Weight parameter balancing the objective
        :param l: Desired level of blood glucose
        :param x_0: Initial blood glucose level and insulin level (x_0,x_1)
        :param T: Horizon (Should be kept under 0.45)
        """
        self.adj_T = None # Final condition over the adjoint, if any
        self.a = a
        self.b = b
        self.c = c
        self.A = A
        self.l = l

        super().__init__(
            x_0=np.array([
                x_0[0],
                x_0[1],
            ]),                     # Starting state
            x_T=None,               # Terminal state, if any
            T=T,                    # Duration of experiment
            bounds=np.array([       # Bounds over the states (x_0, x_1 ...) are given first,
                [np.NINF, np.inf],      # followed by bounds over controls (u_0,u_1,...)
                [np.NINF, np.inf],
                [np.NINF, np.inf],
            ]),
            terminal_cost=False,
            discrete=False,
        )

    def dynamics(self, x_t: np.ndarray, u_t: np.ndarray, v_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        d_x = np.asarray([
            -self.a*x_t[0] -self.b*x_t[1],
            -self.c*x_t[1] + u_t[0]
            ])

        return d_x

    def cost(self, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> float: ## TODO : rename for max problem?
        return self.A*(x_t[0]-self.l)**2 + u_t[0]**2

    def adj_ODE(self, adj_t: np.ndarray, x_t: np.ndarray, u_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.array([
            -2*self.A*(x_t[0]-self.l) +adj_t[0]*self.a,
            adj_t[0]*self.b + adj_t[1]*self.c
        ])

    def optim_characterization(self, adj_t: np.ndarray, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        char_0 = -adj_t[:,1]/2
        char_0 = char_0.reshape(-1,1)

        return char_0

    def plot_solution(self, x: np.ndarray, u: np.ndarray, adj: np.array) -> None:
        sns.set(style='darkgrid')
        plt.figure(figsize=(12,12))

        x, u, adj = x.T, u.T, adj.T

        ts_x = np.linspace(0, self.T, x[0].shape[0])
        ts_u = np.linspace(0, self.T, u[0].shape[0])
        ts_adj = np.linspace(0, self.T, adj[0].shape[0])

        labels = ["Blood Glucose", "Net Hormonal Concentration"]

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
            plt.plot(ts_u, u_i, label='Insulin level')
        plt.title("Optimal control of dynamic system via forward-backward sweep")
        plt.ylabel("control (u)")

        plt.subplot(3, 1, 3)
        for idx, adj_i in enumerate(adj):
            if idx in to_print:
                plt.plot(ts_adj, adj_i)
        plt.title("Optimal adjoint of dynamic system via forward-backward sweep")
        plt.ylabel("adjoint (lambda)")

        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.show()