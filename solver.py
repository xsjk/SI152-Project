import algorithm
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt

def QP_solver(
    A1: np.ndarray | None,
    A2: np.ndarray | None,
    b1: np.ndarray | None,
    b2: np.ndarray | None,
    g: np.ndarray,
    H: np.ndarray,
    dtype: np.dtype = np.float64,
    method: Literal["IRWA", "ADMM", "cvxpy"] = "IRWA",
    plot: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Solve the quadratic program

    minimize 0.5 x^T H x + g^T x

    subject to A1 x + b1 = 0
               A2 x + b2 <= 0

    Parameters
    ----------
    A1 : np.ndarray (m1, n)
        the equality constraint matrix
    A2 : np.ndarray (m2, n)
        the inequality constraint matrix
    b1 : np.ndarray (m1,)
        the equality constraint vector
    b2 : np.ndarray (m2,)
        the inequality constraint vector
    g : np.ndarray (n,)
        the coefficient vector of the linear term
    H : np.ndarray (n, n)
        the coefficient matrix of the quadratic term (must be symmetric)
    dtype : np.dtype (default=np.float64)
        the data type for computation
    method : Literal["IRWA", "ADMM"] (default="IRWA")
        the optimization algorithm to use
    plot : bool (default=False)
        whether to plot the optimization history
    verbose : bool (default=False)
        whether to print the optimization process

    Returns
    -------
    np.ndarray (n,)
        the optimal solution
    """

    assert H.ndim == 2
    assert g.ndim == 1

    # Check the inputs
    n = H.shape[0]
    m1 = 0 if A1 is None or b1 is None else A1.shape[0]
    m2 = 0 if A2 is None or b2 is None else A2.shape[0]
    m = m1 + m2

    if verbose:
        if m1 == 0:
            print("No equality constraints")
        if m2 == 0:
            print("No inequality constraints")

    if A1 is None or b1 is None:
        A1 = np.empty((0, n))
        b1 = np.empty(0)

    if A2 is None or b2 is None:
        A2 = np.empty((0, n))
        b2 = np.empty(0)

    assert H.shape == (n, n)
    assert g.shape == (n,)
    assert A1.shape == (m1, n)
    assert A2.shape == (m2, n)
    assert b1.shape == (m1,)
    assert b2.shape == (m2,)

    assert np.allclose(H, H.T), "H must be symmetric"

    A1 = A1.astype(dtype)
    A2 = A2.astype(dtype)
    b1 = b1.astype(dtype)
    b2 = b2.astype(dtype)
    H = H.astype(dtype)
    g = g.astype(dtype)

    # Extract m1, m2, n
    A = np.vstack([A1, A2])
    b = np.hstack([b1, b2])

    if plot and n != 2:
        print("Only plot 2D problems")
        plot = False


    if method == "IRWA":
        # Set the hyperparameters
        x0 = np.zeros(n, dtype=dtype)
        η = 0.9
        M = 1
        γ = 0.5
        σx = 1e-8
        σε = 1e-7

        x, histories = algorithm.IRWA(
            A, b, m1, g, H, x0, η, M, γ, σx, σε, max_retries=40, max_k=1000, verbose=verbose, return_history=plot
        )
        y = 0.5 * x @ H @ x + g @ x

        if plot:
            for i, h in enumerate(histories):
                points = np.vstack(h)
                plt.plot(points[:, 0], points[:, 1], "o-", color="red", alpha=0.5, markersize=5 / (i + 1))

    elif method == "ADMM":

        # Step 0. (Initialization)
        x0 = np.zeros(n, dtype=dtype)
        u0 = np.zeros(m, dtype=dtype)
        μ = 2
        σx = 1e-5
        σε = 1e-7

        x, history = algorithm.ADMM(
            A, b, m1, g, H, x0, u0, μ, σx, σε, max_k=5000, verbose=verbose, return_history=plot
        )

        # check the constraints
        c = A @ x + b
        assert np.all(c[m1:] <= 0 | np.isclose(c[m1:], 0, atol=max(σx, σε))), c[m1:]
        assert np.allclose(c[:m1], 0, atol=σε), c[:m1]

        if verbose > 0:
            print("k:", len(history))

        if plot:
            points = np.vstack(history)
            plt.plot(points[:, 0], points[:, 1], "o-", color="red", alpha=0.5, markersize=5)

        y = 0.5 * x @ H @ x + g @ x

    elif method == "cvxpy":

        import cvxpy
        x = cvxpy.Variable(n)
        obj = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, H) + g @ x)
        c1 = [A1 @ x + b1 == 0] if m1 > 0 else []
        c2 = [A2 @ x + b2 <= 0] if m2 > 0 else []
        prob = cvxpy.Problem(obj, c1 + c2)
        y = prob.solve(solver=cvxpy.CLARABEL)
        x = x.value

        if plot:
            print("Cannot plot cvxpy results")
            plot = False

    else:
        raise ValueError("Invalid method")

    if plot and n == 2:

        xlim = np.array(plt.xlim())
        ylim = np.array(plt.ylim())

        X, Y = np.meshgrid(np.linspace(*xlim, 1000),
                           np.linspace(*ylim, 1000))
        P = np.array([X, Y]).transpose(1, 2, 0)[..., None]
        PT = P.transpose(0, 1, 3, 2)

        # plot equality constraints
        for i in range(m1):
            plt.plot(xlim, (-b1[i] - A1[i, 0] * xlim) / A1[i, 1], "--", alpha=0.5, linewidth=3, color="brown", label="Equality constraints" if i == 0 else None)

        # plot inequality constraints
        Z = np.all(A2 @ P + b2[..., None] <= 0, axis=(2, 3))
        plt.contourf(X, Y, Z, levels=1, alpha=0.5, cmap="coolwarm")

        # plot the objective function
        Z = (0.5 * PT @ H @ P + g[None] @ P)[..., 0, 0]
        plt.contour(X, Y, Z, levels=20, colors="black", alpha=0.5, linestyles="--", linewidths=1)

        plt.plot(x[0], x[1], "bo", markersize=10, label="Optimal solution", zorder=10, alpha=0.5)
        plt.plot(x0[0], x0[1], "ro", markersize=10, label="Initial solution", zorder=10, alpha=0.5)

        plt.legend()

        plt.xlim(xlim)
        plt.ylim(ylim)

    if verbose:
        print("Optimal value:", y)
        print("Optimal solution:", x)


    return x

