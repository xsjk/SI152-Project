import numpy as np
from typing import Literal
from numba import njit

@njit
def IRWA(A, b, l, g, H, x0, η, M, γ, σx, σε, max_retries=10, max_k=1000, verbose=False, return_history=False):

    m = A.shape[0]
    n = g.shape[0]

    ε = np.empty((m,))
    dx = np.full((n,), np.inf)
    w = np.empty((m,))
    v = np.empty((m,))

    histories = []

    trial = 0

    while trial < max_retries:

        x = x0
        ε[:] = 1
        k = 0

        if return_history:
            histories.append([x])

        ## Step 3. (Check stopping criteria)
        while not (np.sum(dx**2) < σx**2 and np.sum(ε**2) < σε**2):

            if verbose > 1:
                print(x)

            ## Step 1. (Solve the reweighted subproblem for x^(k+1))
            Ax = A @ x
            c = Ax + b

            w[:l] = np.abs(c[:l])
            w[l:] = np.maximum(c[l:], 0)

            W = np.diag(1 / np.sqrt(w**2 + ε**2))

            v[:l] = b[:l]
            v[l:] = np.maximum(b[l:], -Ax[l:])

            x_ = np.linalg.solve(H + A.T @ W @ A, -g - A.T @ W @ v)
            dx[:] = x_ - x

            ## Step 2. (Set the new relaxation vector ϵ) Set
            q = A @ dx
            r = (1 - v) * c

            if np.all(np.abs(q) <= M * (r**2 + ε**2) ** (0.5 + γ)):
                ε_ = ε * η
            else:
                ε_ = ε

            x = x_
            ε = ε_
            k += 1

            if return_history:
                histories[-1].append(x)

            if k > max_k:
                break

        if verbose > 0:
            print(f"IRWA finished in {k} iterations.")


        # Standard IRWA finished
        # Check if the solution satisfies the constraints
        if np.allclose(c[:l], 0, atol=σε) and np.all(
            c[l:] < 0 | np.isclose(c[l:], 0, atol=σε)
        ) and np.allclose(x0, x, atol=σx):
            break
        else:
            # The solution does not satisfy the constraints,
            # rescale the i-th constraint
            if m > 0:
                i = np.argmax(w)
                scale = np.log(1 + (w.max() + σε) / (w.min() + σε) + w.max() / σε) + 1
                A[i] *= scale
                b[i] *= scale
                if verbose > 0:
                    print(f"Rescale the {i}-th constraint by", np.round(scale, 2))

            x0 = x

        trial += 1

        if trial > max_retries:
            if verbose > 0:
                print("The algorithm did not converge.")
            break

    if verbose > 0:
        print(f"IRWA finished in {trial} trials.")

    return x, histories


@njit
def ADMM(A, b, l, g, H, x0, u0, μ, σx, σε, max_k=10000, verbose=0, return_history=False):

    n = g.shape[0]
    m = A.shape[0]

    K = np.linalg.inv(H + μ * A.T @ A)

    p = np.zeros(m)
    p_ = np.empty(m)
    c = np.zeros(m)
    dx = np.full(n, np.inf)

    u = u0
    x = x0

    xs = []
    if return_history:
        xs.append(x)

    k = 0

    ## Step 3. (Check stop criteria)
    while not (np.sum(dx**2) < σx**2 and (m == 0 or np.amax(np.abs(c - p)) <= σε)):

        if verbose > 1:
            print(x, u)

        ## Step 1. (Solve the augmented Lagrangian subproblems for x_ and p_)
        x_ = K @ -(g + A.T @ (u + μ * (b - p)))
        dx = x_ - x
        c_ = A @ x_ + b
        p_[:l] = np.sign(c_[:l]) * np.maximum(np.abs(c_[:l]) - 1 / μ, 0)
        p_[l:] = np.minimum(c_[l:], np.maximum(c_[l:] - 1 / μ, 0))

        ## Step 2. (Set the new multipliers u_)
        u_ = u + (c_ - p_) / μ

        x = x_
        u = u_
        c = c_
        p = p_
        k += 1

        if return_history:
            xs.append(x)

        if k > max_k:
            if verbose > 0:
                print("The algorithm did not converge.")
            break

    return x, xs


