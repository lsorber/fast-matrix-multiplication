import numpy as np
import pulp

from itertools import permutations


def multiplication_tensor(N=2):
    """Multiplication tensor.

    The multiplication tensor T of order N is defined by:

        C == A @ B <=> vec(C) == T x1 vec(A.T) x2 vec(B.T)

    where A, B, and C are N x N matrices and vec is the column-wise
    vectorization operator.
    """
    T = np.zeros((N ** 2, N ** 2, N ** 2), dtype=np.int64)
    for n in range(N):
        for m in range(N):
            u = np.ravel_multi_index(
                (n * np.ones(N, dtype=np.int64), np.arange(N)), (N, N))
            v = np.ravel_multi_index(
                (np.arange(N), m * np.ones(N, dtype=np.int64)), (N, N))
            w = np.ravel_multi_index(
                (m, n), (N, N)) * np.ones(len(u), dtype=np.int64)
            T[u, v, w] = 1
    # Assert cyclic symmetry.
    assert np.all(T == np.transpose(T, [2, 0, 1]))
    assert np.all(T == np.transpose(T, [1, 2, 0]))
    return T


def vecperm(N=2):
    """Vec-permutation matrix.

    The vec-permutation matrix P of order N is defined by [1]:

        vec(A) == P @ vec(A.T)

    where A is an N x N matrix.

    [1] https://ecommons.cornell.edu/bitstream/handle/1813/32747/BU-645-M.pdf
    """
    A = np.reshape(np.arange(N ** 2), (N, N), order='F')
    i, j = np.arange(N ** 2), np.ravel(A.T, order='F')
    P = np.zeros((N ** 2, N ** 2))
    P[i, j] = 1
    return P


def matrixperms(N=2):
    """Permutation matrices of order N."""
    P = np.zeros((N, N, np.math.factorial(N)))
    for k, p in enumerate(permutations(range(N))):
        i, j = np.arange(N), np.array(p)
        P[i, j, k] = 1
    return P


def expand_pd(U, V, W):
    """Expand a polyadic decomposition.

    The polyadic expansion T of the factor matrices U, V, and W is defined by:

        T[i, j, k] = \sum_r U[i, r] * V[j, r] * W[k, r].
    """
    I, J, K, R = U.shape[0], V.shape[0], W.shape[0], U.shape[1]
    T = np.zeros((I, J, K))
    for i in range(I):
        for j in range(J):
            for k in range(K):
                for r in range(R):
                    T[i, j, k] += U[i, r] * V[j, r] * W[k, r]
    return T


def solution_N2_R7():
    A = np.array([
        [1],
        [0],
        [0],
        [1]
    ])
    B = np.array([
        [1,  0],
        [0, -1],
        [1,  0],
        [0, -1]
    ])
    C = np.array([
        [0,  1],
        [0,  0],
        [0,  0],
        [-1, 0]
    ])
    D = np.array([
        [1,  0],
        [-1, 0],
        [0, -1],
        [0,  1]
    ])
    U = np.hstack((A, B, C, D))
    V = np.hstack((A, D, B, C))
    W = np.hstack((A, C, D, B))
    # Assert the solution is valid.
    N = 2
    P = vecperm(N)
    assert np.all(expand_pd(U, V, W) == multiplication_tensor(N))
    assert np.all(expand_pd(W, U, V) == multiplication_tensor(N))
    assert np.all(expand_pd(V, W, U) == multiplication_tensor(N))
    assert np.all(expand_pd(P @ V, P @ U, P @ W) == multiplication_tensor(N))
    assert np.all(expand_pd(P @ U, P @ W, P @ V) == multiplication_tensor(N))
    assert np.all(expand_pd(P @ W, P @ V, P @ U) == multiplication_tensor(N))
    return U, V, W


def solution_N3_R23():
    A = np.array([
        [0, 1,  0, 0,  0],
        [0, 0,  0, 0,  0],
        [0, 0,  0, 0,  0],
        [0, 0,  0, 0,  0],
        [1, 0,  0, 0,  0],
        [0, 0, -1, 0,  1],
        [0, 0,  0, 0,  0],
        [0, 0,  0, 1, -1],
        [0, 0,  1, 1, -1]
    ])
    B = np.array([
        [0,  0, 0,  0,  0,  0],
        [0,  0, 0,  0,  0, -1],
        [1,  0, 0, -1,  0,  0],
        [0,  0, 0,  0,  1,  0],
        [0,  0, 0,  0,  0,  0],
        [0,  0, 0,  0,  0,  0],
        [0,  0, 1,  0, -1,  0],
        [-1, 1, 0,  0,  0, -1],
        [0,  0, 0,  0,  0,  0]
    ])
    C = np.array([
        [-1, -1,  0,  1,  0, -1],
        [1,   1,  0, -1,  0,  0],
        [1,   1, -1, -1,  0,  0],
        [0,   0,  1,  0,  0,  0],
        [0,   0, -1,  0, -1,  0],
        [0,   1, -1,  0,  0,  0],
        [-1, -1,  0,  0,  0, -1],
        [0,   0,  1,  0,  1,  0],
        [0,   0,  0,  0,  0,  0]
    ])
    D = np.array([
        [0,   0,  0,  0,  0, 0],
        [0,   0, -1,  0, -1, 0],
        [0,   0, -1,  0,  0, 0],
        [-1,  0,  0,  1,  0, 1],
        [1,   1,  0, -1,  0, 0],
        [1,   1,  0, -1,  0, 0],
        [0,   0,  0, -1,  0, 0],
        [-1, -1,  0,  1,  0, 0],
        [-1, -1,  0,  1,  0, 0]
    ])
    # Construct the factor matrices.
    U = np.hstack((A, B, C, D))
    V = np.hstack((A, D, B, C))
    W = np.hstack((A, C, D, B))
    # Assert the solution is valid.
    N = 3
    P = vecperm(N)
    assert np.all(expand_pd(U, V, W) == multiplication_tensor(N))
    assert np.all(expand_pd(W, U, V) == multiplication_tensor(N))
    assert np.all(expand_pd(V, W, U) == multiplication_tensor(N))
    assert np.all(expand_pd(P @ V, P @ U, P @ W) == multiplication_tensor(N))
    assert np.all(expand_pd(P @ U, P @ W, P @ V) == multiplication_tensor(N))
    assert np.all(expand_pd(P @ W, P @ V, P @ U) == multiplication_tensor(N))
    return U, V, W


def construct_instance(
        N=2, R=7, S=None, T=None,
        factors_equal_nz=True,
        factors_nz_lb=True,
        factors_col_nz_lb=1,
        factors_row_nz_lb=1,
        value_bound=False,
        cyclic_symmetry=True,
        transpose_symmetry=False,
        permutation_symmetry=False):
    """Set up a MILP formulation of the fast matrix multiplication problem.

    The MILP's objective is threefold:

        1. Minimize the L1 norm between the multiplication tensor T_ijk of
           order N and an approximating tensor \sum_r VALUE_ijkr = [[U, V, W]],
           where VALUE_ijkr := U_ir * V_jr * W_kr and U, V, and W are factor
           matrices of size N ** 2 by R.
        2. Minimize the L1 norm of the factor matrices U, V, and W to obtain
           the most sparse solution.
        3. Minimize the distance between two measures of the absolute value of
           the factor matrices. One of the two measures is correct if all
           variables are integer. The other is closer to the actual absolute
           value when the variables are still fractional.

    Args:
        N (int): The order of the matrices that are multiplied.
        R (int): How many scalar multiplications should be used to perform the
            matrix multiplication. The naive algorithm uses N ** 3 multiplies.
            It is known that for N=2: R == 7, for N=3: 19 <= R <= 23, and for
            N=4: 34 <= R <= 49.
        S (int): Optional. Imposes cyclic invariance on the factor matrices if
            set together with T. The factor matrices will take on the form

                U := [A, B, C, D], V := [A, D, B, C], W:= [A, C, D, B]

            where A is an N ** 2 by S matrix, and B, C, and D are N ** 2 by T
            matrices. The rank R is therefore equal to S + 3 * T. Imposing this
            structure reduces the number of integer variables by a factor of
            three.
        T (int): Optional, see S.
        factors_equal_nz (bool): If True, adds a constraint that requires all
            factor matrices to have an equal number of nonzeros.
        factors_nz_lb (bool): If True, adds a constraint that sets a lower
            bound on the total number of nonzeros in the factor matrices as
            3 * N ** 3 + R. If factors_equal_nz is also True, this bound is
            rounded up to the nearest multiple of three.
        factors_col_nz_lb (int): The minimal number of nonzeros per column in
            each factor matrix. Set to 1 to impose each rank-1 term to be
            nonzero.
        factors_row_nz_lb (int): The minimal number of nonzeros per row in each
            factor matrix. Should be at least 1 so that the approximating
            tensor does not contain any all-zero slices.
        value_bound (bool): If True, adds a constraint that requires the
            approximating tensor to have values between 0 and 1.
        cyclic_symmetry (bool): If True, adds a constraint that requires the
            approximating tensor to be cyclically symmetric. If S and T are
            positive integers, this option will have no effect since the
            constraint is automatically satisfied.
        transpose_symmetry (bool): If True, adds a constraint that requires the
            approximating tensor to satisfy the transpose symmetry of matrix
            multiplication.
        permutation_symmetry (bool): If True, adds a constraint that requires
            the approximating tensor to satisfy a certain permutation property
            of matrix multiplication.

    Returns:
        prob (pulp.LpProblem): The resulting MILP problem.
    """
    # Construct the variables.
    factors = [
        pulp.LpVariable.dicts(
            factor_name, indexs=(range(N ** 2), range(R)),
            lowBound=-1, upBound=1,
            cat='Continuous')
        for factor_name in ['U', 'V', 'W']
    ]
    abs_factors = [
        pulp.LpVariable.dicts(
            factor_name, indexs=(range(N ** 2), range(R)),
            lowBound=0, upBound=1,
            cat='Continuous')
        for factor_name in ['ABS_U', 'ABS_V', 'ABS_W']
    ]
    abs_alt_factors = [
        pulp.LpVariable.dicts(
            factor_name, indexs=(range(N ** 2), range(R)),
            lowBound=0, upBound=1,
            cat='Continuous')
        for factor_name in ['ABS_ALT_U', 'ABS_ALT_V', 'ABS_ALT_W']
    ]
    abs_diff_factors = [
        pulp.LpVariable.dicts(
            factor_name, indexs=(range(N ** 2), range(R)),
            lowBound=0, upBound=1,
            cat='Continuous')
        for factor_name in ['ABS_DIFF_U', 'ABS_DIFF_V', 'ABS_DIFF_W']
    ]
    cyclic_invariant = S is not None or T is not None
    if cyclic_invariant:
        assert R == S + 3 * T
        neg_variables = [
            pulp.LpVariable.dicts(
                'NEG_A', indexs=(range(N ** 2), range(S)),
                lowBound=0, upBound=1,
                cat='Binary')] + \
            [pulp.LpVariable.dicts(
                factor_name, indexs=(range(N ** 2), range(T)),
                lowBound=0, upBound=1,
                cat='Binary') for factor_name in ['NEG_B', 'NEG_C', 'NEG_D']]
        pos_variables = [
            pulp.LpVariable.dicts(
                'POS_A', indexs=(range(N ** 2), range(S)),
                lowBound=0, upBound=1,
                cat='Binary')] + \
            [pulp.LpVariable.dicts(
                factor_name, indexs=(range(N ** 2), range(T)),
                lowBound=0, upBound=1,
                cat='Binary') for factor_name in ['POS_B', 'POS_C', 'POS_D']]
    else:
        neg_variables = \
            [pulp.LpVariable.dicts(
                factor_name, indexs=(range(N ** 2), range(R)),
                lowBound=0, upBound=1,
                cat='Binary') for factor_name in ['NEG_U', 'NEG_V', 'NEG_W']]
        pos_variables = \
            [pulp.LpVariable.dicts(
                factor_name, indexs=(range(N ** 2), range(R)),
                lowBound=0, upBound=1,
                cat='Binary') for factor_name in ['POS_U', 'POS_V', 'POS_W']]
    value = pulp.LpVariable.dicts(
        'VALUE',
        indexs=(range(N ** 2), range(N ** 2), range(N ** 2), range(R)),
        lowBound=-1, upBound=1,
        cat='Continuous')
    residual = pulp.LpVariable.dicts(
        'RESIDUAL',
        indexs=(range(N ** 2), range(N ** 2), range(N ** 2)),
        lowBound=0, upBound=1,
        cat='Continuous')
    # Construct the objective function.
    if cyclic_invariant:
        prob = pulp.LpProblem(
            'fastxgemm-N{N}-R{R}-S{S}-T{T}'.format(
                N=N, R=R, S=S, T=T), pulp.LpMinimize)
    else:
        prob = pulp.LpProblem(
            'fastxgemm-N{N}-R{R}'.format(N=N, R=R), pulp.LpMinimize)
    sep_factors_nz = int(10 ** np.ceil(np.log10(3 * N ** 2 * R)))
    prob += \
        sep_factors_nz * pulp.lpSum(residual) + \
        pulp.lpSum(abs_alt_factors) + \
        (1 / sep_factors_nz) * pulp.lpSum(abs_diff_factors)
    # Define the factor matrix entries and their absolute values:
    #    factors = pos_variables - neg_variables
    #    abs_factors = pos_variables + neg_variables
    #    pos + neg <= 1 is automatically satisfied by the bounds on abs_factors
    #    If requested, impose cyclic invariant structure on the factor matrices
    if cyclic_invariant:
        cycles = [(0, 1, 2, 3), (0, 3, 1, 2), (0, 2, 3, 1)]
        for f, cycle in enumerate(cycles):
            for n in range(N ** 2):
                r = 0
                for v in cycle:
                    for t in range(len(neg_variables[v][n])):
                        prob += factors[f][n][r] == \
                            pos_variables[v][n][t] - neg_variables[v][n][t]
                        prob += abs_factors[f][n][r] == \
                            pos_variables[v][n][t] + neg_variables[v][n][t]
                        r += 1
                assert r == R
    else:
        for f in range(len(factors)):
            for n in range(N ** 2):
                for r in range(R):
                    prob += factors[f][n][r] == \
                        pos_variables[f][n][r] - neg_variables[f][n][r]
                    prob += abs_factors[f][n][r] == \
                        pos_variables[f][n][r] + neg_variables[f][n][r]
    # Add an alternative to abs_factors, and measure their difference:
    #    -abs_alt_factors <= factors <= abs_alt_factors
    #    -abs_diff_factors <= abs_factors - abs_alt_factors <= abs_diff_factors
    for f in range(len(factors)):
        for n in range(len(factors[f])):
            for r in range(len(factors[f][n])):
                prob += -abs_alt_factors[f][n][r] <= factors[f][n][r]
                prob += factors[f][n][r] <= abs_alt_factors[f][n][r]
                prob += -abs_diff_factors[f][n][r] <= \
                    abs_alt_factors[f][n][r] - abs_factors[f][n][r]
                prob += abs_alt_factors[f][n][r] - abs_factors[f][n][r] <= \
                    abs_diff_factors[f][n][r]
    # Force the value of the rank-one tensors:
    #    (1,1,1), (-1,-1,-1)
    #     U_ir + V_jr + W_kr - 2 <= VALUE_ijkr <= U_ir + V_jr + W_kr + 2
    #    (1,1,-1), (1,-1,-1)
    #     U_ir - V_jr - W_kr - 2 <= VALUE_ijkr <=  U_ir - V_jr - W_kr + 2
    #    -U_ir - V_jr + W_kr - 2 <= VALUE_ijkr <= -U_ir - V_jr + W_kr + 2
    #    -U_ir + V_jr - W_kr - 2 <= VALUE_ijkr <= -U_ir + V_jr - W_kr + 2
    #    (0,x,y)
    #    -ABS_U_ir <= VALUE_ijkr <= ABS_U_ir
    #    -ABS_V_jr <= VALUE_ijkr <= ABS_V_jr
    #    -ABS_W_kr <= VALUE_ijkr <= ABS_W_kr
    for i in range(N ** 2):
        for j in range(N ** 2):
            for k in range(N ** 2):
                for r in range(R):
                    prob += \
                        factors[0][i][r] + \
                        factors[1][j][r] + \
                        factors[2][k][r] - 2 <= value[i][j][k][r]
                    prob += \
                        factors[0][i][r] + \
                        factors[1][j][r] + \
                        factors[2][k][r] + 2 >= value[i][j][k][r]
                    for f in range(len(factors)):
                        prob += \
                            ((f == 0) * 2 - 1) * factors[0][i][r] + \
                            ((f == 1) * 2 - 1) * factors[1][j][r] + \
                            ((f == 2) * 2 - 1) * factors[2][k][r] - 2 \
                            <= value[i][j][k][r]
                        prob += \
                            ((f == 0) * 2 - 1) * factors[0][i][r] + \
                            ((f == 1) * 2 - 1) * factors[1][j][r] + \
                            ((f == 2) * 2 - 1) * factors[2][k][r] + 2 \
                            >= value[i][j][k][r]
                    prob += -abs_factors[0][i][r] <= value[i][j][k][r]
                    prob += value[i][j][k][r] <= abs_factors[0][i][r]
                    prob += -abs_factors[1][j][r] <= value[i][j][k][r]
                    prob += value[i][j][k][r] <= abs_factors[1][j][r]
                    prob += -abs_factors[2][k][r] <= value[i][j][k][r]
                    prob += value[i][j][k][r] <= abs_factors[2][k][r]
    # Define the residual:
    #    -RESIDUAL_ijk <= T_ijk - \sum_r VALUE_ijkr <= RESIDUAL_ijk
    T = multiplication_tensor(N)
    for i in range(N ** 2):
        for j in range(N ** 2):
            for k in range(N ** 2):
                prob += -residual[i][j][k] <= T[i][j][k] - pulp.lpSum(
                    value[i][j][k][r] for r in range(R))
                prob += T[i][j][k] - pulp.lpSum(
                    value[i][j][k][r] for r in range(R)) <= residual[i][j][k]
    # Optimization 0: All factor matrices have an equal number of nonzeros.
    if factors_equal_nz and not cyclic_invariant:
        for f in range(1, len(factors)):
            prob += pulp.lpSum(
                abs_factors[0][i][r]
                for i in range(N ** 2)
                for r in range(R)) == pulp.lpSum(
                abs_factors[f][i][r]
                for i in range(N ** 2)
                for r in range(R))
            prob += pulp.lpSum(
                abs_alt_factors[0][i][r]
                for i in range(N ** 2)
                for r in range(R)) == pulp.lpSum(
                abs_alt_factors[f][i][r]
                for i in range(N ** 2)
                for r in range(R))
    # Optimization 1: Lower bound on the number of nonzeros in factor matrices.
    if factors_nz_lb:
        lb = 3 * N ** 3 + 1
        prob += pulp.lpSum(
            abs_factors[f][i][r]
            for f in range(len(factors))
            for i in range(N ** 2)
            for r in range(R)) >= \
            3 * int(np.ceil(lb / 3)) \
            if factors_equal_nz or cyclic_invariant else lb
        prob += pulp.lpSum(
            abs_alt_factors[f][i][r]
            for f in range(len(factors))
            for i in range(N ** 2)
            for r in range(R)) >= \
            3 * int(np.ceil(lb / 3)) \
            if factors_equal_nz or cyclic_invariant else lb
    # Optimization 2: Lower bound on the number of nonzeros in factor columns.
    if factors_col_nz_lb > 0:
        for f in range(len(factors)):
            for r in range(R):
                prob += pulp.lpSum(
                    abs_factors[f][i][r] for i in range(N ** 2)) >= \
                    factors_col_nz_lb
                prob += pulp.lpSum(
                    abs_alt_factors[f][i][r] for i in range(N ** 2)) >= \
                    factors_col_nz_lb
    # Optimization 3: Lower bound on the number of nonzeros in factor rows.
    if factors_row_nz_lb > 0:
        for f in range(len(factors)):
            for i in range(N ** 2):
                prob += pulp.lpSum(
                    abs_factors[f][i][r] for r in range(R)) >= \
                    factors_row_nz_lb
                prob += pulp.lpSum(
                    abs_alt_factors[f][i][r] for r in range(R)) >= \
                    factors_row_nz_lb
    # Optimization 4: Bound the approximating tensor to {0, 1}:
    #    0 <= \sum_r VALUE_ijkr <= 1
    if value_bound:
        for i in range(N ** 2):
            for j in range(N ** 2):
                for k in range(N ** 2):
                    prob += 0 <= pulp.lpSum(
                        value[i][j][k][r] for r in range(R))
                    prob += pulp.lpSum(
                        value[i][j][k][r] for r in range(R)) <= 1
    # Optimization 5: Cyclic symmetry:
    #    T_ijk == T_kij == T_jki
    #    \sum_r VALUE_ijkr == \sum_r VALUE_kijr == \sum_r VALUE_jkir
    if cyclic_symmetry and not cyclic_invariant:
        for i in range(N ** 2):
            for j in range(N ** 2):
                for k in range(N ** 2):
                    if (i != k) or (j != i) or (k != j):
                        prob += pulp.lpSum(
                            value[i][j][k][r] for r in range(R)) == \
                            pulp.lpSum(
                                value[k][i][j][r] for r in range(R))
                    if (i != j) or (j != k) or (k != i):
                        prob += pulp.lpSum(
                            value[i][j][k][r] for r in range(R)) == \
                            pulp.lpSum(
                                value[j][k][i][r] for r in range(R))
                    assert T[i][j][k] == T[k][i][j]
                    assert T[i][j][k] == T[j][k][i]
    # Optimization 6: Transpose symmetry:
    #    AB = C <=> B'A' = C'
    #    T_ijk == T_jik x1 P x2 P x3 P
    #    \sum_r VALUE_ijkr == (\sum_r VALUE_jikr) x1 P x2 P x3 P
    if transpose_symmetry:
        _, p = np.where(vecperm(N=N))
        for i in range(N ** 2):
            for j in range(N ** 2):
                for k in range(N ** 2):
                    if (i == p[j]) and (j == p[i]) and (k == p[k]):
                        continue
                    prob += pulp.lpSum(
                        value[i][j][k][r] for r in range(R)) == \
                        pulp.lpSum(
                            value[p[j]][p[i]][p[k]][r] for r in range(R))
                    assert T[i][j][k] == T[p[j]][p[i]][p[k]]
    # Optimization 7: Matrix multiplication property:
    #    (XAY^{-1})(YBZ) = XCZ
    #    T == T x1 kron(X, Y^{-T}) x2 kron(Y, Z^T) x2 kron(Z^T, X)
    P = matrixperms(N=N)
    num_perms = P.shape[2] if permutation_symmetry else 0
    for x in range(num_perms):
        for y in range(num_perms):
            A = np.round(np.kron(P[:, :, x], np.linalg.inv(P[:, :, y]).T))
            _, A = np.where(A)
            for z in range(num_perms):
                # Limit the number of permutation matrix combinations to 8.
                # We could try more later on.
                if not (x == 0 or x == (num_perms - 1)):
                    continue
                if not (y == 0 or y == (num_perms - 1)):
                    continue
                if not (z == 0 or z == (num_perms - 1)):
                    continue
                B = np.round(np.kron(P[:, :, y], P[:, :, z].T))
                _, B = np.where(B)
                C = np.round(np.kron(P[:, :, z].T, P[:, :, x]))
                _, C = np.where(C)
                for i in range(N ** 2):
                    for j in range(N ** 2):
                        for k in range(N ** 2):
                            if (i == A[i]) and (j == B[j]) and (k == C[k]):
                                continue
                            prob += pulp.lpSum(
                                    value[i][j][k][r]
                                    for r in range(R)) == \
                                pulp.lpSum(
                                    value[A[i]][B[j]][C[k]][r]
                                    for r in range(R))
                            assert T[i, j, k] == T[A[i], B[j], C[k]]
    return prob


def write_instance(prob):
    """Write a MILP instance and an accompanying initial solution to disk."""
    params = prob.name.split('-')
    N, R = int(params[1][1:]), int(params[2][1:])
    S, T = None, None
    if len(params) > 3:
        S, T = int(params[3][1:]), int(params[4][1:])
    print('Writing instance N={N} R={R} S={S} T={T}...'.format(
        N=N, R=R, S=S, T=T))
    Ttrue = multiplication_tensor(N)
    if N == 2:
        U, V, W = solution_N2_R7()
    elif N == 3:
        U, V, W = solution_N3_R23()
    if R < U.shape[1]:
        if S is not None:
            assert S >= 0
        U, V, W = U[:, -R:], V[:, -R:], W[:, -R:]
    Ttrue = multiplication_tensor(N)
    That = expand_pd(U, V, W)
    start = {}
    for i in range(N ** 2):
        for j in range(N ** 2):
            for k in range(N ** 2):
                for r in range(R):
                    start['U_%i_%i' % (i, r)] = U[i, r]
                    start['V_%i_%i' % (j, r)] = V[j, r]
                    start['W_%i_%i' % (k, r)] = W[k, r]
                    if S is not None and r < S:
                        start['NEG_A_%i_%i' % (i, r)] = U[i, r] < 0
                        start['POS_A_%i_%i' % (i, r)] = U[i, r] > 0
                    elif S is not None and r >= S:
                        t = r - S
                        letter, col = 'BCD'[t // T], t % T
                        start['NEG_%s_%i_%i' % (letter, i, col)] = U[i, r] < 0
                        start['POS_%s_%i_%i' % (letter, i, col)] = U[i, r] > 0
                    start['ABS_U_%i_%i' % (i, r)] = np.abs(U[i, r])
                    start['ABS_V_%i_%i' % (j, r)] = np.abs(V[j, r])
                    start['ABS_W_%i_%i' % (k, r)] = np.abs(W[k, r])
                    start['ABS_ALT_U_%i_%i' % (i, r)] = np.abs(U[i, r])
                    start['ABS_ALT_V_%i_%i' % (j, r)] = np.abs(V[j, r])
                    start['ABS_ALT_W_%i_%i' % (k, r)] = np.abs(W[k, r])
                    start['ABS_DIFF_U_%i_%i' % (i, r)] = 0
                    start['ABS_DIFF_V_%i_%i' % (j, r)] = 0
                    start['ABS_DIFF_W_%i_%i' % (k, r)] = 0
                    start['NEG_U_%i_%i' % (i, r)] = U[i, r] < 0
                    start['NEG_V_%i_%i' % (j, r)] = V[j, r] < 0
                    start['NEG_W_%i_%i' % (k, r)] = W[k, r] < 0
                    start['POS_U_%i_%i' % (i, r)] = U[i, r] > 0
                    start['POS_V_%i_%i' % (j, r)] = V[j, r] > 0
                    start['POS_W_%i_%i' % (k, r)] = W[k, r] > 0
                    start['VALUE_%i_%i_%i_%i' % (i, j, k, r)] = \
                        U[i, r] * V[j, r] * W[k, r]
                    start['RESIDUAL_%i_%i_%i' % (i, j, k)] = \
                        np.abs(Ttrue[i][j][k] - That[i][j][k])
    with open('instances/' + prob.name + '.lp.mst', 'w') as f:
        f.write('# MIP start\n')
        for v in prob.variables():
            f.write('%s %f\n' % (v.name, start[v.name]))
    prob.writeLP('instances/' + prob.name + '.lp')
    return


if __name__ == '__main__':
    # Set active optimizations.
    optimizations = {
        'factors_equal_nz': True,
        'factors_nz_lb': True,
        'factors_col_nz_lb': 1,
        'factors_row_nz_lb': 1,
        'value_bound': False,
        'cyclic_symmetry': False,
        'transpose_symmetry': False,
        'permutation_symmetry': False
    }
    # Write instances that exploit cyclical invariance.
    NRST = \
        [(2, S + 3 * 2, S, 2) for S in range(0, 1 + 1)] + \
        [(3, S + 3 * 6, S, 6) for S in range(3, 5 + 1)]
    for N, R, S, T in NRST:
        prob = construct_instance(N, R, S, T, **optimizations)
        write_instance(prob)
