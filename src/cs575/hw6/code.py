import numpy as np
import cs575.hw6.nanohmm.nanohmm as nanohmm

A = [[0.66, 0.34],
     [1, 0]]
B = [[0.5, 0.25, 0.25],
     [0.1, 0.1, 0.8]]
pi = [0.8, 0.2]


def forward_slow(observed_sequence, A_input, B_input, pi_input):
    T_len = len(observed_sequence)
    N_len = len(A_input)
    alpha = []
    for index1 in range(len(pi_input)):
        for a_index in range(len(A_input[0])):
            for a_index_2 in range(0, len(A_input[0])):
                output = pi_input[index1] * A_input[index1][a_index] * A_input[index1][a_index_2]
                for observed in observed_sequence:
                    output *= B_input[a_index][observed]
                alpha.append(output)
    return alpha


# not working
def forward_slow2(observed_sequence, A_input, B_input, pi_input):
    N = len(A_input)
    T = len(observed_sequence)
    return forward_slow_recursive(observed_sequence, A_input, B_input, pi_input, T)


# not working
def forward_slow_recursive(observed_sequence, A_input, B_input, pi_input, depth):
    if depth is 0:
        return
    return forward_slow_recursive(observed_sequence, A_input, B_input, pi_input, depth - 1)


def forward_fast(observed_sequence, A_input, B_input, pi_input):
    N = len(A_input)
    T = len(observed_sequence)
    alpha = np.zeros((T, N))
    for i in range(N):
        alpha[0][i] = pi_input[i] * B_input[i][observed_sequence[0]]
    for t in range(1, T):
        for j in range(0, N):
            sum = 0
            for i in range(0, N):
                sum += alpha[t - 1][i] * A_input[i][j]
            alpha[t][j] = sum * B_input[j][observed_sequence[t]]
    return alpha


def backward(observed_sequence, A_input, B_input, pi_input):
    N = len(A_input[0])
    T = len(observed_sequence)
    beta = np.ones((T, N))
    for t in reversed(range(0, T - 1)):
        for j in range(N):
            sum = 0
            for i in range(N):
                sum += A_input[i][j] * B_input[i][observed_sequence[t + 1]] * beta[t + 1][i]
            beta[t][j] = sum
    sum = 0
    for i in range(N):
        beta[-1][i] = pi_input[i] * B_input[i][observed_sequence[0]] * beta[1][i]
    return beta


def likelihood_verify(O_list, A, B, pi):
    final = np.zeros((len(O_list), len(O_list[0])))
    for i in range(len(O_list)):
        for t in range(len(O_list[0])):
            fwd = forward_fast(O_list[i], A, B, pi)
            bkd = backward(O_list[i], A, B, pi)
            out = np.dot(fwd[i, :], bkd[i, :])
            final[i][t] = out
    return final


def match_sequences(O_list, A_list, B_list, pi_list):
    final = np.zeros((len(O_list), len(A_list[0])))
    for i in range(len(O_list)):
        for t in range(len(A_list[0])):
            L = np.sum(forward_fast(O_list[i], A_list[t], B_list[t], pi_list[t])[-1])
            final[t][i] = L
    return final


def nano_match_sequences(O_list, A_list, B_list, pi_list):
    final = np.zeros((len(O_list), len(A_list[0])))
    for i in range(len(O_list)):
        for t in range(len(A_list[0])):
            lambda_ = nanohmm.hmm_t(A_list[t], B_list[t], pi_list[t])
            f = nanohmm.forward_t(lambda_)
            L = nanohmm.forward(f, O_list[i])
            final[t][i] = L
    return final


O1 = (4, 2, 5, 1, 5, 1, 5, 3, 2, 3, 2, 0, 1, 0, 0, 4, 4, 3, 0, 1)
O2 = (3, 2, 3, 3, 5, 5, 5, 5, 1, 0, 1, 4, 2, 4, 3, 0, 5, 3, 1, 0)
O3 = (4, 3, 0, 3, 4, 0, 1, 0, 2, 0, 5, 3, 2, 0, 0, 5, 5, 3, 5, 4)
O4 = (3, 4, 2, 0, 5, 4, 4, 3, 1, 5, 3, 3, 2, 3, 0, 4, 2, 5, 2, 4)
O5 = (2, 0, 5, 4, 4, 2, 0, 5, 5, 4, 4, 2, 0, 5, 4, 4, 5, 5, 5, 5)


def train(O_list, states, reps):
    for O in O_list:
        A = np.random.rand(states, states).tolist()
        B = np.random.rand(states, 6).tolist()
        pi = np.random.rand(1, states)[0].tolist()

        lambda_ = nanohmm.hmm_t(A, B, pi)
        bw = nanohmm.baumwelch_t(lambda_)
        LL, lambda_ = nanohmm.baumwelch(bw, O, 100)
        LL_max = LL
        for i in range(reps):
            A = np.random.rand(4, 4).tolist()
            B = np.random.rand(4, 6).tolist()
            pi = np.random.rand(1, 4)[0].tolist()

            lambda_ = nanohmm.hmm_t(A, B, pi)
            bw = nanohmm.baumwelch_t(lambda_)

            LL, lambda_ = nanohmm.baumwelch(bw, O2, 100)
            #print("LL =", LL)
            if LL > LL_max:
                LL_max = LL
                A_out = lambda_.A
                B_out = lambda_.B
                pi_out = lambda_.pi
        print("Final trained")
        print("A = ", A_out)
        print("B = ", B_out)
        print("pi = ", pi_out)
        print()


O_list = [O1, O2, O3, O4, O5]
train(O_list, 4, 100)

"""A = [[0.6, 0.4],
     [1, 0]]
B = [[0.7, 0.3, 0],
     [0.1, 0.1, 0.8]]
pi = [0.7, 0.3]
O1 = [1, 0, 0, 0, 1, 0, 1]
O2 = [0, 0, 0, 1, 1, 2, 0]
O3 = [1, 1, 0, 1, 0, 1, 2]
O4 = [0, 1, 0, 2, 0, 1, 0]
O5 = [2, 2, 0, 1, 1, 0, 1]
O_list = [O1, O2, O3, O4, O5]
print(likelihood_verify(O_list, A, B, pi))"""

"""O1 = (1, 0, 0, 0, 1, 0, 1)
O2 = (0, 0, 0, 1, 1, 2, 0)
O3 = (1, 1, 0, 1, 0, 1, 2)
O4 = (0, 1, 0, 2, 0, 1, 0)
O5 = (2, 2, 0, 1, 1, 0, 1)

# HMM 1:
A1 = [[1.0, 0.0], [0.5, 0.5]]
B1 = [[0.4, 0.6, 0.0], [0.0, 0.0, 1.0]]
pi1 = [0.0, 1.0]

# HMM 2:
A2 = [[0.25, 0.75], [1.0, 0.0]]
B2 = [[0, 1.0, 0], [0.66, 0.0, 0.34]]
pi2 = [1.0, 0.0]

# HMM 3:
A3 = [[0.0, 1.0], [1.0, 0.0]]
B3 = [[1.0, 0.0, 0.0], [0.0, 0.66, 0.34]]
pi3 = [1.0, 0.0]

# HMM 4:
A4 = [[1, 0], [0.44, 0.56]]
B4 = [[0.36, 0.42, 0.22], [1.0, 0, 0]]
pi4 = [0, 1.0]

# HMM 5:
A5 = [[0.0, 1.0], [1.0, 0.0]]
B5 = [[0.25, 0.75, 0.0], [1.0, 0.0, 0.0]]
pi5 = [1.0, 0.0]

A_list = [A1, A2, A3, A4, A5]
B_list = [B1, B2, B3, B4, B5]
pi_list = [pi1, pi2, pi3, pi4, pi5]
O_list = [O1, O2, O3, O4, O5]

print(match_sequences(O_list, A_list, B_list, pi_list))

O1 = (4, 2, 5, 1, 5, 1, 5, 3, 2, 3, 2, 0, 1, 0, 0, 4, 4, 3, 0, 1)
O2 = (3, 2, 3, 3, 5, 5, 5, 5, 1, 0, 1, 4, 2, 4, 3, 0, 5, 3, 1, 0)
O3 = (4, 3, 0, 3, 4, 0, 1, 0, 2, 0, 5, 3, 2, 0, 0, 5, 5, 3, 5, 4)
O4 = (3, 4, 2, 0, 5, 4, 4, 3, 1, 5, 3, 3, 2, 3, 0, 4, 2, 5, 2, 4)
O5 = (2, 0, 5, 4, 4, 2, 0, 5, 5, 4, 4, 2, 0, 5, 4, 4, 5, 5, 5, 5)

# HMM 1:
A1 = [[0.33, 0, 0, 0.67, 0],
     [0.67, 0, 0.33, 0, 0],
     [0, 1.0, 0.0, 0, 0],
     [0, 0, 0, 0.25, 0.75],
     [0.0, 0.0, 0.6, 0, 0.4]]
B1 = [[0.67, 0, 0, 0, 0, 0.33],
     [0.0, 1.0, 0, 0, 0, 0],
     [0.5, 0, 0, 0, 0, 0.5],
     [0, 0, 0, 0.25, 0.75, 0],
     [0, 0.0, 0.6, 0.4, 0, 0.0]]
pi1 = [0.0, 0.0, 0.0, 1.0, 0.0]

# HMM 2:
A2 = [[0.0, 0.0, 1.0, 0, 0.0],
     [0.0, 0, 0.0, 0.0, 1.0],
     [0.38, 0.0, 0.23, 0.38, 0.0],
     [0.0, 0.31, 0.0, 0.69, 0],
     [0.0, 0.75, 0.0, 0.25, 0.0]]
B2 = [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
     [0.0, 0.6, 0.2, 0.2, 0.0, 0.0],
     [0.0, 0.0, 0, 1.0, 0.0, 0],
     [0, 0.0, 0, 0.22, 0.0, 0.78],
     [0.6, 0.0, 0.0, 0.0, 0.4, 0.0]]
pi2 = [0.0, 0.0, 1.0, 0.0, 0.0]

# HMM 3:
A3 = [[0, 0.0, 0.32, 0.18, 0.5],
     [0.0, 0.0, 0.0, 1.0, 0.0],
     [0, 0.0, 0, 0.0, 1.0],
     [0, 0.64, 0, 0.0, 0.36],
     [1.0, 0.0, 0, 0, 0]]
B3 = [[0.0, 0.17, 0.33, 0.0, 0.0, 0.5],
     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
     [0.47, 0.0, 0.0, 0.0, 0.0, 0.53],
     [0.27, 0.0, 0.0, 0.0, 0.73, 0.0],
     [0.66, 0.0, 0.0, 0.33, 0.0, 0.0]]
pi3 = [0.0, 0.0, 0.0, 1.0, 0.0]

# HMM 4:
A4 = [[0.0, 0.0, 1.0, 0, 0.0],
     [0.0, 0, 0.62, 0, 0.38],
     [0.0, 0.5, 0.0, 0.5, 0.0],
     [0.0, 0.23, 0.0, 0.0, 0.77],
     [0.0, 0, 0, 1.0, 0]]
B4 = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.62, 0, 0.38, 0.0],
     [0, 0.0, 0.0, 0.0, 1, 0],
     [0, 0.0, 0, 0.41, 0.18, 0.41],
     [0.31, 0.16, 0.37, 0.16, 0, 0.0]]
pi4 = [1.0, 0.0, 0.0, 0.0, 0]

# HMM 5:
A5 = [[0.5, 0.33, 0, 0.17, 0.0],
     [0.0, 0.0, 0.0, 0.0, 1.0],
     [0.75, 0.0, 0.25, 0.0, 0.0],
     [0.0, 0.0, 0, 1.0, 0.0],
     [0.0, 0.0, 1.0, 0.0, 0.0]]
B5 = [[0.0, 0.0, 0.0, 0.0, 1.0, 0],
     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0, 1.0],
     [0.0, 0.0, 0.0, 0.0, 0, 1.0],
     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
pi5 = [0.0, 1.0, 0.0, 0.0, 0.0]

A_list = [A1, A2, A3, A4, A5]
B_list = [B1, B2, B3, B4, B5]
pi_list = [pi1, pi2, pi3, pi4, pi5]
O_list = [O1, O2, O3, O4, O5]

print(nano_match_sequences(O_list, A_list, B_list, pi_list))"""

"""A = [[0.6, 0.4],
     [1, 0]]
B = [[0.7, 0.3, 0],
     [0.1, 0.1, 0.8]]
pi = [0.7, 0.3]
O1 = [1, 0, 0, 0, 1, 0, 1]
O2 = [0, 0, 0, 1, 1, 2, 0]
O3 = [1, 1, 0, 1, 0, 1, 2]
O4 = [0, 1, 0, 2, 0, 1, 0]
O5 = [2, 2, 0, 1, 1, 0, 1]

print("Likelihood for O1", np.sum(forward_fast(O1, A, B, pi)[-1]))
print("Likelihood for O2", np.sum(forward_fast(O2, A, B, pi)[-1]))
print("Likelihood for O3", np.sum(forward_fast(O3, A, B, pi)[-1]))
print("Likelihood for O4", np.sum(forward_fast(O4, A, B, pi)[-1]))
print("Likelihood for O5", np.sum(forward_fast(O5, A, B, pi)[-1]))

A = [[0.8, 0.1, 0.1],
     [0.4, 0.2, 0.4],
     [0, 0.3, 0.7]]
B = [[0.66, 0.34, 0],
     [0, 0, 1],
     [0.5, 0.4, 0.1]]
pi = [0.6, 0, 0.4]

O = [0, 1, 0, 2, 0, 1, 0]
print("Slow")
print(forward_slow(O, A, B, pi))
# print("Likelihood", np.sum(forward_slow(O, A, B, pi)))

print("Fast")
print(forward_fast(O, A, B, pi))

A = [[0.8, 0.1, 0.1],
     [0.4, 0.2, 0.4],
     [0, 0.3, 0.7]]
B = [[0.66, 0.34, 0],
     [0, 0, 1],
     [0.5, 0.4, 0.1]]
pi = [0.6, 0, 0.4]
O = [0, 1, 0, 2, 0, 1, 0]
print("Backward")
print(backward(O, A, B, pi))

A = [[0.5, 0.5], [0.0, 1.0]]
B = [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]]
pi = [0.5, 0.5]
O = [0, 1, 0, 2]
print("compare")
print(forward_fast(O, A, B, pi))"""
