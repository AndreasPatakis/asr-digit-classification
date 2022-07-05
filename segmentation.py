import numpy as np


class BackgroundForegroundClassifier:
    '''Handles sentence segmentation.'''

    def segment(self, sentence):
        '''Extracts digit audio signals from `sentence`.'''
        # TODO Segment the sentence
        return []

    #V=Visible Data, A=Transition Probabilities, B=Emission Probabilities, initial_distribution=initial_probabilities
    def forward(self, V, A, B, initial_distribution):
        #a(t, j)  is the probability that the machine will be at time step t at hidden state sj to derive the probability of the next time step
        a = np.zeros((V.shape[0], A.shape[0]))
        a[0, :] = initial_distribution * B[:, V[0]]

        # Matrix Computation Steps
        for t in range(1, V.shape[0]):
            for j in range(A.shape[0]):
                a[t, j] = a[t - 1].dot(A[:, j]) * B[j, V[t]]

        return a

    #The and will generate the remaining part of the sequence
    #V=Visible Data, A=Transition Probabilities, B=Emission Probabilities
    def backward(self, V, A, B):
        #b(t, i) is the probability that the machine will be at time step t in hidden state si and will generate the remaining part of the sequence of the visible symbol 
        b = np.zeros((V.shape[0], A.shape[0]))

        # setting b(T) = 1
        b[V.shape[0] - 1] = np.ones((A.shape[0]))

        # Loop in backward way from T-1 
        # Due to python indexing the actual loop will be T-2 to 0
        for t in range(V.shape[0] - 2, -1, -1):
            for i in range(A.shape[0]):
                b[t, i] = (b[t + 1] * B[:, V[t + 1]]).dot(A[i, :])

        return b

    #HMM Training is to estimate for Aij and Bjk using the training data.
    def baum_welch(self, V, A, B, initial_distribution, n_iter):
        M = A.shape[0]
        T = len(V)

        for n in range(n_iter):
            a = self.forward(V, A, B, initial_distribution)
            b = self.backward(V, A, B)

            xi = np.zeros((M, M, T - 1))  #xi dimensions = M x M x T-1
            for t in range(T - 1):
                denominator = np.dot(np.dot(a[t, :].T, A) * B[:, V[t + 1]].T, b[t + 1, :])
                for i in range(M):
                    numerator = a[t, i] * A[i, :] * B[:, V[t + 1]].T * b[t + 1, :].T
                    xi[i, :, t] = numerator / denominator

            gamma = np.sum(xi, axis=1)
            A = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

            # Add additional T'th element in gamma
            gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

            K = B.shape[1]
            denominator = np.sum(gamma, axis=1)
            for l in range(K):
                B[:, l] = np.sum(gamma[:, V == l], axis=1)

            B = np.divide(B, denominator.reshape((-1, 1)))

        return {"A":A, "B":B}

    #Viterbi is to solve the decoding problem. We need to find the most probable hidden state in every iteration of t.
    def viterbi(self, V, A, B, initial_distribution):
        T = V.shape[0]
        M = A.shape[0]

        omega = np.zeros((T, M))
        omega[0, :] = np.log(initial_distribution * B[:, V[0]])

        prev = np.zeros((T - 1, M))

        for t in range(1, T):
            for j in range(M):
                # Same as Forward Probability
                probability = omega[t - 1] + np.log(A[:, j]) + np.log(B[j, V[t]])

                # This is our most probable state given previous state at time t (1)
                prev[t - 1, j] = np.argmax(probability)

                # This is the probability of the most probable state (2)
                omega[t, j] = np.max(probability)

        # Path Array
        S = np.zeros(T)

        # Find the most probable last hidden state
        last_state = np.argmax(omega[T - 1, :])

        S[0] = last_state

        backtrack_index = 1
        for i in range(T - 2, -1, -1):
            S[backtrack_index] = prev[i, int(last_state)]
            last_state = prev[i, int(last_state)]
            backtrack_index += 1

        # Flip the path array since we were backtracking
        S = np.flip(S, axis=0)
     
        # Convert numeric values to actual hidden states
        result = []
        """for s in S:
            if s == 0:
                result.append("A")
            else:
                result.append("B")
        """
        return result
