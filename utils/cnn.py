import numpy as np

class CNN:

    def __init__(self, num_kernels, pool_size, out_nodes, seed):
        self.kernels = self.Kernels(num_kernels)  # 8x8x1  to 6x6x10
        # TODO add pooling initialization
        # self.pool = self.Pool(pool_size)  # 6x6x10 to 3x3x10
        # TODO change sigmoid to relu
        self.sigmoid = self.ReLu(6 * 6 * num_kernels, out_nodes)  # 6x6x10 to 10
        np.random.seed(seed)

    class Kernels:
        # TODO possibly add bias forward and backprop to kernel outputs
        # The set of kernels and kernel functions

        def __init__(self, num_kernels):
            self.num_kernels = num_kernels
            self.size_kernel = 3
            self.kernels = np.random.randn(num_kernels, 3, 3) / (self.size_kernel ** 2)

        def forward(self, x):
            self.cur_x = x

            height, width = x.shape
            result = np.zeros((height - 2, width - 2, self.num_kernels))

            for i in range(height-2):
                for j in range(width-2):
                    block = x[i:(i+3), j:(j+3)]
                    result[i,j] = np.sum(block * self.kernels, axis=(1, 2))
            return result

        def back(self, delta_cnn, alpha):
            # TODO change size to recieve backprop from pool size
            height, width = self.cur_x.shape
            dk = np.zeros(self.kernels.shape)
            delta_cnn = delta_cnn.reshape(height-2, width-2,self.num_kernels)
            for i in range(height - 2):
                for j in range(width - 2):
                    block_x = self.cur_x[i:(i+3), j:(j+3)]
                    # block_delta = delta_cnn[i:(i+3), j:(j+3)]
                    for k in range(self.num_kernels):
                        dk[k] += delta_cnn[i,j,k] * block_x

            self.kernels += -alpha * dk


    class Pool:
        # use max pooling with size 2x2
        def __init__(self, pool_size):
            self.dim = pool_size  # size of the pool

        def forward(self, x):
            self.cur_input = x
            height, width, num_k = x.shape
            result = np.zeros((height // 2, width // 2, num_k))
            for i in range(height // 2):
                for j in range(width // 2):
                    block = x[(i*2):(i*2+2), (j*2):(j*2+2)]
                    result[i,j] = np.amax(block, axis=(0,1))
            return result

        def back(self, delta_pool):
            # TODO recieve from output size
            height, width, num_k = self.cur_input.shape
            delta_pool = delta_pool.reshape((height//2, width//2, num_k))
            delta_cnn = np.zeros((height, width, num_k))

            for i in range(height // 2):
                for j in range(width // 2):
                    block = self.cur_input[(i*2):(i*2+2),(j*2):(j*2+2)]
                    b_h, b_w, b_k = block.shape
                    maximum = np.amax(block, axis=(0,1))
                    for s in range(b_h):
                        for t in range(b_w):
                            for u in range(b_k):
                                if block[s, t, u] == maximum[u]:
                                    delta_cnn[i*2+s, j*2+t, u] = delta_pool[i,j,u]
            return delta_cnn


    class ReLu:
        # Use a fully connected layer using sigmoid activation

        def __init__(self, in_nodes, out_nodes):
            self.w = np.random.randn(out_nodes, in_nodes) / in_nodes
            self.b = np.zeros((out_nodes, 1))

        def f(self, z):
            return np.where(z > 0, z, 0)

        def d_f(self, z):
            test = np.where(z>0, 1.0, 0.0)
            return np.where(z>0, 1.0, 0.0)

        def loss(self, y, yhat):
            self.y_vector = np.zeros((len(yhat), 1))
            self.y_vector[y, 0] = 1.0
            loss = np.linalg.norm(self.y_vector - yhat)
            accuracy = 1 if np.argmax(yhat[:, 0]) == y else 0
            return loss, accuracy

        def forward(self, x):
            # TODO setup to receive pool size
            self.z_cnn = x.reshape(-1, 1)
            self.a_cnn = self.f(self.z_cnn)

            self.z_out = np.matmul(self.w, self.a_cnn) + self.b
            self.a_out = self.f(self.z_out)

            return self.a_out

        def back(self, alpha):

            w_grad = np.zeros(self.w.shape)
            b_grad = np.zeros(self.b.shape)

            # get delta for out layer
            delta_out = -(self.y_vector - self.a_out) * self.d_f(self.z_out)

            # get delta for hidden layer
            delta_pool = np.dot(self.w.T, delta_out) * self.d_f(self.z_cnn)

            w_grad += -alpha * np.matmul(delta_out, self.a_cnn.T)
            b_grad += -alpha * delta_out

            return delta_pool



