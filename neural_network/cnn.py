import numpy as np
from scipy.signal import convolve2d

class Conv2D:
    """2D Convolutional layer."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * \
                       np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        batch_size, _, h, w = x.shape
        if self.padding > 0:
            x = np.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        out_h = (h + 2*self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2*self.padding - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    output[b, oc] += convolve2d(x[b, ic], self.weights[oc, ic], mode='valid')[::self.stride, ::self.stride]
                output[b, oc] += self.bias[oc]
        return output

class MaxPool2D:
    """Max pooling layer."""
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        out_h, out_w = h // self.stride, w // self.stride
        output = np.zeros((batch_size, channels, out_h, out_w))
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        patch = x[b, c, i*self.stride:i*self.stride+self.pool_size,
                                  j*self.stride:j*self.stride+self.pool_size]
                        output[b, c, i, j] = np.max(patch)
        return output

class CNN:
    """Simple CNN architecture."""
    def __init__(self, num_classes=10):
        self.conv1 = Conv2D(1, 32, 3, padding=1)
        self.pool1 = MaxPool2D(2, 2)
        self.conv2 = Conv2D(32, 64, 3, padding=1)
        self.pool2 = MaxPool2D(2, 2)
        self.fc_weights = np.random.randn(64 * 7 * 7, 128) * 0.01
        self.fc_bias = np.zeros(128)
        self.out_weights = np.random.randn(128, num_classes) * 0.01
        self.out_bias = np.zeros(num_classes)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        x = self.relu(self.conv1.forward(x))
        x = self.pool1.forward(x)
        x = self.relu(self.conv2.forward(x))
        x = self.pool2.forward(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.relu(np.dot(x, self.fc_weights) + self.fc_bias)
        x = np.dot(x, self.out_weights) + self.out_bias
        return x
