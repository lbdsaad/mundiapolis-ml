#!/usr/bin/env python3
"""Neural Network class"""
import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        Constructor
         Raises:
            TypeError: integer
            ValueError: Positive
        
        """
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """W1 getter"""
        return self.__W1

    @property
    def b1(self):
        """b1 getter"""
        return self.__b1

    @property
    def A1(self):
        """A1 getter"""
        return self.__A1

    @property
    def W2(self):
        """W2 getter"""
        return self.__W2

    @property
    def b2(self):
        """b2 getter"""
        return self.__b2

    @property
    def A2(self):
        """A2 getter"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
        Args:
            X: input data

        Returns:
            Activation functions (A1 & A2) - calculated with sigmoid function
        """
        A1_prev = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-A1_prev))

        A2_prev = np.matmul(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-A2_prev))

        return self.A1, self.A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y: contains the correct labels for the input data
            A: containing the activated output of the neuron for each example

        Returns:
            The cost
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                  np.multiply(1 - Y, np.log(1.0000001 - A)))
        return cost
