import numpy as np
import pandas as pd
from sklearn import datasets

class LogisticRegression:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.coef = None

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _gradient_descent(self, X, Y):
        num_epochs = 100

        # Running gradient descent for 1000 iterations
        for i in range(num_epochs):
            # Calculating the gradient of the likelihood function
            gradient = np.dot(X.T, (Y - self._sigmoid(np.dot(X, self.coef))))
            # Updating the coeffcients
            self.coef = self.coef + self.alpha * gradient

    def fit(self, X, Y):
        """
        Fit the model with X and Y using gradient descent.
        X dimensions - n x p
        Y dimensions - n x 1
        """
        if X.shape[0] != Y.shape[0]:
            raise Exception("Number of observations in X and Y do not match.")

        # Adding new columns for bias term
        new_X = np.ones((X.shape[0], X.shape[1] + 1))
        new_X[:,:-1] = X

        # Creating array of coefficients
        self.coef = np.random.rand(new_X.shape[1], 1)
        # Running gradient descent to fit the model
        self._gradient_descent(new_X, Y)

    def predict(self, X):
        if X.shape[1] != self.coef.shape[0] - 1:
            raise Exception("Number of predictors in X does not match number of coefficients in the model.")
        # Generating probabilities
        pred =  self._sigmoid(np.dot(X, self.coef[:-1, :]) + self.coef[-1, :])

        # Returning classification where decision rule is 1 - p >= 0.5, else 1
        return np.where(pred >= 0.5, 1, 0)


def main():
    model = LogisticRegression()
    iris = datasets.load_iris()
    X = iris.data[0:99, :]
    Y = np.expand_dims(iris.target[0:99], axis=1)

    model.fit(X, Y)

    Y_pred = model.predict(X)

    print("Accuracy: ", str((1-np.sum(Y_pred - Y)) * 100) + '%')



if __name__ == '__main__':
    main()
