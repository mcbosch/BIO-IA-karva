import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D 

class SimpleLinearRegression:
    def __init__(self):
        self.coefficient_ = None
        self.intercept_ = None
        self.r2score_ = None
        
    def fit(self, X, y):
        n = len(X)
        X_b = np.c_[np.ones((n,1)), X]  
   
        self.coefficients_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = self.coefficients_[0]
        y_pred = X_b.dot(self.coefficients_)
        # R²
        self.r2score_ = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))
        self.y_pred_ = y_pred
        
    def predict(self, X):
        X_b = np.c_[np.ones((len(X),1)), X]
        return X_b.dot(self.coefficients_)

    def plot(self, X, y):
        self.fit(X, y)


        print(f"Simple LR Coefficients: {self.coefficients_}")
        print(f"R² Score: {self.r2score_:.2f}")

        plt.scatter(X, y, color='blue', label='Data')
        plt.plot(X, self.y_pred_, color='red', label='Regression Line')
        plt.title("Simple Linear Regression")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.show()


class MultipleLinearRegression:
    def __init__(self):
        self.coefficients_ = None 
        self.intercept_ = None     
        self.r2score_ = None    
        
    def fit(self, X, y):
        n = X.shape[0]
        # Concatenamos columna de 1s en X (intercept)
        X_b = np.c_[np.ones((n,1)), X]  
        self.coefficients_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = self.coefficients_[0]
        y_pred = X_b.dot(self.coefficients_)
        self.r2score_ = 1 - (np.sum((y - y_pred)**2)/np.sum((y - np.mean(y))**2))
        self.y_pred_ = y_pred
        
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0],1)), X] 
        return X_b.dot(self.coefficients_)

    def plot(self, X, y, cols=[0,1]):
        self.fit(X, y)
        print(f"Multiple LR Coefficients: {self.coefficients_}")
        print(f"R² Score: {self.r2score_:.2f}")

        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')

        x1 = X[:,cols[0]]
        x2 = X[:,cols[1]]
        ax.scatter(x1, x2, y, color='blue', label='Data')

        x1_surf, x2_surf = np.meshgrid(
            np.linspace(x1.min(), x1.max(), 10),
            np.linspace(x2.min(), x2.max(), 10)
        )
        # Build a feature matrix for every grid point. Features not being
        # plotted are held at their mean so the surface is a 2D slice.
        grid = np.tile(X.mean(axis=0), (x1_surf.size, 1))
        grid[:, cols[0]] = x1_surf.ravel()
        grid[:, cols[1]] = x2_surf.ravel()
        z_surf = self.predict(grid).reshape(x1_surf.shape)

        ax.plot_surface(x1_surf, x2_surf, z_surf,
                        color='red', 
                        alpha=0.5, 
                        rstride=1, 
                        cstride=1)

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('y')
        ax.set_title("Multiple Linear Regression with Regression Plane")
        ax.legend()
        plt.show()