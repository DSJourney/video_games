import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd


class CA(object):
    """Simple corresondence analysis.

    Inputs
    ------
    ct : array_like
      Two-way contingency table. If `ct` is a pandas DataFrame object,
      the index and column values are used for plotting.

    Notes
    -----
    The implementation follows that presented in 'Correspondence
    Analysis in R, with Two- and Three-dimensional Graphics: The ca
    Package,' Journal of Statistical Software, May 2007, Volume 20,
    Issue 3.

    Credits
    -------
    The code in this file is from Taro Sato:
    https://okomestudio.net/biboroku/2014/05/brand-positioning-by-correspondence-analysis/
    """

    def __init__(self, ct):
        self.rows = ct.index.values if hasattr(ct, 'index') else None
        self.cols = ct.columns.values if hasattr(ct, 'columns') else None

        # contingency table
        N = np.matrix(ct, dtype=float)

        # correspondence matrix from contingency table
        P = N / N.sum()

        # row and column marginal totals of P as vectors
        r = P.sum(axis=1)
        c = P.sum(axis=0).T

        # diagonal matrices of row/column sums
        D_r_rsq = np.diag(1. / np.sqrt(r.A1))
        D_c_rsq = np.diag(1. / np.sqrt(c.A1))

        # the matrix of standarized residuals
        S = D_r_rsq * (P - r * c.T) * D_c_rsq

        # compute the SVD
        U, D_a, V = svd(S, full_matrices=False)
        D_a = np.asmatrix(np.diag(D_a))
        V = V.T

        # principal coordinates of rows
        F = D_r_rsq * U * D_a

        # principal coordinates of columns
        G = D_c_rsq * V * D_a

        # standard coordinates of rows
        X = D_r_rsq * U

        # standard coordinates of columns
        Y = D_c_rsq * V

        # the total variance of the data matrix
        inertia = sum([(P[i,j] - r[i,0] * c[j,0])**2 / (r[i,0] * c[j,0])
                       for i in range(N.shape[0])
                       for j in range(N.shape[1])])

        self.F = F.A
        self.G = G.A
        self.X = X.A
        self.Y = Y.A
        self.inertia = inertia
        self.eigenvals = np.diag(D_a)**2

    def plot(self, dictionary=False, sizes=False, df_size=None, color=None):
        """Plot the first and second dimensions."""
        coordinates = {}
        xmin, xmax = None, None
        ymin, ymax = None, None
        if self.rows is not None:
            for i, t in enumerate(self.rows):
                if t == 'Racing':
                    x, y = self.F[i,0], self.F[i,1]
                    plt.text(x, y-0.05, t, va='center', ha='center', color='r')
                    xmin = min(x, xmin if xmin else x)
                    xmax = max(x, xmax if xmax else x)
                    ymin = min(y, ymin if ymin else y)
                    ymax = max(y, ymax if ymax else y)
                    coordinates[t] = [x,y]

                else:
                    x, y = self.F[i,0], self.F[i,1]
                    plt.text(x, y, t, va='center', ha='center', color='r')
                    xmin = min(x, xmin if xmin else x)
                    xmax = max(x, xmax if xmax else x)
                    ymin = min(y, ymin if ymin else y)
                    ymax = max(y, ymax if ymax else y)
                    coordinates[t] = [x,y]
        else:
            plt.plot(self.F[:, 0], self.F[:, 1], 'ro')

        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)

        if self.cols is not None:
            for i, t in enumerate(self.cols):
                x, y = self.G[i,0], self.G[i,1]
                plt.text(x, y, t, va='center', ha='center', color='b', bbox=props)
                xmin = min(x, xmin if xmin else x)
                xmax = max(x, xmax if xmax else x)
                ymin = min(y, ymin if ymin else y)
                ymax = max(y, ymax if ymax else y)
        else:
            plt.plot(self.G[:, 0], self.G[:, 1], 'bs')

        if xmin and xmax:
            pad = (xmax - xmin) * 0.1
            plt.xlim(xmin - pad, xmax + pad)
        if ymin and ymax:
            pad = (ymax - ymin) * 0.1
            plt.ylim(ymin - pad, ymax + pad)

        plt.grid()
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.axhline(0, c='darkgray', linestyle='dashed', linewidth=1)
        plt.axvline(0, c='darkgray', linestyle='dashed', linewidth=1)

        if dictionary == True:
            print(coordinates)

        # plot PS sizes
        if sizes == True:
            plt.scatter(x="x", y="y_edited", s="values", c=color, data=df_size)

    def scree_diagram(self, perc=True, *args, **kwargs):
        """Plot the scree diagram."""
        eigenvals = self.eigenvals
        xs = np.arange(1, eigenvals.size + 1, 1)
        ys = 100. * eigenvals / eigenvals.sum() if perc else eigenvals
        plt.plot(xs, ys, *args, **kwargs)
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalue' + (' [%]' if perc else ''))



def _test(dataframe):
    import pandas as pd

    df = dataframe

    #print(df.describe())
    #print(df.head())

    ca = CA(df)

    plt.figure(100)
    ca.plot()
    plt.savefig('img/15.Correspondence_Analysis.png', dpi=200, transparent=True)

    plt.figure(101)
    ca.scree_diagram()
    plt.savefig('img/15.Scree_plot.png', dpi=200)

    plt.show()


if __name__ == '__main__':
    _test()
