import numpy as np

class SSA:
    """
        Singular Spectrum Analaysis (SSA) is a method that converts a timeseries into its fundementals.
        SSA given a timeseries T will decomse T into three components, trend, harmonics, and noise.
        The SSA method implemented in this package is pulled from the book
        Analysis of Time Series Structure SSA and Related Techniques
        By Nina Golyandina, Vladimir Nekrutkin, Anatoly A Zhigljavsky · 2001
    """

    def __decompose(self):
        """
            Performs the decomposition step by computing the trajectory matrix X of
            the timeseries F
        """
        self.X = np.matrix(np.zeros(shape=(self.L, self.K)))
        for tidx in range(self.K):
            self.X[:, tidx] = self.F[0, tidx:tidx+self.L].T

    def __svd(self):
        """
            Performs the SVD step where the SVD is computed and reformed into
            a list of uniary matricies such that the sum of the eigentriple
            grouping is equalt to the trajectory matrix
        """
        # Compute the SVD of X*X.T
        u, s, vh = np.linalg.svd(self.X*self.X.T)
        self._svd = (u, s, vh)

        # Grab the rank of s
        if False in s:
            d = np.argmin(s > 0)
        else:
            d = len(s)

        # Set the rank for to
        self.rank = d

        # Restructure the U, S, V^T into its eigentriple grouping
        # where X = [X_0, X_1, ...]
        xsvd = []
        for i in range(d):
            vi = (self.X.T*u[:, i])/np.sqrt(s[i])
            xsvd.append(np.matrix(np.sqrt(s[i]) * u[:,i] * vi.T))

        # Rewrite X in this form
        self.X = xsvd

    def __reconstruct(self):
        """
            Reconstruct every subset defined by the newly cunstrcted X 
        """
        self.Ys = np.matrix(np.zeros(shape=(self.rank, self.N)))
        for Yidx in range(len(self.X)):
            Y = self.X[Yidx]
            L, K = Y.shape
            Ls = min([L, K])
            Ks = max([L, K])
            N = L + K - 1
            reconstruction = []

            # This performs diagonal averaging of the trajectory matrix Y
            # leaving a reconstructed time series
            for k in range(N):
                if 0 <= k < Ls - 1:
                    total = 0
                    for m in range(1, k + 2):
                        if L < K:
                            total += Y[m-1, (k-m+2)-1]
                        else:
                            total += Y[k-(m-1)+2, m-1]
                    total *= (1.0 / (k + 1))
                    reconstruction.append(total)
                elif Ls - 1 <= k < Ks:
                    total = 0
                    for m in range(1, Ls+1):
                        if L < K:
                            total += Y[m-1, (k-m+2)-1]
                        else:
                            total += Y[k-(m-1)+2, m-1]
                    total *= (1.0 / (Ls))
                    reconstruction.append(total)
                elif Ks <= k < N:
                    total = 0
                    for m in range(k-Ks+2, N-Ks+1+1):
                        if L < K:
                            total += Y[m-1, (k-m+2)-1]
                        else:
                            total += Y[(k-m+2)-1, m-1]
                    total *= (1.0 / (N-k))
                    reconstruction.append(total)

            reconstruction = np.matrix(reconstruction)
            self.Ys[Yidx, :] = reconstruction

    def __compute_w(self):
        self.wcorr = np.matrix(np.zeros(shape=(self.Ys.shape[0], self.Ys.shape[0])))
        Ls = min([self.X[0].shape[0], self.X[0].shape[1]])
        Ks = min([self.X[0].shape[0], self.X[0].shape[1]])
        for i in range(self.wcorr.shape[0]):
            for j in range(i+1):
                # Compute the weighted inner product
                f1f2 = 0
                f1 = 0
                f2 = 0
                for k in range(0, self.N):
                    if 0 <= k <= Ls - 1:
                        f1f2 += (i+1)*self.Ys[i, k]*self.Ys[j, k]
                        f1 += (i+1)*self.Ys[i, k]*self.Ys[i, k]
                        f2 += (i+1)*self.Ys[j, k]*self.Ys[j, k]
                    elif Ls <= k <= Ks:
                        f1f2 += Ls*self.Ys[i, k]*self.Ys[j, k]
                        f1 += Ls*self.Ys[i, k]*self.Ys[i, k]
                        f2 += Ls*self.Ys[j, k]*self.Ys[j, k]
                    elif Ks <= k < self.N:
                        f1f2 += (self.N-i)*self.Ys[i, k]*self.Ys[j, k]
                        f1 += (self.N-i)*self.Ys[i, k]*self.Ys[i, k]
                        f2 += (self.N-i)*self.Ys[j, k]*self.Ys[j, k]
                self.wcorr[i, j] = np.abs(f1f2 / (np.sqrt(f1) * np.sqrt(f2)))
                self.wcorr[j, i] = self.wcorr[i, j]

    def __statgroup(self):
        """
            A simple technique that uses statistics to perform eigentriple
            grouping.
        """
        try:
            wcorr_group = np.matrix(np.zeros(shape=self.wcorr.shape))

            # Due to the nature of the wcorr matrix and the nature of the the singular
            # values. A statistical approach to the eigentriple grouping problem does
            # and excellent job at identifying primative groups. These groups can be
            # found by finding outleries in the wcorr column. The first step in the
            # grouping process is to identy the outliers of each column. These outliers
            # construct a directed graph which is represented in the wcorr_group
            # matrix.
            for colidx in range(self.wcorr.shape[0]):
                u = np.mean(self.wcorr[:, colidx])
                o2 = np.std(self.wcorr[:, colidx])
                outliers = self.wcorr[:, colidx] > (u + 2*o2)
                wcorr_group[:, colidx] = outliers

                # Only take consective outliers
                for rowidx in range(1, wcorr_group.shape[0] - 1):
                    if wcorr_group[rowidx-1,colidx] == wcorr_group[rowidx+1,colidx] == False \
                       and wcorr_group[rowidx,colidx] == True and rowidx != colidx:
                        wcorr_group[rowidx,colidx] = False

            # Now that the connected graph is found, we expect the resulting matrix to
            # be symetric along the main diagonal. We can take the intersection of the
            # lower and upper triangular matrix which will remove outliers that have
            # been included in the statistical filter.
            upper = np.triu(wcorr_group)
            lower = np.tril(wcorr_group).T
            np.where((upper == lower), upper, 0)
            wcorr_group = np.add(upper, np.tril(upper.T, k=-1))

            # After we construct a symmetric matrix we now can define a group as the true
            # elements in the column of a matrix. Below we construct the groups array where
            # each element contains a range described by [a, b)
            groups = []
            colidx = 0
            while colidx < self.wcorr.shape[0]:
                rg = np.where(wcorr_group[:, colidx] == True)[0]
                if len(rg) == 1:
                    groups.append(np.array([rg[0], rg[0] + 1]))
                else:
                    rg[-1] += 1
                    groups.append(rg)
                colidx += len(rg)

            # The final step is to take the union of groups that are next to each other.
            # If the groupings overlap because they share a _weak_ seperable component then
            # the groups can be merged together.
            groups_merged = [groups[0]]
            groupidx = 1
            while groupidx < len(groups):
                if groups_merged[-1][-1] > groups[groupidx][0]:
                    groups_merged[-1][-1] = groups[groupidx][-1]
                else:
                    groups_merged.append(groups[groupidx])
                groupidx += 1

            self.wcorr_group = wcorr_group
            self.auto_grouping = groups_merged
        except:
            print('warn: __statgroup failed, auto grouping not working')

    def __computelrf(self, r, a, i):
        """
            Helper function that solves the linear recurrance formula described in
            chapter 2 section 2. Analysis of Time Series Structure SSA and Related Techniques
        """
        if i < r.shape[0]:
            return r[i]
        else:
            total = 0
            for j in range(1, self.rank):
                total += np.multiply(a[a.shape[0]-j-1,0], self.__computelrf(r, a, i-j))
            return total

    def __init__(self, ts, window=None):
        """
            :param ts: The timeseries to perform the SSA decomposition on
            :param window: The window length to use when performing SSA.
            If window > len(ts) // 2, initalization will fail and an assertion
            error will be thrown. If window is not given len(ts) // 2 will be
            used as the window length. If window < 2, an assertion error will
            be thrown.
        """

        # Verify valid window
        if window is None:
            window = len(ts) // 2
        else:
            assert(window > 2)
            assert(window <= len(ts) // 2)

        # L is the window length
        self.L = window

        # N is the length of the timeseries
        self.N = len(ts)

        # F is the original timeseries
        self.F = np.matrix(ts)

        # The number of lagged vectors present in the trajectory matrix
        self.K = self.N - self.L + 1

        # Perform the decomposition step
        # Section 1.1.1, part 1 of Analysis of Time Series Structure SSA and Related Techniques
        self.__decompose()

        # Perform the svd step
        # Section 1.1.1, part 2 of Analysis of Time Series Structure SSA and Related Techniques
        self.__svd()

        # Perform the reconstruction step
        # Section 1.1.2, part 1 of Analysis of Time Series Structure SSA and Related Techniques
        self.__reconstruct()

        # Compute the w corrilation matrix
        self.__compute_w()

        # Perform eigentriple grouping based on statistical filter, and connected graph confirmation
        self.__statgroup()

    def forecast(self, i, j=None, forecast=None):
        """
            Given a grouping defined by [i, j), forecasts ahead 1 period, where the period
            is equal to the window length defined by the decomposition.

            :param i: First element in reconstruction grouping
            :param j: Last element in reconstruction grouping
            :returns: An array that contains the extension of the reconstruction and True/False
            if the given LRF is stable. See https://en.wikipedia.org/wiki/Recurrence_relation#Stability
            for more information.
        """
        # Construct the timeseries grouping
        reconstruction = self.get_reconstructed(i, j)
        forecasted = reconstruction.copy()

        # The columns of U span a LRF space which may or may not be stable.
        if j is not None:
            basis = self._svd[0][:, i:j]
        else:
            basis = self._svd[0][:, i]

        pi = basis.T[:, -1]
        v2 = np.sum(np.power(pi, 2.0))
        p_hat = basis[:-1, :]

        alpha = np.matrix(np.zeros(shape=p_hat[:, 0].shape))
        for i in range(basis.shape[1]):
            alpha[:, 0] += pi[i, 0]*p_hat[:, i]

        alpha *= (1.0 / (1.0 - v2))

        if forecast is None:
            forecast = self.L

        for i in range(0, forecast):
            gi = self.__computelrf(forecasted, alpha, reconstruction.shape[0]+i)
            forecasted = np.concatenate((forecasted, gi), axis=0)
        return forecasted

    def get_reconstructed(self, i, j=None):
        """
            Gets the reconstructed timeseries. If both i and j are specified reconstructed
            will sum all the timeseries between [i, j) and return the result.

            :param i: First reconstruction
            :param j: Ending reconstruction
        """

        if j is not None:
            assert(i < j)
            assert(0 <= i < self.Ys.shape[0])
            assert(i <  j <= self.Ys.shape[0])
            return np.sum(self.Ys[i:j+1,:], axis=0).T
        else:
            assert(0 <= i < self.Ys.shape[0])
            return self.Ys[i, :].T

    def get_singulars(self):
        """
            Gets the singular values in decending order from the svd step of the decomposition
        """
        return self._svd[1]

    def get_wcorr(self):
        """
            Gets the w-corrilation matrix of the reconstructed timeseries
        """
        return self.wcorr

    def get_wcorr_grouping(self):
        """
            Gets the grouping matrix of the w-corrilation matrix
        """
        return self.wcorr_group

    def get_grouping(self):
        """
            Returns a list of eigentriple groupings
        """
        return self.auto_grouping
