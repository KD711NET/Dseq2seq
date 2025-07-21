import numpy as np


class HarmonicCurrentCalculator:
    def __init__(self, u, **kwargs):
        self.pred_len=24
        self.u = np.array(u, dtype=float)
        if self.u.ndim > 1 and self.u.shape[0] > 1:
            self.u = self.u.T
        self.ensInterval = 3 * 60*60
        self.lat = 21
        self.idTides = np.array(['st', 'k1', 'o1', 'p1', 'q1', 'm2', 's2', 'n2', 'k2', 'f '])
        self.tidesGiven = False
        self.steadyID = False
        self.calHarmonic = {}
        for key, value in kwargs.items():
            if key.lower().startswith('latitude'):
                self.lat = value
            elif key.lower().startswith('tides'):
                self.idTides = np.array(value, dtype=str)
                self.tidesGiven = True
            elif key.lower().startswith('ensinterval'):
                self.ensInterval = value

        self.time = self.ensInterval * (np.arange(len(self.u)) - 1)
        self.calculate_harmonic()
        #self.pre_harmonic()

    def calculate_harmonic(self):
        # Define omega
        omega = {
            'k1': 2 * np.pi / (23.934469 * 3600),
            'o1': 2 * np.pi / (25.81934 * 3600),
            'p1': 2 * np.pi / (24.06589 * 3600),
            'q1': 2 * np.pi / (26.86836 * 3600),
            'm2': 2 * np.pi / (12.420601 * 3600),
            's2': 2 * np.pi / (12.0 * 3600),
            'n2': 2 * np.pi / (12.658348 * 3600),
            'k2': 2 * np.pi / (11.967234 * 3600),
            'f': 2 * (2 * np.pi / (24 * 3600)) * np.sin(self.lat * np.pi / 180)
        }

        # Calculate components
        components = {
            'ac': np.ones(len(self.time))
        }
        for tide, w in omega.items():
            components['a' + tide] = np.cos(w * self.time)
            components['b' + tide] = np.sin(w * self.time)

        # Choose tides
        self.index = np.isin(['st', 'k1', 'o1', 'p1', 'q1', 'm2', 's2', 'n2', 'k2', 'f '], self.idTides)
        if 'st' in self.idTides:
            self.steadyID = True

        ID = np.repeat(self.index, 2)
        ID = ID[1:]

        OCM = np.vstack([components[key] for key in sorted(components.keys())]).T

        coef_matrix = OCM[:, ID]

        m, n = self.u.shape if self.u.ndim > 1 else (1, len(self.u))
        results = np.zeros((m, n))

        for i in range(m):
            st = self.u[i, :] if m > 1 else self.u
            X = np.linalg.lstsq(coef_matrix, st, rcond=None)[0]

            Xa, Xb = X[1::2], X[2::2]
            la = Xa
            sa = Xb
            Am = np.sqrt(la ** 2 + sa ** 2)
            Ph = np.arctan2(sa, la)

            s0 = X[0] if ID[0] else np.nan
            Res = np.zeros((coef_matrix.shape[1] // 2, n))
            for k in range(Res.shape[0]):
                Res[k, :] = X[2 * k + 1] * coef_matrix[:, 2 * k].T + X[2 * k + 2] * coef_matrix[:, 2 * k + 1].T
            results[i, :] = Res.sum(axis=0)
        self.s0 = s0
        self.la = la
        self.sa = sa
        self.Am = Am
        self.Ph = Ph
        self.results = results


    def get_results(self):
        self.pre_harmonic()
        return {
            'steadyU': self.s0,
            'longAxis': self.la,
            'shortAxis': self.sa,
            'amplitude': self.Am,
            'phase': self.Ph,
            'finalRes': self.results,
            'calHarmonic': self.calHarmonic
        }
    def pre_harmonic(self):
        # Define omega
        omega = {
            'k1': 2 * np.pi / (23.934469 * 3600),
            'o1': 2 * np.pi / (25.81934 * 3600),
            'p1': 2 * np.pi / (24.06589 * 3600),
            'q1': 2 * np.pi / (26.86836 * 3600),
            'm2': 2 * np.pi / (12.420601 * 3600),
            's2': 2 * np.pi / (12.0 * 3600),
            'n2': 2 * np.pi / (12.658348 * 3600),
            'k2': 2 * np.pi / (11.967234 * 3600),
            'f': 2 * (2 * np.pi / (24 * 3600)) * np.sin(self.lat * np.pi / 180)
        }
        self.time = self.ensInterval * np.arange(len((self.u).T))
        self.timepre = self.ensInterval * (np.arange((len((self.u).T) + self.pred_len)))
        # Calculate components
        components = {
            'ac': np.ones(len(self.time))
        }
        for tide, w in omega.items():
            components['a' + tide] = np.cos(w * self.time)
            components['b' + tide] = np.sin(w * self.time)

        # Choose tides
        self.index = np.isin(['st', 'k1', 'o1', 'p1', 'q1', 'm2', 's2', 'n2', 'k2', 'f '], self.idTides)
        if 'st' in self.idTides:
            self.steadyID = True

        ID = np.repeat(self.index, 2)
        ID = ID[1:]
        # ID = np.insert(ID, 0, self.index[0])


        # Calculate components

        components = {
            'ac': np.ones(len(self.timepre))
        }
        componentsraw = {
            'ac': np.ones(len(self.time))
        }
        for tide, w in omega.items():
            components['a' + tide] = np.cos(w * self.timepre)
            components['b' + tide] = np.sin(w * self.timepre)
        for tide, w in omega.items():
            componentsraw['a' + tide] = np.cos(w * self.time)
            componentsraw['b' + tide] = np.sin(w * self.time)

        OCMraw = np.vstack([componentsraw[key] for key in componentsraw.keys()]).T
        # Choose tides
        OCM = np.vstack([components[key] for key in components.keys()]).T
        coef_matrix= OCMraw[:, ID]
        coef_matrix1 = OCM[:, ID]
        m, n = self.u.shape if self.u.ndim > 1 else (1, len(self.timepre))
        results = np.zeros((m, n))
        for i in range(m):
            st = self.u[i, :] if m > 1 else self.u
            # X = np.linalg.lstsq(coef_matrix1, st, rcond=None)[0]
            coef_matrixt = coef_matrix.T
            L= np.dot(coef_matrixt,coef_matrix)

            L_1=np.linalg.inv(L)
            X1=np.dot(L_1,coef_matrixt)
            X=np.dot(X1,st.T)

            Xa, Xb = X[1::2], X[2::2]
            la = Xa
            sa = Xb
            Am = np.sqrt(la ** 2 + sa ** 2)
            Ph = np.arctan2(sa, la)

            s0 = X[0] if ID[0] else np.nan
            nk=(len(X) - 1) // 2
            n2 = n + self.pred_len
            Res = np.zeros((1, n2*nk))
            k = 0
            while k < nk:

                start_index = k  * n2
                end_index = (k+1) * n2
          
                Res[:, start_index:end_index] = np.dot(X[2 * k+1,:],coef_matrix1[:, 2 * k+1].T.reshape(1, -1)) + np.dot(X[
                    2 * k + 2] , coef_matrix1[:, 2 * k + 2].T.reshape(1, -1))
                k += 1

        if ID[0] != 0:
            mm = (coef_matrix.shape[1] - 1) // 2
        else:
            mm = coef_matrix.shape[1] // 2

        nn = n2
        res_out = np.zeros((m, mm, nn))

        for i in range(m):
            for j in range(mm):
                start_index = j * nn
                end_index = (j + 1) * nn
                res_out[i, j, :] = Res[i, start_index:end_index]

        res_out = np.squeeze(res_out)
        res=res_out
        self.s0 = s0
        self.la = la
        self.sa = sa
        self.Am = Am
        self.Ph = Ph

        calHarmonic = {}


        calHarmonic['diurU'] = np.sum(res[0:4, :], axis=0)
        calHarmonic['semiU'] = np.sum(res[4:8, :], axis=0)


        calHarmonic['K1U'] = res[0, :]
        calHarmonic['O1U'] = res[1, :]
        calHarmonic['P1U'] = res[2, :]
        calHarmonic['Q1U'] = res[3, :]
        calHarmonic['M2U'] = res[4, :]
        calHarmonic['S2U'] = res[5, :]
        calHarmonic['N2U'] = res[6, :]
        calHarmonic['K2U'] = res[7, :]
        calHarmonic['inertialU'] = res[8, :]

        self.calHarmonic=calHarmonic


