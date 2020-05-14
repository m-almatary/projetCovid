import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
from scipy import optimize, integrate
import random


class SIRModel():
    def init(self, beta_init=0.1, gamma_init=0.3):
        self.beta_0 = beta_init
        self.gamma_0 = gammainit
        self.beta = None
        self.gamma_ = None


    def fit(self, X, y, N):
        infected = y
        newN = N

        # Initial number of infected and recovered individuals, I0 and R0.
        I0, R0 = infected[0], 0
        # Everyone else, S0, is susceptible to infection initially.
        S0 = newN - I0 - R0

        x = X

        def sir_model(y, x, betas, gammas):
            sus = -betas * y[0] * y[1] / newN
            rec = gammas * y[1]
            inf = -(sus + rec)
            return sus, inf, rec

        def fit_odeint(x, betas, gammas):
            return integrate.odeint(sir_model, (S0, I0, R0), x, args=(betas, gammas))[:,1]

        popt, pcov = optimize.curve_fit(fit_odeint, x, y)
        fitted = fit_odeint(x, *popt)
        self.beta = popt[0]
        self.gamma_ = popt[1]
        return self


    def predict(self, t, N, I0):
        # Initial number of infected and recovered individuals, I0 and R0.
        I0, R0 = I0, 0
        # Everyone else, S0, is susceptible to infection initially.
        S0 = N - I0 - R0
        # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
        beta, gamma = self.beta, self.gamma_

        # The SIR model differential equations.
        def deriv(y, t, N, beta, gamma):
            S, I, R = y
            dSdt = -beta * S * I / N
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I
            return dSdt, dIdt, dRdt

        # Initial conditions vector
        y0 = S0, I0, R0
        # Integrate the SIR equations over the time grid, t.
        ret = odeint(deriv, y0, t, args=(N, beta, gamma))
        S, I, R = ret.T
        return S, I, R


    def foo(self, a, b):
        pass
    
 
#exemple d'usage
#a = SIRModel()
#infected = list((data[data.sexe==0]).groupby(['jour'])["hosp"].sum())
#t = [i for i in range (0,len(infected))]
#a.fit(t,infected,1000000)
#print(a.beta, a.gamma_)
#a.predict(t)