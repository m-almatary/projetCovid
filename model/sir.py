from scipy.integrate import odeint
from scipy import optimize, integrate


class SIRModel():
    def __init__(self, beta_init=0.1, gamma_init=0.3, R0=0,I0=1,N=10000):
        
        self.beta = beta_init
        self.gamma = gamma_init

        # S = suspects
        self.S0 = N - I0 - R0
        # I = Infected 
        self.I0 = I0
        # R = Recovered
        self.R0 = R0 
        
        # N = Population
        self.N = N

        
    """
    
    """
    def deriv(self, SIR, t, N, beta, gamma):
        S, I, R = SIR
        dSdt = -beta * S * I / N
        dRdt = gamma * I
        dIdt = - (dSdt + dRdt)
        return dSdt, dIdt, dRdt

    """
    Ajustement des parametres d'alpha et beta pour l'objet SIR
    """
    def fit(self, X, y):
        # Initial number of infected and recovered individuals, I0 and R0.
        y0 = self.S0, self.I0, self.R0
        # Everyone else, S0, is susceptible to infection initially.

        def fit_odeint(x, betas, gammas):
            return odeint(self.deriv, y0, x, args=(self.N,betas, gammas))[:,1]

        popt, pcov = optimize.curve_fit(fit_odeint, X, y)
    
        self.beta = popt[0]
        self.gamma = popt[1]
        return self

    """

    """
    def predict(self, t):
        # Initial conditions vector
        y0 = self.S0, self.I0, self.R0
        # Integrate the SIR equations over the time grid, t.
        ret = odeint(self.deriv, y0, t, args=(self.N, self.beta, self.gamma))
        S, I, R = ret.T
        return S, I, R