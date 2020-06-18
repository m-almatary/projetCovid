from scipy.integrate import odeint
from scipy import optimize, integrate


class SEIRDModel():
    def __init__(self, N, beta_init=0.1, gamma_init=0.3, R0=0,I0=1,E0=0,D0=0):
        
        self.beta_0 = beta_init
        self.gamma_0 = gamma_init
        
        self.beta = None
        self.gamma = None
        self.delta = None
        self.alpha = None
        self.rho = None

        # S = suspects
        self.S0 = N - I0 - R0 - E0 - D0
        # E = Exposed
        self.E0 = E0
        # I = Infected
        self.I0 = I0
        # R = Recovered
        self.R0 = R0
        # D = Deasth
        self.D0 = D0
        
        # N = Population
        self.N = N

        
    """
    
    """
    def deriv(self, SEIRD, t, N, beta, gamma, alpha, delta, rho):
        S, E, I, R, D = SEIRD
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - delta * E
        dIdt = delta * E - (1 - alpha) * gamma * I - alpha * rho * I
        dRdt = (1 - alpha) * gamma * I
        dDdt = alpha * rho * I
        
        return dSdt, dEdt, dIdt, dRdt, dDdt

    """
    Ajustement des parametres d'alpha, beta, gamma, delta et rho pour l'objet SEIRD
    """
    def fit(self, X, y):
        y0 = self.S0, self.E0, self.I0, self.R0, self.D0

        def fit_odeint(x, betas, gammas, alphas, deltas, rhos):
            return odeint(self.deriv, y0, x, args=(self.N,betas, gammas, alphas, deltas, rhos))[:,2]

        popt, pcov = optimize.curve_fit(fit_odeint, X, y)
    
        self.beta = popt[0]
        self.gamma = popt[1]
        self.alpha = popt[2]
        self.delta = popt[3]
        self.rho = popt[4]
        return self

    """

    """
    def predict(self, t):
        # Initial conditions vector
        y0 = self.S0, self.E0, self.I0, self.R0, self.D0
        # Integrate the SIR equations over the time grid, t.
        ret = odeint(self.deriv, y0, t, args=(self.N, self.beta, self.gamma, self.alpha, self.delta, self.rho))
        S, E, I, R, D = ret.T
        return S, E, I, R, D