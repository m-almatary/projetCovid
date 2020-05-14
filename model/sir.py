from scipy.integrate import odeint
from scipy import optimize, integrate

class SIRModel():
    """
    Blabla sur SIR model
    
    TODO : equations

    """

    def __init__(self, SIR, t, N, beta_init=0.1, gamma_init=0.3):
        self.SIR = SIR
        self.t = [i for i in range(0,t)]
        
        self.beta_0 = beta_init
        self.gamma_0 = gamma_init
        self.beta_ = None
        self.gamma_ = None
        
        self.sus = -self.beta_0 * self.SIR[0] * self.SIR[1] / N
        self.rec = self.gamma_0 * self.SIR[1]
        self.inf = -(self.sus+self.rec)

    def fit(self, beta, gamma):
        return integrate.odeint(self, self.SIR, self.t, args=(beta, gamma))[:,1]

    def predict(self):
        popt, pcov = optimize.curve_fit(self.fit, self.t, self.SIR)
        return popt, pcov

if __name__ == "__main__":
    N = 64000000
    sir = SIRModel((N,1,0),50,N)

    print(sir.predict())