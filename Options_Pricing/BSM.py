#simple non-liquidatable option pricing calculation
import numpy as np
from scipy.stats import norm as norm

N = norm.cdf

# Black-Scholes-Merton formula for calls
class BSM_Mod:
    def BSM_CALL(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + (sigma**2)/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * N(d1) - K * np.exp(-r*T)* N(d2)
    # Black-Scholes-Merton formula for puts
    def BSM_PUT(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma* np.sqrt(T)
        return K*np.exp(-r*T)*N(-d2) - S*N(-d1)
