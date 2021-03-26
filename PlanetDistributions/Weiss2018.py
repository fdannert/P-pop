"""
# =============================================================================
# P-POP
# A Monte-Carlo tool to simulate exoplanet populations
# =============================================================================
"""


# =============================================================================
# IMPORTS
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# =============================================================================
# WEISS2018
# =============================================================================

class PlanetDistribution():
    """
    https://ui.adsabs.harvard.edu/abs/2018AJ....155...48W/abstract
    """
    
    def __init__(self,
                 Scenario):
        """
        Parameters
        ----------
        Scenario: 'baseline', 'pessimistic', 'optimistic'
            Scenario for planet occurrence rates.
        """
        
        # Print.
        print('--> Initializing Weiss2018 planet distribution')
        
        # Model parameters
        self.returns = ['Rp', 'Porb']
        if (Scenario == 'baseline'):
            self.F0 = [2.42, 0.25]
            self.Gamma = [0.38, 0.73]
            self.alpha = [-0.19, -1.18]
            self.beta = [0.26, 0.59]
        elif (Scenario == 'pessimistic'):
            self.F0 = [1.14, 0.14]
            self.Gamma = [0.138, 0.72]
            self.alpha = [0.277, -1.56]
            self.beta = [0.204, 0.51]
        elif (Scenario == 'optimistic'):
            self.F0 = [5.60, 0.46]
            self.Gamma = [1.06, 0.78]
            self.alpha = [-0.68, -0.82]
            self.beta = [0.32, 0.67]
        else:
            print('--> WARNING: '+str(Scenario)+' is an unknown scenario')
            Scenario = 'baseline'
            self.F0 = [2.42, 0.25]
            self.Gamma = [0.38, 0.73]
            self.alpha = [-0.19, -1.18]
            self.beta = [0.26, 0.59]
        print('--> Using scenario '+str(Scenario))
        self.Rbrk = [0., 3.4, np.inf] # Rearth
        self.ytod = 365.24
        self.Rp_lims = [0.5, 16.] # Rearth
        self.Porb_lims = [0.5, 500.] # d
        self.CR0 = 1./(self.Rbrk[1]**self.alpha[0]/self.alpha[0]-self.Rp_lims[0]**self.alpha[0]/self.alpha[0])
        self.CP0 = 1./((self.Porb_lims[1]/self.ytod)**self.beta[0]/self.beta[0]-(self.Porb_lims[0]/self.ytod)**self.beta[0]/self.beta[0])
        self.CR1 = 1./(self.Rp_lims[1]**self.alpha[1]/self.alpha[1]-self.Rbrk[1]**self.alpha[1]/self.alpha[1])
        self.CP1 = 1./((self.Porb_lims[1]/self.ytod)**self.beta[1]/self.beta[1]-(self.Porb_lims[0]/self.ytod)**self.beta[1]/self.beta[1])
        
        # Max number of re-draws if correlation is too extreme.
        self.MaxTrials = np.inf
        
        pass
    
    def iCDF_R0(self,
                x):
        """
        Parameters
        ----------
        x: float, 0 <= x <= 1
            Uniformly distributed random number.
        
        Returns
        -------
        Rp: float
            Planet radius (Rearth) given Rp < Rbrk distributed according to
            SAG13 planet distribution.
        """
        
        return (self.alpha[0]*x/self.CR0+self.Rp_lims[0]**self.alpha[0])**(1./self.alpha[0])
    
    def iCDF_P0(self,
                x):
        """
        Parameters
        ----------
        x: float, 0 <= x <= 1
            Uniformly distributed random number.
        
        Returns
        -------
        Porb: float
            Planet orbital period (d) given Rp < Rbrk distributed according to
            SAG13 planet distribution.
        """
        
        return (self.beta[0]*x/self.CP0+(self.Porb_lims[0]/self.ytod)**self.beta[0])**(1./self.beta[0])
    
    def iCDF_R1(self,
                x):
        """
        Parameters
        ----------
        x: float, 0 <= x <= 1
            Uniformly distributed random number.
        
        Returns
        -------
        Rp: float
            Planet radius (Rearth) given Rp > Rbrk distributed according to
            SAG13 planet distribution.
        """
        
        return (self.alpha[1]*x/self.CR1+self.Rbrk[1]**self.alpha[1])**(1./self.alpha[1])
    
    def iCDF_P1(self,
                x):
        """
        Parameters
        ----------
        x: float, 0 <= x <= 1
            Uniformly distributed random number.
        
        Returns
        -------
        Porb: float
            Planet orbital period (d) given Rp > Rbrk distributed according to
            SAG13 planet distribution.
        """
        
        return (self.beta[1]*x/self.CP1+(self.Porb_lims[0]/self.ytod)**self.beta[1])**(1./self.beta[1])
    
    def draw(self,
             Rp_range=[0.5, 16.], # Rearth
             Porb_range=[0.5, 500.], # d
             Nplanets=None,
             Scale=1.,
             Star=None):
        """
        Parameters
        ----------
        Rp_range: list
            Requested planet radius range (Rearth).
        Porb_range: list
            Requested planet orbital period range (d).
        Nplanets: None, int
            Number of planets to be drawn.
        Scale: float
            Scaling factor for the planet occurrence rates.
        Star: instance
            Instance of class Star.
        
        Returns
        -------
        Rp: array
            Radius (Rearth) of drawn planets.
        Porb: array
            Orbital period (d) of drawn planets.
        """
        
        # Draw planet radius and planet orbital period according to the SAG13
        # planet distribution.
        Rp, Porb = self.drawSAG13(Rp_range,
                                  Porb_range,
                                  Nplanets,
                                  Scale=Scale) # Rearth, d
        
        # The drawn system is a multi-planet system.
        Nplanets = len(Rp)
        if (Nplanets > 1):
            
            # Sort planets by orbital period.
            ww = np.argsort(Porb)
            Rp = Rp[ww] # Rearth
            Porb = Porb[ww] # d
            
            # Draw planet radius and planet orbital period according to the
            # correlations from Weiss et al. 2018 for multi-planet systems.
            Rp = self.RadiusCorrSkew(Rp) # Rearth
            Porb = self.PeriodCorrSkew(Porb) # d
        
        return Rp, Porb
    
    def drawSAG13(self,
                  Rp_range=[0.5, 16.], # Rearth
                  Porb_range=[0.5, 500.], # d
                  Nplanets=None,
                  Scale=1.):
        """
        Parameters
        ----------
        Rp_range: list
            Requested planet radius range (Rearth).
        Porb_range: list
            Requested planet orbital period range (d).
        Nplanets: None, int
            Number of planets to be drawn.
        Scale: float
            Scaling factor for the planet occurrence rates.
        
        Returns
        -------
        Rp: array
            Radius (Rearth) of drawn planets.
        Porb: array
            Orbital period (d) of drawn planets.
        """
        
        Rp = [] # Rearth
        Porb = [] # d
        
        # Apply scaling for the planet occurrence rates.
        tempF0 = np.array(self.F0).copy()*Scale
        
        # If the number of planets is not given, draw it from a Poisson
        # distribution. Note that the final number of drawn planets might be
        # smaller than the drawn number because of clipping to the requested Rp
        # and Porb range.
        if (Nplanets is None):
            Nplanets = np.random.poisson(np.sum(tempF0))
            for i in range(Nplanets):
                
                # Randomly select whether Rp < Rbrk or Rp > Rbrk.
                temp = np.random.choice(len(tempF0), p=tempF0/np.sum(tempF0))
                if (temp == 0):
                    tempRp = self.iCDF_R0(np.random.rand()) # Rearth
                    tempPorb = self.iCDF_P0(np.random.rand())*self.ytod # d
                elif (temp == 1):
                    tempRp = self.iCDF_R1(np.random.rand()) # Rearth
                    tempPorb = self.iCDF_P1(np.random.rand())*self.ytod # d
                if (Rp_range[0] <= tempRp <= Rp_range[1] and Porb_range[0] <= tempPorb <= Porb_range[1]):
                    Rp += [tempRp] # Rearth
                    Porb += [tempPorb] # d
        
        # If the number of planets is given, draw exactly this number of
        # planets in the requested Rp and Porb range.
        else:
            while (len(Rp) < Nplanets):
                
                # Randomly select whether Rp < Rbrk or Rp > Rbrk.
                temp = np.random.choice(len(tempF0), p=tempF0/np.sum(tempF0))
                if (temp == 0):
                    tempRp = self.iCDF_R0(np.random.rand()) # Rearth
                    tempPorb = self.iCDF_P0(np.random.rand())*self.ytod # d
                elif (temp == 1):
                    tempRp = self.iCDF_R1(np.random.rand()) # Rearth
                    tempPorb = self.iCDF_P1(np.random.rand())*self.ytod # d
                if (Rp_range[0] <= tempRp <= Rp_range[1] and Porb_range[0] <= tempPorb <= Porb_range[1]):
                    Rp += [tempRp] # Rearth
                    Porb += [tempPorb] # d
        
        return np.array(Rp), np.array(Porb)
    
    def RadiusCorrSkew(self,
                       Rp, # Rearth
                       Ratio_lims=[0., np.inf],
                       Rp_lims=[0.1, 100.]): # Rearth
        """
        Parameters
        ----------
        Rp: array
            Radius (Rearth) of drawn planets sorted by orbital period.
        Ratio_lims: list
            Requested planet radius ratio range.
        Rp_lims: list
            Requested planet radius range (Rearth).
        
        Returns
        -------
        Rp: array
            Radius (Rearth) of drawn planets according to Weiss et al. 2018.
        """
        
        # Parameters from Emile.
        mu, sigma, skew = 0.69007816, 0.70029624, 3.52948576
        
        for i in range(len(Rp)-1):
            
            # Draw radius ratio inside limits.
            RadiusRatio = -1.
            Rp[i+1] = -1.
            Trials = 1
            while (Trials <= self.MaxTrials and (RadiusRatio <= Ratio_lims[0] or RadiusRatio >= Ratio_lims[1] or Rp[i+1] <= Rp_lims[0] or Rp[i+1] >= Rp_lims[1])):
                RadiusRatio = stats.skewnorm(skew, loc=mu, scale=sigma).rvs()
                Rp[i+1] = Rp[i]*RadiusRatio # Rearth
                Trials += 1
        
        return Rp
    
    def PeriodCorrSkew(self,
                       Porb, # d
                       Ratio_lims=[0., np.inf],
                       Porb_lims=[0., np.inf]): # d
        """
        Parameters
        ----------
        Porb: array
            Orbital period (d) of drawn planets sorted by orbital period.
        Ratio_lims: list
            Requested planet orbital period ratio range.
        Porb_lims: list
            Requested planet orbital period range (d).
        
        Returns
        -------
        Porb: array
            Orbital period (d) of drawn planets according to Weiss et al. 2018.
        """
        
        # Parameters from Emile.
        mu, sigma, skew = 1.28170899, 0.91055566, 9.35402502
        
        for i in range(len(Porb)-1):
            
            # Draw period ratio inside limits.
            PeriodRatio = -1.
            Porb[i+1] = -1.
            Trials = 1
            while (Trials <= self.MaxTrials and (PeriodRatio <= Ratio_lims[0] or PeriodRatio >= Ratio_lims[1] or Porb[i+1] <= Porb_lims[0] or Porb[i+1] >= Porb_lims[1])):
                PeriodRatio = stats.skewnorm(skew, loc=mu, scale=sigma).rvs()
                Porb[i+1] = Porb[i]*PeriodRatio # d
                Trials += 1
        
        return Porb
    
    def SummaryPlot(self,
                    Ntest=100000,
                    Rp_range=[0.5, 16.], # Rearth
                    Porb_range=[0.5, 500.], # d
                    FigDir=None,
                    block=True):
        """
        Parameters
        ----------
        Ntest: int
            Number of test draws for summary plot.
        Rp_range: list
            Requested planet radius range (Rearth).
        Porb_range: list
            Requested planet orbital period range (d).
        FigDir: str
            Directory to which summary plots are saved.
        block: bool
            If True, blocks plots when showing.
        """
        
        Ntest = Ntest//10
        Rp = []
        Porb = []
        for i in range(Ntest):
            tempRp, tempPorb = self.draw(Rp_range,
                                         Porb_range)
            Rp += [tempRp]
            Porb += [tempPorb]
        Rp = np.concatenate(Rp)
        Porb = np.concatenate(Porb)
        
        print('--> Weiss2018:\n%.2f planets/star in [%.1f, %.1f] Rearth and [%.1f, %.1f] d' % (len(Rp)/float(Ntest), Rp_range[0], Rp_range[1], Porb_range[0], Porb_range[1]))
        
        Weight = 1./len(Rp)
        f, ax = plt.subplots(1, 2)
        ax[0].hist(Rp, bins=np.logspace(np.log10(np.min(Rp)), np.log10(np.max(Rp)), 25), weights=np.ones_like(Rp)*Weight)
        ax[0].set_xscale('log')
        ax[0].grid(axis='y')
        ax[0].set_xlabel('Planet radius [$R_\oplus$]')
        ax[0].set_ylabel('Fraction')
        ax[1].hist(Porb, bins=np.logspace(np.log10(np.min(Porb)), np.log10(np.max(Porb)), 25), weights=np.ones_like(Porb)*Weight)
        ax[1].set_xscale('log')
        ax[1].grid(axis='y')
        ax[1].set_xlabel('Planet orbital period [d]')
        ax[1].set_ylabel('Fraction')
        plt.suptitle('Weiss2018')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if (FigDir is not None):
            plt.savefig(FigDir+'PlanetDistribution_Weiss2018.pdf')
        plt.show(block=block)
        plt.close()
        
        pass
