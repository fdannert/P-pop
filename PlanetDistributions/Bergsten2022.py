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

import scipy.stats as stats
from scipy.integrate import quad


# =============================================================================
# BERGSTEN2022
# =============================================================================

class rv_lessless(stats.rv_continuous):
    
    def __init__(self, **kwargs):
        
        self.Rp_lims = [1., 3.5] # Rearth
        self.Porb_lims = [2., 100.] # d
        
        super(rv_lessless, self).__init__(**kwargs)
        
        pass
    
    def t(self, Porb, Pcen, s):
        
        return 0.5 - 0.5 * np.tanh((np.log10(Porb) - np.log10(Pcen)) / np.log10(s))
    
    def G_less(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * chi1 + (1. - self.t(Porb, Pcen, s)) * chi2
    
    def G_more(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * (1. - chi1) + (1. - self.t(Porb, Pcen, s)) * (1. - chi2)
    
    def dNdP_lessless(self, Porb, Pbrk, beta1, Pcen, s, chi1, chi2, Rval):
        
        return (Porb / Pbrk)**beta1 * self.G_less(Porb, Pcen, s, chi1, chi2) * (self.Rp_lims[1] - Rval) / ((Rval - self.Rp_lims[0]) + self.G_less(Porb, Pcen, s, chi1, chi2) * (self.Rp_lims[1] - Rval) - self.G_less(Porb, Pcen, s, chi1, chi2) * (Rval - self.Rp_lims[0]))
    
    def _pdf(self, Porb, Pbrk, beta1, Pcen, s, chi1, chi2, Rval, norm):
        
        return self.dNdP_lessless(Porb, Pbrk, beta1, Pcen, s, chi1, chi2, Rval) / norm
    
    def _argcheck(self, *args):
        
        return 1

class rv_lessmore(stats.rv_continuous):
    
    def __init__(self, **kwargs):
        
        self.Rp_lims = [1., 3.5] # Rearth
        self.Porb_lims = [2., 100.] # d
        
        super(rv_lessmore, self).__init__(**kwargs)
        
        pass
    
    def t(self, Porb, Pcen, s):
        
        return 0.5 - 0.5 * np.tanh((np.log10(Porb) - np.log10(Pcen)) / np.log10(s))
    
    def G_less(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * chi1 + (1. - self.t(Porb, Pcen, s)) * chi2
    
    def G_more(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * (1. - chi1) + (1. - self.t(Porb, Pcen, s)) * (1. - chi2)
    
    def dNdP_lessmore(self, Porb, Pbrk, beta1, Pcen, s, chi1, chi2, Rval):
        
        return (Porb / Pbrk)**beta1 * self.G_more(Porb, Pcen, s, chi1, chi2) * (self.Rp_lims[1] - Rval) / ((Rval - self.Rp_lims[0]) + self.G_more(Porb, Pcen, s, chi1, chi2) * (self.Rp_lims[1] - Rval) - self.G_more(Porb, Pcen, s, chi1, chi2) * (Rval - self.Rp_lims[0]))
    
    def _pdf(self, Porb, Pbrk, beta1, Pcen, s, chi1, chi2, Rval, norm):
        
        return self.dNdP_lessmore(Porb, Pbrk, beta1, Pcen, s, chi1, chi2, Rval) / norm
    
    def _argcheck(self, *args):
        
        return 1

class rv_moreless(stats.rv_continuous):
    
    def __init__(self, **kwargs):
        
        self.Rp_lims = [1., 3.5] # Rearth
        self.Porb_lims = [2., 100.] # d
        
        super(rv_moreless, self).__init__(**kwargs)
        
        pass
    
    def t(self, Porb, Pcen, s):
        
        return 0.5 - 0.5 * np.tanh((np.log10(Porb) - np.log10(Pcen)) / np.log10(s))
    
    def G_less(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * chi1 + (1. - self.t(Porb, Pcen, s)) * chi2
    
    def G_more(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * (1. - chi1) + (1. - self.t(Porb, Pcen, s)) * (1. - chi2)
    
    def dNdP_moreless(self, Porb, Pbrk, beta2, Pcen, s, chi1, chi2, Rval):
        
        return (Porb / Pbrk)**beta2 * self.G_less(Porb, Pcen, s, chi1, chi2) * (self.Rp_lims[1] - Rval) / ((Rval - self.Rp_lims[0]) + self.G_less(Porb, Pcen, s, chi1, chi2) * (self.Rp_lims[1] - Rval) - self.G_less(Porb, Pcen, s, chi1, chi2) * (Rval - self.Rp_lims[0]))
    
    def _pdf(self, Porb, Pbrk, beta2, Pcen, s, chi1, chi2, Rval, norm):
        
        return self.dNdP_moreless(Porb, Pbrk, beta2, Pcen, s, chi1, chi2, Rval) / norm
    
    def _argcheck(self, *args):
        
        return 1

class rv_moremore(stats.rv_continuous):
    
    def __init__(self, **kwargs):
        
        self.Rp_lims = [1., 3.5] # Rearth
        self.Porb_lims = [2., 100.] # d
        
        super(rv_moremore, self).__init__(**kwargs)
        
        pass
    
    def t(self, Porb, Pcen, s):
        
        return 0.5 - 0.5 * np.tanh((np.log10(Porb) - np.log10(Pcen)) / np.log10(s))
    
    def G_less(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * chi1 + (1. - self.t(Porb, Pcen, s)) * chi2
    
    def G_more(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * (1. - chi1) + (1. - self.t(Porb, Pcen, s)) * (1. - chi2)
    
    def dNdP_moremore(self, Porb, Pbrk, beta2, Pcen, s, chi1, chi2, Rval):
        
        return (Porb / Pbrk)**beta2 * self.G_more(Porb, Pcen, s, chi1, chi2) * (self.Rp_lims[1] - Rval) / ((Rval - self.Rp_lims[0]) + self.G_more(Porb, Pcen, s, chi1, chi2) * (self.Rp_lims[1] - Rval) - self.G_more(Porb, Pcen, s, chi1, chi2) * (Rval - self.Rp_lims[0]))
    
    def _pdf(self, Porb, Pbrk, beta2, Pcen, s, chi1, chi2, Rval, norm):
        
        return self.dNdP_moremore(Porb, Pbrk, beta2, Pcen, s, chi1, chi2, Rval) / norm
    
    def _argcheck(self, *args):
        
        return 1

class PlanetDistribution():
    """
    https://ui.adsabs.harvard.edu/abs/2022AJ....164..190B/abstract
    """
    
    def __init__(self,
                 Scenario):
        """
        Parameters
        ----------
        Scenario: 'baseline', 'pessimistic', 'optimistic', 'mc'
            Scenario for planet occurrence rates.
        """
        
        # Print.
        print('--> Initializing Bergsten2022 planet distribution')
        
        # Model parameters.
        self.returns = ['Rp', 'Porb']
        self.Ms_bins = [0.56, 0.81, 0.91, 1.01, 1.16, 1.63] # Msun
        self.F0 = [0.89, 0.70, 0.63, 0.61, 0.50]
        self.Pbrk = [14.29, 6.13, 6.92, 12.02, 6.96] # d
        self.beta1 = [0.15, 1.19, 0.91, 0.44, 1.90]
        self.beta2 = [-1.15, -0.68, -0.84, -1.10, -0.69]
        self.Pcen = [7.54, 11.26, 12.98, 17.57, 16.57] # d
        self.s = [1.46, 2.02, 2.53, 2.16, 2.15] # d
        self.chi1 = [0.75, 0.73, 0.83, 0.83, 0.87]
        self.chi2 = [0.33, 0.36, 0.26, 0.31, 0.39]
        self.Rval = [1.82, 1.93, 1.98, 2.04, 2.17]
        self.Rp_lims = [1., 3.5] # Rearth
        self.Porb_lims = [2., 100.] # d
        self.C_lessless_norm = []
        self.C_lessless = []
        self.C_lessmore_norm = []
        self.C_lessmore = []
        self.C_moreless_norm = []
        self.C_moreless = []
        self.C_moremore_norm = []
        self.C_moremore = []
        for i in range(len(self.F0)):
            self.C_lessless_norm += [quad(self.dNdP_lessless, self.Porb_lims[0], self.Pbrk[i], args=(self.Pbrk[i], self.beta1[i], self.Pcen[i], self.s[i], self.chi1[i], self.chi2[i], self.Rval[i]))[0]]
            self.C_lessless += [(self.Rval[i] - self.Rp_lims[0]) * self.C_lessless_norm[-1]]
            self.C_lessmore_norm += [quad(self.dNdP_lessmore, self.Porb_lims[0], self.Pbrk[i], args=(self.Pbrk[i], self.beta1[i], self.Pcen[i], self.s[i], self.chi1[i], self.chi2[i], self.Rval[i]))[0]]
            self.C_lessmore += [(self.Rp_lims[1] - self.Rval[i]) * self.C_lessmore_norm[-1]]
            self.C_moreless_norm += [quad(self.dNdP_moreless, self.Pbrk[i], self.Porb_lims[1], args=(self.Pbrk[i], self.beta2[i], self.Pcen[i], self.s[i], self.chi1[i], self.chi2[i], self.Rval[i]))[0]]
            self.C_moreless += [(self.Rval[i] - self.Rp_lims[0]) * self.C_moreless_norm[-1]]
            self.C_moremore_norm += [quad(self.dNdP_moremore, self.Pbrk[i], self.Porb_lims[1], args=(self.Pbrk[i], self.beta2[i], self.Pcen[i], self.s[i], self.chi1[i], self.chi2[i], self.Rval[i]))[0]]
            self.C_moremore += [(self.Rp_lims[1] - self.Rval[i]) * self.C_moremore_norm[-1]]
        
        pass
    
    def t(self, Porb, Pcen, s):
        
        return 0.5 - 0.5 * np.tanh((np.log10(Porb) - np.log10(Pcen)) / np.log10(s))
    
    def G_less(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * chi1 + (1. - self.t(Porb, Pcen, s)) * chi2
    
    def G_more(self, Porb, Pcen, s, chi1, chi2):
        
        return self.t(Porb, Pcen, s) * (1. - chi1) + (1. - self.t(Porb, Pcen, s)) * (1. - chi2)
    
    def dNdP_lessless(self, Porb, Pbrk, beta1, Pcen, s, chi1, chi2, Rval):
        
        return (Porb / Pbrk)**beta1 * self.G_less(Porb, Pcen, s, chi1, chi2) * (self.Rp_lims[1] - Rval) / ((Rval - self.Rp_lims[0]) + self.G_less(Porb, Pcen, s, chi1, chi2) * (self.Rp_lims[1] - Rval) - self.G_less(Porb, Pcen, s, chi1, chi2) * (Rval - self.Rp_lims[0]))
    
    def dNdP_lessmore(self, Porb, Pbrk, beta1, Pcen, s, chi1, chi2, Rval):
        
        return (Porb / Pbrk)**beta1 * self.G_more(Porb, Pcen, s, chi1, chi2) * (self.Rp_lims[1] - Rval) / ((Rval - self.Rp_lims[0]) + self.G_more(Porb, Pcen, s, chi1, chi2) * (self.Rp_lims[1] - Rval) - self.G_more(Porb, Pcen, s, chi1, chi2) * (Rval - self.Rp_lims[0]))
    
    def dNdP_moreless(self, Porb, Pbrk, beta2, Pcen, s, chi1, chi2, Rval):
        
        return (Porb / Pbrk)**beta2 * self.G_less(Porb, Pcen, s, chi1, chi2) * (self.Rp_lims[1] - Rval) / ((Rval - self.Rp_lims[0]) + self.G_less(Porb, Pcen, s, chi1, chi2) * (self.Rp_lims[1] - Rval) - self.G_less(Porb, Pcen, s, chi1, chi2) * (Rval - self.Rp_lims[0]))
    
    def dNdP_moremore(self, Porb, Pbrk, beta2, Pcen, s, chi1, chi2, Rval):
        
        return (Porb / Pbrk)**beta2 * self.G_more(Porb, Pcen, s, chi1, chi2) * (self.Rp_lims[1] - Rval) / ((Rval - self.Rp_lims[0]) + self.G_more(Porb, Pcen, s, chi1, chi2) * (self.Rp_lims[1] - Rval) - self.G_more(Porb, Pcen, s, chi1, chi2) * (Rval - self.Rp_lims[0]))
    
    def draw(self,
             Rp_range=[1., 3.5], # Rearth
             Porb_range=[2., 100.], # d
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
        
        Rp = [] # Rearth
        Porb = [] # d
        
        # Apply scaling for the planet occurrence rates.
        tempF0 = np.array(self.F0).copy()*Scale
        
        # Select the correct stellar mass bin.
        if (Star is not None):
            ww = np.where(np.array(self.Ms_bins) > Star.Mass)[0]
            if (len(ww) == 0):
                ww = 4
            elif (np.min(ww) == 0):
                ww = 0
            else:
                ww = np.min(ww) - 1
        else:
            ww = 2
        
        # If the number of planets is not given, draw it from a Poisson
        # distribution. Note that the final number of drawn planets might be
        # smaller than the drawn number because of clipping to the requested Rp
        # and Porb range.
        if (Nplanets is None):
            Nplanets = np.random.poisson(tempF0[ww])
            for i in range(Nplanets):
                
                # Randomly select whether Porb < Pbrk or Porb > Pbrk and Rp < Rval or Rp > Rval.
                p = np.array([self.C_lessless[ww], self.C_lessmore[ww], self.C_moreless[ww], self.C_moremore[ww]])
                temp = np.random.choice(4, p=p/np.sum(p))
                if (temp == 0):
                    rv = rv_lessless(a=self.Porb_lims[0], b=self.Pbrk[ww])
                    tempRp = self.Rp_lims[0] + np.random.rand() * (self.Rval[ww] - self.Rp_lims[0]) # Rearth
                    tempPorb = rv.rvs(Pbrk=self.Pbrk[ww], beta1=self.beta1[ww], Pcen=self.Pcen[ww], s=self.s[ww], chi1=self.chi1[ww], chi2=self.chi2[ww], Rval=self.Rval[ww], norm=self.C_lessless_norm[ww]) # d
                elif (temp == 1):
                    rv = rv_lessmore(a=self.Porb_lims[0], b=self.Pbrk[ww])
                    tempRp = self.Rval[ww] + np.random.rand() * (self.Rp_lims[1] - self.Rval[ww]) # Rearth
                    tempPorb = rv.rvs(Pbrk=self.Pbrk[ww], beta1=self.beta1[ww], Pcen=self.Pcen[ww], s=self.s[ww], chi1=self.chi1[ww], chi2=self.chi2[ww], Rval=self.Rval[ww], norm=self.C_lessmore_norm[ww]) # d
                elif (temp == 2):
                    rv = rv_moreless(a=self.Pbrk[ww], b=self.Porb_lims[1])
                    tempRp = self.Rp_lims[0] + np.random.rand() * (self.Rval[ww] - self.Rp_lims[0]) # Rearth
                    tempPorb = rv.rvs(Pbrk=self.Pbrk[ww], beta2=self.beta2[ww], Pcen=self.Pcen[ww], s=self.s[ww], chi1=self.chi1[ww], chi2=self.chi2[ww], Rval=self.Rval[ww], norm=self.C_moreless_norm[ww]) # d
                elif (temp == 3):
                    rv = rv_moremore(a=self.Pbrk[ww], b=self.Porb_lims[1])
                    tempRp = self.Rval[ww] + np.random.rand() * (self.Rp_lims[1] - self.Rval[ww]) # Rearth
                    tempPorb = rv.rvs(Pbrk=self.Pbrk[ww], beta2=self.beta2[ww], Pcen=self.Pcen[ww], s=self.s[ww], chi1=self.chi1[ww], chi2=self.chi2[ww], Rval=self.Rval[ww], norm=self.C_moremore_norm[ww]) # d
                if (Rp_range[0] <= tempRp <= Rp_range[1] and Porb_range[0] <= tempPorb <= Porb_range[1]):
                    Rp += [tempRp] # Rearth
                    Porb += [tempPorb] # d
        
        # If the number of planets is given, draw exactly this number of
        # planets in the requested Rp and Porb range.
        else:
            while (len(Rp) < Nplanets):
                
                # Randomly select whether Porb < Pbrk or Porb > Pbrk and Rp < Rval or Rp > Rval.
                p = np.array([self.C_lessless[ww], self.C_lessmore[ww], self.C_moreless[ww], self.C_moremore[ww]])
                temp = np.random.choice(4, p=p/np.sum(p))
                if (temp == 0):
                    rv = rv_lessless(a=self.Porb_lims[0], b=self.Pbrk[ww])
                    tempRp = self.Rp_lims[0] + np.random.rand() * (self.Rval[ww] - self.Rp_lims[0]) # Rearth
                    tempPorb = rv.rvs(Pbrk=self.Pbrk[ww], beta1=self.beta1[ww], Pcen=self.Pcen[ww], s=self.s[ww], chi1=self.chi1[ww], chi2=self.chi2[ww], Rval=self.Rval[ww], norm=self.C_lessless_norm[ww]) # d
                elif (temp == 1):
                    rv = rv_lessmore(a=self.Porb_lims[0], b=self.Pbrk[ww])
                    tempRp = self.Rval[ww] + np.random.rand() * (self.Rp_lims[1] - self.Rval[ww]) # Rearth
                    tempPorb = rv.rvs(Pbrk=self.Pbrk[ww], beta1=self.beta1[ww], Pcen=self.Pcen[ww], s=self.s[ww], chi1=self.chi1[ww], chi2=self.chi2[ww], Rval=self.Rval[ww], norm=self.C_lessmore_norm[ww]) # d
                elif (temp == 2):
                    rv = rv_moreless(a=self.Pbrk[ww], b=self.Porb_lims[1])
                    tempRp = self.Rp_lims[0] + np.random.rand() * (self.Rval[ww] - self.Rp_lims[0]) # Rearth
                    tempPorb = rv.rvs(Pbrk=self.Pbrk[ww], beta2=self.beta2[ww], Pcen=self.Pcen[ww], s=self.s[ww], chi1=self.chi1[ww], chi2=self.chi2[ww], Rval=self.Rval[ww], norm=self.C_moreless_norm[ww]) # d
                elif (temp == 3):
                    rv = rv_moremore(a=self.Pbrk[ww], b=self.Porb_lims[1])
                    tempRp = self.Rval[ww] + np.random.rand() * (self.Rp_lims[1] - self.Rval[ww]) # Rearth
                    tempPorb = rv.rvs(Pbrk=self.Pbrk[ww], beta2=self.beta2[ww], Pcen=self.Pcen[ww], s=self.s[ww], chi1=self.chi1[ww], chi2=self.chi2[ww], Rval=self.Rval[ww], norm=self.C_moremore_norm[ww]) # d
                if (Rp_range[0] <= tempRp <= Rp_range[1] and Porb_range[0] <= tempPorb <= Porb_range[1]):
                    Rp += [tempRp] # Rearth
                    Porb += [tempPorb] # d
        
        return np.array(Rp), np.array(Porb)
    
    def SummaryPlot(self,
                    Ntest=100000,
                    Rp_range=[1., 3.5], # Rearth
                    Porb_range=[2., 100.], # d
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
        
        Rp = []
        Porb = []
        for i in range(Ntest):
            tempRp, tempPorb = self.draw(Rp_range,
                                         Porb_range)
            Rp += [tempRp]
            Porb += [tempPorb]
        Rp = np.concatenate(Rp)
        Porb = np.concatenate(Porb)
        
        print('--> Bergsten2022:\n%.2f planets/star in [%.1f, %.1f] Rearth and [%.1f, %.1f] d' % (len(Rp)/float(Ntest), Rp_range[0], Rp_range[1], Porb_range[0], Porb_range[1]))
        
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
        plt.suptitle('Bergsten2022')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if (FigDir is not None):
            plt.savefig(FigDir+'PlanetDistribution_Bergsten2022.pdf')
        plt.show(block=block)
        plt.close()
        
        pass
