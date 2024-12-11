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
import scipy.optimize as so


# =============================================================================
# RANDOM
# =============================================================================

class OrbitModel():
    
    def __init__(self):
        
        pass
    
    def getOrbit(self,
                 ep,
                 fixip=True):
        """
        Parameters
        ----------
        ep: array
            Eccentricity of drawn planets.
        fixip: bool
            If True, returns same inclination for all drawn planets.
        
        Returns
        -------
        ip: array
            Inclination (rad) of drawn planets.
        Omegap: array
            Longitude of the ascending node (rad) of drawn planets.
        omegap: array
            Argument of periapsis (rad) of drawn planets.
        thetap: array
            True anomaly (rad) of drawn planets.
        """
        
        ip = np.deg2rad(90.)*np.ones(len(ep)) # rad; face-on
        Omegap = np.deg2rad(90.)*np.ones(len(ep)) # rad
        omegap = np.deg2rad(0.)*np.ones(len(ep)) # rad
        thetap = np.deg2rad(0.)*np.ones(len(ep)) # rad
        
        return ip, Omegap, omegap, thetap
    
    def SummaryPlot(self,
                    EccentricityModel,
                    Ntest=100000,
                    FigDir=None,
                    block=True):
        """
        Parameters
        ----------
        EccentricityModel: instance
            Instance of class EccentricityModel.
        Ntest: int
            Number of test draws for summary plot.
        FigDir: str
            Directory to which summary plots are saved.
        block: bool
            If True, blocks plots when showing.
        """
        
        ep = EccentricityModel.getEccentricity(np.zeros(Ntest))
        ip, Omegap, omegap, thetap = self.getOrbit(ep,
                                                   fixip=False)
        
        print('--> Quadrature:\n%.2f+-%.2f rad planet inclination\n%.2f+-%.2f rad planet true anomaly' % (np.mean(ip), np.std(ip), np.mean(thetap), np.std(thetap)))
        
        Weight = 1./len(ip)
        f, ax = plt.subplots(2, 2)
        ax[0, 0].hist(ip, bins=25, weights=np.ones_like(ip)*Weight)
        ax[0, 0].grid(axis='y')
        ax[0, 0].set_xlabel('Planet inclination [rad]')
        ax[0, 0].set_ylabel('Fraction')
        ax[0, 1].hist(thetap, bins=25, weights=np.ones_like(thetap)*Weight)
        ax[0, 1].grid(axis='y')
        ax[0, 1].set_xlabel('Planet true anomaly [rad]')
        ax[0, 1].set_ylabel('Fraction')
        ax[1, 0].hist(Omegap, bins=25, weights=np.ones_like(Omegap)*Weight)
        ax[1, 0].grid(axis='y')
        ax[1, 0].set_xlabel('Planet lon. of asc. node [rad]')
        ax[1, 0].set_ylabel('Fraction')
        ax[1, 1].hist(omegap, bins=25, weights=np.ones_like(omegap)*Weight)
        ax[1, 1].grid(axis='y')
        ax[1, 1].set_xlabel('Planet arg. of per. [rad]')
        ax[1, 1].set_ylabel('Fraction')
        plt.suptitle('Random')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if (FigDir is not None):
            plt.savefig(FigDir+'OrbitModel.pdf')
        plt.show(block=block)
        plt.close()
        
        pass
