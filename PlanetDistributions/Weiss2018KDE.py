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
        Scenario: 'baseline', 'pessimistic', 'optimistic', 'mc'
            Scenario for planet occurrence rates.
        """
        
        # Print.
        print('--> Initializing Weiss2018KDE planet distribution')
        
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
        elif (Scenario == 'mc'):
            raise UserWarning('Scenario mc is not implemented yet')
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
        self.RadiusKDE = stats.gaussian_kde(self.RadiusData())
        self.PeriodKDE = stats.gaussian_kde(self.PeriodData())
        
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
            Rp = self.RadiusCorrKernel(Rp, # Rearth
                                       self.RadiusKDE)
            Porb = self.PeriodCorrKernel(Porb, # d
                                         self.PeriodKDE)
        
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
    
    def RadiusCorrKernel(self,
                         Rp, # Rearth
                         KDE,
                         Ratio_lims=[0., np.inf],
                         Rp_lims=[0.1, 100.]): # Rearth
        """
        Parameters
        ----------
        Rp: array
            Radius (Rearth) of drawn planets sorted by orbital period.
        KDE: class
            Instance of class scipy.stats.gaussian_kde.
        Ratio_lims: list
            Requested planet radius ratio range.
        Rp_lims: list
            Requested planet radius range (Rearth).
        
        Returns
        -------
        Rp: array
            Radius (Rearth) of drawn planets according to Weiss et al. 2018.
        """
        
        for i in range(len(Rp)-1):
            
            # Draw radius ratio inside limits.
            RadiusRatio = -1.
            Rp[i+1] = -1.
            Trials = 1
            while (Trials <= self.MaxTrials and (RadiusRatio <= Ratio_lims[0] or RadiusRatio >= Ratio_lims[1] or Rp[i+1] <= Rp_lims[0] or Rp[i+1] >= Rp_lims[1])):
                RadiusRatio = KDE.resample(1)[0][0]
                Rp[i+1] = Rp[i]*RadiusRatio # Rearth
                Trials += 1
        
        return Rp
    
    def PeriodCorrKernel(self,
                         Porb, # d
                         KDE,
                         Ratio_lims=[0., np.inf],
                         Porb_lims=[0., np.inf]): # d
        """
        Parameters
        ----------
        Porb: array
            Orbital period (d) of drawn planets sorted by orbital period.
        KDE: class
            Instance of class scipy.stats.gaussian_kde.
        Ratio_lims: list
            Requested planet orbital period ratio range.
        Porb_lims: list
            Requested planet orbital period range (d).
        
        Returns
        -------
        Porb: array
            Orbital period (d) of drawn planets according to Weiss et al. 2018.
        """
        
        for i in range(len(Porb)-1):
            
            # Draw period ratio inside limits.
            PeriodRatio = -1.
            Porb[i+1] = -1.
            Trials = 1
            while (Trials <= self.MaxTrials and (PeriodRatio <= Ratio_lims[0] or PeriodRatio >= Ratio_lims[1] or Porb[i+1] <= Porb_lims[0] or Porb[i+1] >= Porb_lims[1])):
                PeriodRatio = KDE.resample(1)[0][0]
                Porb[i+1] = Porb[i]*PeriodRatio # d
                Trials += 1
        
        return Porb
    
    def RadiusData(self):
        """
        Returns
        -------
        Rp: array
            TBD by Emile!
        """
        
        return np.array([1.752315450359564153e+00,6.487212967096627203e-01,2.072236874378123872e-01,3.911745565908651412e-01,3.674905510080838766e+00,3.276746365834301011e-01,2.644416021524667482e+00,1.571072260923016772e+00,1.176647192467850322e+00,2.422241763774232837e+00,1.798394982538083386e+00,3.124876976634854975e-01,1.704825252417857762e+00,6.887197458169287412e-01,2.363266726663697526e+00,1.728166262333804770e+00,9.922464998520554857e-01,1.116109687158163499e+00,5.730444853796422544e-01,2.777200206029148077e+00,4.217503543231215790e-01,2.602977189078567122e+00,1.177588027987660046e+00,5.831473786837220175e-01,2.862345411263644834e+00,1.087964589211180000e+00,1.197703016912988305e+00,2.366334644676807208e+00,1.197485543245306872e+00,1.380302207107484636e+00,7.741652105179597232e-01,1.030719396772091789e+00,1.085312810200593114e+00,1.925621282203020757e+00,3.899591920792160327e-01,1.332210428151989978e+00,1.291352262192712486e+00,1.591267483013472539e+00,1.592252863198919233e+00,1.104937204321240873e+00,1.194737118477405824e+00,7.016479141631303262e-01,1.267959395034442105e+00,2.334609366940362030e+00,1.653987880143361178e+00,7.239115353247970175e-01,6.645752210955137018e-01,7.548304472645459562e-01,1.143795011975400344e+00,2.228109518566096625e+00,7.347454904188286395e-01,2.541743201007026443e+00,3.253785234452130082e-01,1.194233027876757447e+00,5.191014061029793902e-01,2.727395187053456294e+00,5.707066474785181809e-01,1.711817289132552711e+00,2.660375161076542483e+00,4.007266977983311995e-01,1.012203540688284997e+00,1.587544064452571346e+00,1.177255122309648572e+00,7.693738580049901410e-01,1.651045254517583327e+00,1.051230266892242859e+00,9.840454682662348462e-01,1.042538037362666925e+00,2.665519722305699890e+00,2.380099835990298729e+00,2.765700933897586200e+00,4.796245986743473688e-01,3.911817529902205659e-01,6.614661154026845180e-01,8.422965284214087855e-01,6.315184611931508041e-01,6.172197121948466148e-01,9.165546125614099093e-01,1.295344358545347019e+00,1.402431680087352639e+00,1.031081694940976456e+00,5.732245628646835955e-01,1.145552537716610475e+00,2.088244754894187150e+00,3.722752253030616743e-01,8.571251820786791598e-01,2.387917205609837890e+00,1.041373964596457924e+00,1.030184661672609359e+00,1.373282563275870727e+00,1.406736405582328375e+00,5.144339199770274762e-01,1.113393334056058670e+00,1.980036012399210232e+00,9.158266152954639416e-01,1.055305652174514774e+00,2.794497109014885794e+00,2.643535139985301874e-01,1.293852654522842149e+00,9.934299750580419808e-01,1.746025045138129173e+00,9.994646397872030130e-01,9.011745546369110960e-01,4.382468436568719983e-01,2.141161372980054267e+00,2.762911097512532610e+00,7.777252984124100044e-01,1.023996563559364192e+00,8.309812732220078768e-01,7.288816282244545697e-01,8.706958879520259043e-01,9.692410084994308450e-01,2.621188586853730573e+00,5.982997638477258739e-01,1.377968258351409681e+00,1.002311216222491641e+00,1.415221513950786436e+00,1.752765769788386496e+00,9.460850696343965360e-01,2.559642416817784927e+00,2.643673267422145035e+00,1.855621153209402019e+00,9.121537391346357104e-01,9.116010391379917310e-01,1.511106498904367434e+00,1.044651605034034869e+00,1.024810475647266239e+00,7.863117181857955940e-01,1.343677960609663158e+00,1.025463900042556276e+00,1.608465254790204080e+00,1.028790505975090808e+00,1.134478003718337291e+00,1.304045116043936359e+00,1.432896553816479290e+00,1.134623797853904881e+00,9.294955669467748205e-01,1.080770370711439554e+00,6.970790896751382082e-01,1.064282808387510215e+00,9.384382053316351335e-01,1.092547052339658986e+00,1.878507394986311807e+00,7.776999648778283314e-01,1.440013940955226612e+00,1.187992087626337012e+00,1.871199792310260568e+00,1.095919784120203078e+00,5.288387897604686083e-01,1.034315015531350523e+00,1.169220301406295626e+00,1.427161353141061406e+00,1.564450697157371062e+00,1.360498523622858835e+00,1.091049860420979511e+00,1.900086112531333304e+00,1.939152261383264797e+00,8.417643717538902948e-01,1.239506110281250351e+00,1.663674295548200233e+00,1.614193103340648694e+00,2.058353164979547678e+00,1.844267740880617179e+00,2.104379965482511494e+00,1.025999291350686127e+00,5.926411527143984337e-01,1.574350890880808373e+00,2.488442660026523701e+00,1.030074770975389020e+00,1.169449632242443027e+00,1.759953995290411966e+00,9.276312598009209820e-01,7.884353244387912740e-01,1.178028591597828356e+00,1.790289689570631060e+00,1.162601042768724335e+00,1.170938895990142559e+00,9.135188411538208708e-01,7.744451973299991687e-01,1.895231883828319530e+00,9.286616019761447793e-01,2.386687694849365826e+00,1.827552304872278377e+00,1.051227560168768171e+00,8.496556571949922532e-01,1.067003065000965423e+00,9.309621070867275527e-01,5.929758318587861066e-01,6.115234229501376184e-01,1.419071783721094437e+00,6.039678234347791719e-01,9.749778830336751634e-01,2.177189523156829676e+00,8.885684168361590585e-01,8.663829791344208608e-01,1.176710404094525320e+00,1.306718135044355744e+00,1.012731738060630660e+00,2.360083793343005443e+00,1.382943456044909647e+00,8.701027530248188757e-01,1.597758684123475881e+00,7.144171386417875746e-01,8.839789985421342422e-01,8.208802875245966302e-01,1.736877069654418904e+00,1.778622878367817517e+00,7.663347664114421587e-01,1.133946766778392234e+00,1.314993975734128240e+00,1.584196298798070224e+00,8.195796368631418360e-01,1.728942018012803006e+00,9.369146906091380123e-01,1.205248559284884280e+00,8.004141631121818312e-01,1.956747987877118122e+00,6.281619273652051527e-01,1.289393014313743979e+00,1.839668943074462604e+00,9.179163620136380208e-01,9.632183494964169368e-01,1.178538094221683430e+00,3.185645554077177199e-01,2.215524545240109777e+00,7.673376172158677955e-01,7.863683293525401963e-01,1.695731864699311364e+00,4.421217339496746335e-01,1.007915165578139094e+00,1.925664167303240193e+00,1.334456028454019672e+00,2.191867649320174749e+00,6.855131378324016422e-01,9.719083864453205157e-01,3.065330608463485262e+00,1.514769804892346139e+00,1.779151004490856725e+00,1.150991328672114289e+00,2.667245639585551498e+00,3.704437909339096113e-01,9.248332609056432174e-01,1.495374986041666743e+00,1.754347376641212319e+00,1.463373560227288150e+00,4.493641295537769698e-01,6.956494603138225763e-01,6.968051380506099513e-01,1.389611506901073934e+00,1.549517797029153376e+00,8.509639049595334459e-01,1.381412713450395069e+00,1.007489881594998637e+00,6.561111464037397978e-01,1.438390089589695942e+00,1.841393801159408206e+00,1.300144135219158281e+00,2.637280915755968014e+00,1.148291379007859936e+00,2.033087279710555251e+00,9.228840758024525748e-01,1.988774132447459042e+00,1.181783828037265671e+00,9.108272884067711850e-01,1.603855606885066454e+00,1.164869412232308665e+00,6.184601709186825502e-01,1.143519013702266607e+00,3.021673434844676098e+00,9.430586013500916742e-01,8.145116166695167692e-01,1.303880519661693382e+00,9.298728238026602311e-01,1.091765838502267494e+00,1.786831337961090060e+00,1.184695021583211494e+00,1.013723952003227513e+00,7.711241792072506263e-01,9.426896268871111095e-01,1.176732202249067827e+00,8.541742628700740303e-01,8.524277982941398379e-01,1.874508433123826201e+00,6.454455342550922126e-01,1.452472916435484152e+00,9.131535388628666539e-01,1.069496645352042252e+00,8.912916524919033412e-01,8.611608552815684936e-01,2.000463830409365595e+00,1.468474700787375742e+00,1.072973533043674710e+00,6.603359415586382841e-01,8.171479110789710010e-01,1.453337768571333344e+00,1.283716695515920048e+00,6.175576557230243457e-01,1.073180513158835003e+00,1.253509627461631970e+00,1.050558319126701390e+00,1.058209305270770617e+00,1.105707096292035008e+00,1.081996331208107343e+00,1.472125346293450709e+00,9.293495680493915057e-01,1.194968316492199811e+00,8.143156803203051020e-01,9.441949389614332500e-01,7.967518828372992523e-01,1.026148721497799121e+00,1.304422362954630543e+00,1.739606761870353235e+00,1.880418569102280335e+00,1.125082387017163343e+00,1.628854407430581741e+00,1.142922613965407397e+00,7.929420652265271041e-01,1.072656259288860214e+00,9.989150581809435048e-01,1.840365736821221843e+00,1.573069325920356532e+00,8.882252934677035139e-01,1.083547342665800661e+00,1.441497307745600587e+00,1.108460533157596517e+00,1.029615200036867240e+00,1.268221049706181658e+00,1.564867978495868561e+00,8.288600539901830322e-01,2.277651206227665259e+00,1.612977007892966386e+00,8.814727919780673826e-01,9.845198928194719423e-01,1.038817048559580902e+00,1.407579521180028159e+00,1.597618152236183198e+00,1.546492383603808474e+00,2.662796386054001374e+00,7.382190200640917910e-01,1.158834296597492042e+00,1.916847325497079257e+00,9.537730795047474341e-01,8.047970388950977938e-01,1.419929721330409400e+00,9.928198544063048248e-01,1.067237183416991364e+00,9.357053534243530146e-01,1.410596801793942312e+00,2.742210488789099898e+00,7.129502940394480959e-01,9.257366522851745438e-01,8.965451501030846648e-01,3.551984254424210619e+00,7.788624511147782536e-01,1.737677828643085620e+00,1.611012177238849041e+00,8.746302083514626569e-01,1.262024385073194033e+00,1.650902859963576885e+00,1.135876452424817495e+00,1.370291803417524434e+00,1.018316554026156950e+00,1.008470698474726701e+00,1.557288028975604810e+00,9.675953924814275631e-01,5.777836410180877680e-01,1.272527441992522990e+00,1.725184851883340587e+00,2.004491209884344816e+00,1.683215949547688162e+00,8.933099048747840110e-01,1.087995793380434506e+00,1.338274116398945113e+00,1.012626920273352882e+00,8.845792284956054408e-01,1.164353027018574238e+00,1.655212135144163765e+00,1.358831484530935718e+00,1.416154643913164257e+00,9.647497909142274120e-01,1.174654975271542723e+00,8.133475238865712775e-01,9.853926451850230084e-01,1.511822643247775266e+00,1.126955195781791597e+00,8.736497449706578378e-01,8.653781558288203835e-01,1.203283864411869075e+00,1.215672207748022426e+00,8.265875119952927896e-01,9.261036336079200293e-01,8.032561761189063176e-01,9.173090977282861713e-01,1.042920204911105486e+00,1.409570685436371651e+00,1.010633838082865221e+00,1.863884720994973820e+00,1.885650592936956915e+00,1.248087631471638037e+00,1.311483616016024589e+00,1.359556230464705306e+00,2.063501382485267044e+00,1.073995584091294031e+00,8.395573182138228452e-01,1.220874773344300168e+00,1.303938089239660592e+00,9.714346682330354810e-01,1.792957923082562743e+00,1.352181867685576799e+00,1.263508579397745191e+00,1.375415000765877416e+00,7.613932795444554591e-01,9.681639456822033196e-01,9.238979506827899035e-01,1.205245397609315594e+00,9.950017726587676581e-01,1.323769020612526992e+00,1.172189521251834288e+00,1.160248825124767791e+00,1.468414349377322070e+00,9.118066690086275017e-01,8.310904470342121986e-01,1.050370862244759396e+00,1.168945913505493817e+00,8.061385719475674394e-01,1.018341485231998078e+00,1.539528422603151814e+00,1.806459199049707420e+00,1.361270279834406516e+00,7.551786100408791613e-01,1.466280536680966140e+00,1.160993941357403303e+00,1.001453150936619574e+00,1.077008128821638655e+00,1.256023969486314451e+00,7.212618500704786051e-01,1.149106152824353355e+00,1.137447184848604964e+00,1.082271730097562390e+00,9.249829563410723043e-01,1.212202366028331868e+00,9.935943965563075464e-01,9.882022064610020973e-01,1.034225719489145323e+00,1.679568526769005699e+00,1.111653899610143714e+00,7.959228467694859921e-01,8.485909243808207325e-01,1.079784162424551885e+00,8.867271093478774624e-01,1.237496418678406318e+00,9.293366440193786548e-01,1.144254833761478851e+00,1.051070838776300365e+00,9.790063540141226150e-01,1.738102280777396924e+00,1.010893601805699094e+00,2.145217911876831529e+00,1.315245522409542511e+00,1.097139987113683546e+00,1.562935225313065279e+00,1.242161011912065183e+00,1.372326315287416998e+00,9.624543165895705910e-01,1.258908134151504710e+00,7.909181846072025346e-01,1.259354155657684826e+00,7.590277153277544819e-01,9.971134585797761796e-01,7.948665091602381549e-01,1.255841045092199959e+00,1.202181548425998781e+00,9.453694054572131522e-01,1.470207014703439841e+00,1.085347098434903623e+00,9.444742134799756217e-01,1.262617224189948040e+00,1.030716621625571205e+00,1.048009419755302485e+00,1.198465278567024450e+00,1.260979600790656852e+00,1.232912654680249798e+00,9.975599068158217397e-01,2.037447708927588419e+00,1.145967155143265304e+00,9.973791874710399297e-01,1.094513653627082306e+00,1.050932874605420286e+00,9.912018204069623639e-01])
    
    def PeriodData(self):
        """
        Returns
        -------
        Porb: array
            TBD by Emile!
        """
        
        return np.array([1.860864155601992920e+00,2.756976202853799851e+00,1.649983565534847152e+00,1.779787239683948030e+00,1.803708506926315458e+00,3.964300288136020534e+00,1.337511214419654593e+00,1.458239491659028220e+00,1.565758509725915015e+00,1.700375028091736906e+00,2.719343886885984052e+00,1.387595540798557625e+00,2.784715172733939426e+00,2.143482376251296007e+00,2.431189186529074231e+00,2.071167776026533680e+00,2.186686632400727248e+00,2.201289377951328952e+00,1.767050910114344342e+00,1.828357557616575146e+00,1.541320786025923262e+00,1.623666778268607169e+00,1.853301732388607848e+00,2.180385958879117680e+00,1.944483883037936955e+00,2.024685081324506530e+00,2.032121680941550768e+00,1.900963653009758003e+00,1.556195513790302298e+00,1.549818925668807301e+00,1.464451364132892675e+00,1.264064626125660951e+00,1.741824654927329741e+00,1.410292560419374563e+00,1.459136231500989700e+00,2.535639807274684632e+00,1.511539683826262470e+00,1.421874344741921981e+00,3.412838217517561112e+00,2.161920815683406438e+00,1.731741953783189736e+00,2.606034911938465370e+00,2.239337284283846152e+00,2.083684081215940953e+00,2.036116923221890307e+00,1.654473628072183011e+00,2.051120703063785022e+00,1.846153406966596133e+00,3.252590280555476276e+00,3.181448597416078883e+00,1.943671802006837313e+00,1.846967198414354527e+00,2.158741256605364978e+00,2.794477732061942277e+00,2.352451954599862560e+00,1.244217922856161174e+00,1.539086441230591573e+00,1.358665560215821744e+00,1.685953299287025642e+00,1.574530914157604711e+00,2.019004226669854241e+00,1.894880372514881728e+00,2.892177170282875132e+00,2.153409422373183890e+00,1.701566873996222862e+00,2.454155568614533767e+00,1.867927183118335499e+00,2.522235049076735347e+00,2.071688634804673246e+00,1.607827483703416993e+00,1.885645421876658068e+00,2.173908635781124143e+00,2.648221799009809807e+00,1.685828335782244114e+00,2.944039843330661377e+00,3.266939035055890006e+00,3.113284844972819876e+00,1.512100093982544857e+00,1.518387839700647612e+00,1.349915737444886910e+00,1.905947836954526853e+00,1.347473675274747817e+00,1.649110574612263402e+00,2.751038556164017823e+00,3.454690653276334178e+00,2.172910457152369190e+00,2.289395733464529403e+00,2.400939889372396774e+00,3.146985075914853880e+00,2.348437798381063146e+00,2.018331939356708293e+00,1.986820338283913356e+00,2.146134168097420414e+00,3.386383857038067990e+00,1.899684911585072689e+00,1.429459562043961851e+00,1.671927669056820553e+00,1.784617119666073837e+00,1.534204331033623747e+00,2.138013560431874627e+00,3.051467579867981339e+00,1.848387723699428831e+00,1.514782084530771789e+00,1.608029737982612950e+00,2.786254553147968771e+00,1.698125255943811140e+00,2.109569399745051310e+00,1.688183208203779540e+00,1.784457597916633276e+00,1.905558552625672197e+00,1.910461106855801061e+00,1.765722436684684871e+00,1.490868554043053962e+00,1.460637776161719970e+00,1.566900677133315378e+00,2.104313983002350508e+00,2.208415095103928305e+00,2.177064744736044943e+00,1.459936953405619686e+00,2.183930094012780820e+00,2.324489751545863125e+00,1.652732514112303708e+00,1.459644745333691596e+00,1.290808977860758544e+00,1.382835030092462780e+00,1.597411348000863374e+00,2.785813119984508646e+00,2.108945183149530944e+00,2.171777076742365864e+00,3.112890970369151855e+00,1.632245103118714402e+00,2.035336236073490124e+00,1.764593511874346765e+00,1.829367898210650489e+00,2.614818925480414169e+00,1.594516217799585966e+00,1.891185060350237368e+00,1.915502781358679574e+00,1.642696437502047857e+00,1.357395035641195014e+00,1.515835264413899264e+00,1.610864441926725066e+00,2.683397492720913213e+00,2.569761681396856456e+00,2.563820839185826372e+00,2.068764493617365474e+00,2.373918311903134182e+00,2.944093880299776700e+00,2.149855630624628589e+00,1.787401581056637268e+00,2.132746462945489530e+00,2.342598332680487339e+00,2.043037289088859065e+00,2.265269532057604174e+00,2.052799849015243261e+00,2.326269739755375987e+00,2.331168356876341274e+00,2.077558715985795335e+00,2.476863344416261548e+00,1.948997669920819575e+00,2.829468472175137528e+00,2.169335507789879269e+00,2.574341145585227864e+00,1.777591330499776578e+00,1.725329574306068370e+00,2.465943745642939167e+00,3.446965788295221156e+00,1.824675813041102490e+00,3.306713429725750331e+00,2.717187255770721954e+00,1.762304918204563586e+00,2.130239349296666695e+00,1.510334340022299715e+00,2.169029410888405174e+00,2.043802833866314561e+00,2.055805245400043724e+00,2.090761555849850239e+00,1.723456162972445416e+00,1.829339730101878114e+00,2.762249113668257205e+00,3.747576753206555367e+00,2.305916933692250037e+00,1.802702144919079208e+00,2.487370875298682282e+00,2.532782599384948430e+00,1.572700205047397937e+00,1.722062939231855161e+00,1.477970578015895819e+00,1.692577198212766421e+00,1.877781952740011917e+00,1.525917639303335349e+00,1.411666723726793915e+00,1.408132698537780447e+00,1.672495990732307591e+00,1.244480320254939487e+00,2.038560417419056581e+00,1.805638031541564104e+00,1.561873308306627406e+00,2.217490707041450460e+00,2.256732360286711536e+00,1.525762787223017503e+00,1.479230581133693878e+00,1.818661596353245313e+00,1.787780014326887823e+00,1.930654847886720038e+00,1.705348087188897521e+00,2.520368424018246678e+00,1.439640023301262950e+00,1.899005416541420743e+00,1.697758954642290430e+00,1.739260730312212289e+00,2.352390993517495144e+00,2.186315131317889016e+00,2.543073211351578156e+00,2.160238681239202041e+00,2.537581584094493081e+00,2.198385980376934778e+00,1.612994992337363209e+00,1.815944486198864949e+00,1.711834294865936412e+00,1.510972622442711577e+00,2.018925772674070362e+00,1.558308237399028684e+00,2.393113144150077698e+00,2.587986036445737437e+00,1.644499611668051031e+00,2.151057112944363414e+00,1.591053485613329155e+00,2.054128294381759368e+00,1.932050641355035125e+00,2.034776567161849847e+00,2.097416333211961526e+00,2.138793231748553403e+00,2.332219797415257645e+00,1.967273281858288536e+00,2.154056389563426155e+00,1.469480769265272047e+00,1.771035767141087236e+00,1.827562849078911755e+00,1.403890515177056431e+00,1.510924301600187869e+00,1.541778242060706905e+00,3.453829025707981337e+00,1.729360255160105719e+00,1.438299263207717837e+00,1.459999692361225732e+00,1.624262769566201969e+00,1.563340481193125742e+00,2.603438453712463740e+00,2.936272464500379265e+00,1.250398844102459384e+00,1.334057692074030133e+00,2.378945929286934380e+00,2.733500568876342296e+00,2.086623935699150501e+00,1.582014110739181412e+00,1.489896153170356730e+00,1.307854472131137369e+00,1.276371400706038717e+00,1.955202878205625883e+00,1.729762786874020408e+00,1.531674809840686491e+00,1.524336075991013395e+00,1.556426419631176561e+00,1.466737373040175374e+00,1.621976511882892158e+00,2.021752551429163880e+00,1.532085655908875799e+00,1.474844333377305583e+00,1.322998124446206347e+00,1.426016679288774514e+00,1.515374388344806533e+00,1.849236231733612401e+00,2.175541667213043873e+00,3.289788356627820765e+00,3.893977683500980724e+00,1.520275725094325292e+00,1.456559198343805539e+00,1.260243640731841142e+00,1.470617446315528820e+00,1.935817603176843260e+00,1.804401541063315406e+00,2.210076849240306895e+00,1.279432953107032533e+00,1.369770651747646451e+00,1.262706777600562269e+00,1.361542432110638412e+00,1.251085427596231980e+00,1.257895550253480765e+00,2.054760432132969772e+00,1.729898109255972649e+00,1.366132200105882877e+00,1.291050266994159923e+00,1.174779902309075252e+00])
    
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
        
        print('--> Weiss2018KDE:\n%.2f planets/star in [%.1f, %.1f] Rearth and [%.1f, %.1f] d' % (len(Rp)/float(Ntest), Rp_range[0], Rp_range[1], Porb_range[0], Porb_range[1]))
        
        Weight = 1./len(Rp)
        f, ax = plt.subplots(1, 2)
        ax[0].hist(Rp, bins=np.logspace(np.log10(np.min(Rp)), np.log10(np.max(Rp)), 25), weights=np.ones_like(Rp)*Weight)
        ax[0].set_xscale('log')
        ax[0].grid(axis='y')
        ax[0].set_xlabel('Planet radius [$R_\oplus$]')
        ax[0].set_ylabel('Fraction')
        ax[1].hist(Porb, bins=np.logspace(np.log10(np.min(Porb)), np.log10(np.max(Porb)), 25), weights=np.ones_like(Rp)*Weight)
        ax[1].set_xscale('log')
        ax[1].grid(axis='y')
        ax[1].set_xlabel('Planet orbital period [d]')
        ax[1].set_ylabel('Fraction')
        plt.suptitle('Weiss2018KDE')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if (FigDir is not None):
            plt.savefig(FigDir+'PlanetDistribution_Weiss2018KDE.pdf')
        plt.show(block=block)
        plt.close()
        
        pass
