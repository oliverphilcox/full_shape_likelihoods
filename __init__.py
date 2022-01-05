import os
import numpy as np
from montepython.likelihood_class import Likelihood_prior
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.integrate import quad
import scipy.integrate as integrate
from numpy import log, exp, sin, cos
from scipy.special.orthogonal import p_roots

class datasets(object):
        def __init__(self, options):
                """Load Pk, Q0 and Bk data from file, as well as covariance matrix. The `options' argument is a dictionary of options specifying file names etc."""
                
                # Load datasets
                if options.use_P:
                        self.load_power_spectrum(options)
                if options.use_B:
                        self.load_bispectrum(options)
                if options.use_AP:
                        self.load_AP(options)

                # Load covariance
                self.load_covariance(options)

        def load_power_spectrum(self, options):
                """Load power spectrum multipole dataset, optionally including Q0"""
                
                # Load raw Pk measurements
                k_init,Pk0_init,Pk2_init,Pk4_init=np.loadtxt(os.path.join(options.data_directory, options.P_measurements), skiprows = 0, unpack=True)

                # Count number of P bins (nP) and Q bins (nQ)
                self.nP_init = len(k_init)
                self.nPQ = np.sum((k_init<options.kmaxQ)&(k_init>=options.kminP)) 
                self.nQ = np.sum((k_init<options.kmaxQ)&(k_init>=options.kmaxP))
                self.nP = self.nPQ - self.nQ
                self.omit = np.sum((k_init<options.kminP)) # bins to omit at start of Pk array              
                self.omit2 = self.nP + self.omit # bins to omit at start of Q0 array

                self.kP = k_init[self.omit:self.omit+self.nPQ]
                Pk0 = Pk0_init[self.omit:self.omit+self.nPQ]
                Pk2 = Pk2_init[self.omit:self.omit+self.nPQ]
                Pk4 = Pk4_init[self.omit:self.omit+self.nPQ]

                # Define data vectors
                self.P0 = Pk0[:self.nP]
                self.P2 = Pk2[:self.nP]
                self.P4 = Pk4[:self.nP]

                # Compute Q0 from Pk0 measurements
                if options.use_Q:
                        self.Q0 = Pk0[self.nP:]-1./2.*Pk2[self.nP:]+3./8.*Pk4[self.nP:]

        def load_bispectrum(self, options):
                """Load bispectrum dataset."""
                
                # Load discreteness weights from file
                self.discreteness_weights = np.loadtxt(os.path.join(options.data_directory, options.discreteness_weights_file), dtype=np.float64)
        
                # Load bispectrum measurements
                khere, khere2, khere3, self.B0, _ = np.loadtxt(os.path.join(options.data_directory, options.B_measurements), dtype=np.float64, unpack=True)
                assert len(self.B0)==len(self.discreteness_weights), "Number of bispectra bins must match number of weights!"
                self.nB = len(self.B0)

                # 1D triangle centers
                self.kB = np.linspace(options.kminB,options.kmaxB,options.ksizeB)
                self.dkB = self.kB[1]-self.kB[0]
                
                # Indices labelling bispectrum bins
                self.new_triag = [np.asarray(kk/self.dkB,dtype=int)-int(options.kminB/self.dkB) for kk in [khere, khere2, khere3]]
        
        def load_AP(self, options):
                """Load Alcock-Paczynski dataset."""
                
                self.alphas = np.loadtxt(os.path.join(options.data_directory, options.AP_measurements))

        def load_covariance(self, options):
                """Load in the covariance matrix, filtered to the required bins and datasets [with the ordering P0, P2, P4, Q0, B0, AP]."""
                
                # Load full covariance matrix
                cov1 = np.loadtxt(os.path.join(options.data_directory, options.covmat_file),dtype=np.float64)

                # Define which bins we use
                filt = []
                if options.use_P:
                        filt.append(np.arange(self.omit,self.omit+self.nP)) # P0
                        filt.append(np.arange(self.omit+self.nP_init,self.omit+self.nP_init+self.nP)) # P2
                        filt.append(np.arange(self.omit+2*self.nP_init,self.omit+2*self.nP_init+self.nP)) # P4
                if options.use_Q:
                        filt.append(np.arange(self.omit2+3*self.nP_init,self.omit2+3*self.nP_init+self.nQ)) # Q0
                if options.use_B:
                        filt.append(np.arange(4*self.nP_init,4*self.nP_init+self.nB)) # B0
                if options.use_AP:
                        filt.append([-2,-1])
                filt= np.concatenate(filt)
                
                # Filter to the correct bins we want
                self.cov = np.zeros((len(filt),len(filt)),dtype='float64')
                for i,I in enumerate(filt):
                        for j,J in enumerate(filt):
                                self.cov[i,j] = cov1[I,J]

                # Compute matrix determinant for later use
                self.logdetcov = np.linalg.slogdet(self.cov)[1]

class full_shape_spectra(Likelihood_prior):

        # initialisation of the class is done within the parent Likelihood_prior. For
        # this case, it does not differ, actually, from the __init__ method in
        # Likelihood class.

        def __init__(self,path,data,command_line):
                """Initialize the full shape likelihood. This loads the data-set and pre-computes a number of useful quantities."""

                Likelihood_prior.__init__(self,path,data,command_line)

                # First load the data
                self.dataset = datasets(self)

                assert self.use_AP and self.use_P and self.use_Q and self.use_B, "need to change boss_n1 to use not all of AP, P, Q, B currently!"
                        
                # Pre-load useful quantities for bispectra
                if self.use_B:

                        # Define angular grids for bispectrum integration
                        self.n_gauss, self.n_gauss2 = 3, 8
                        [gauss_mu,self.gauss_w], [gauss_mu2,self.gauss_w2] = p_roots(self.n_gauss), p_roots(self.n_gauss2)

                        self.mesh_mu = np.meshgrid(gauss_mu,gauss_mu,gauss_mu,gauss_mu2,gauss_mu2, sparse=True, indexing='ij')

                        # Selection of functions dumped from Mathematica
                        self.D1 = lambda k1,k2,k3,beta: (15. + 10.*beta+beta**2. + 2.*beta**2.*((k3**2.-k1**2.-k2**2.)/(2.*k1*k2))**2.)/15.
                        self.D2 = lambda k1,k2,k3,beta: beta/3+(4 *beta**2.)/15-(k1**2. *beta**2.)/(15 *k2**2.)-(k2**2. *beta**2.)/(15 *k1**2.)-(k1**2. *beta**2.)/(30 *k3**2.)+(k1**4 *beta**2.)/(30 *k2**2. *k3**2.)-(k2**2. *beta**2.)/(30 *k3**2.)+(k2**4. *beta**2.)/(30 *k1**2. *k3**2.)+(k3**2. *beta**2.)/(30 *k1**2.)+(k3**2. *beta**2.)/(30 *k2**2.)+(2 *beta**3)/35-(k1**2. *beta**3.)/(70 *k2**2.)-(k2**2. *beta**3)/(70 *k1**2.)-(k1**2. *beta**3)/(70 *k3**2.)+(k1**4 *beta**3)/(70 *k2**2.*k3**2.)-(k2**2. *beta**3)/(70 *k3**2.)+(k2**4 *beta**3)/(70 *k1**2. *k3**2.)-(k3**2. *beta**3)/(70 *k1**2.)-(k3**2. *beta**3)/(70 *k2**2.)+(k3**4 *beta**3)/(70 *k1**2. *k2**2.)
                        self.D3 = lambda k1,k2,k3,beta: beta/6-(k1**2. *beta)/(12 *k2**2.)-(k2**2. *beta)/(12 *k1**2.)+(k3**2. *beta)/(12 *k1**2.)+(k3**2. *beta)/(12 *k2**2.)+ beta**2./6-(k1**2. *beta**2.)/(12 *k2**2.)-(k2**2. *beta**2.)/(12 *k1**2.)+(k3**2. *beta**2.)/(60 *k1**2.)+(k3**2. *beta**2.)/(60 *k2**2.)+(k3**4. *beta**2.)/(15 *k1**2. *k2**2.)+(2 *beta**3.)/35-(k1**4. *beta**3.)/(140 *k2**4.)-(3 *k1**2. *beta**3.)/(140 *k2**2.)-(3 *k2**2. *beta**3.)/(140 *k1**2.)-(k2**4. *beta**3.)/(140 *k1**4.)-(k3**2. *beta**3.)/(35 *k1**2.)+(3 *k1**2. *k3**2. *beta**3.)/(140 *k2**4.)-(k3**2. *beta**3.)/(35 *k2**2.)+(3 *k2**2. *k3**2. *beta**3.)/(140 *k1**4.)-(3 *k3**4. *beta**3.)/(140 *k1**4.)-(3 *k3**4. *beta**3.)/(140 *k2**4.)+(3 *k3**4. *beta**3.)/(70 *k1**2. *k2**2.)+(k3**6 *beta**3.)/(140 *k1**2. *k2**4.)+(k3**6 *beta**3.)/(140 *k1**4. *k2**2.)+ beta**4./105-(k1**4. *beta**4.)/(420 *k2**4.)-(k1**2. *beta**4.)/(420 *k2**2.)-(k2**2. *beta**4.)/(420 *k1**2.)-(k2**4. *beta**4.)/(420 *k1**4.)-(k3**2. *beta**4.)/(105 *k1**2.)+(k1**2. *k3**2. *beta**4.)/(180 *k2**4.)-(k3**2. *beta**4.)/(105 *k2**2.)+(k2**2. *k3**2. *beta**4.)/(180 *k1**4.)-(k3**4. *beta**4.)/(420 *k1**4.)-(k3**4. *beta**4.)/(420 *k2**4.)+(k3**4. *beta**4.)/(70 *k1**2. *k2**2.)-(k3**6 *beta**4.)/(420 *k1**2. *k2**4.)-(k3**6 *beta**4.)/(420 *k1**4. *k2**2.)+(k3**8 *beta**4.)/(630 *k1**4. *k2**4.)
                        self.F2 = lambda k1,k2,k3,beta,b1,b2,bG2: (b1*(-5.*(k1**2.-k2**2.)**2.+3.*(k1**2.+k2**2.)*k3**2.+2.*k3**4.)*self.D1(k1,k2,k3,beta) + b1*(-3.*(k1**2.-k2**2.)**2.-1.*(k1**2.+k2**2.)*k3**2.+4.*k3**4.)*self.D2(k1,k2,k3,beta) + 7.*self.D1(k1,k2,k3,beta)*(2.*b2*k1**2.*k2**2. + bG2*(k1-k2-k3)*(k1+k2-k3)*(k1-k2+k3)*(k1+k2+k3)))*b1**2./28./k1**2./k2**2. + b1**4.*self.D3(k1,k2,k3,beta)
                        self.F2real = lambda k1,k2,k3,b1,b2,bG2: (b1*(-5.*(k1**2.-k2**2.)**2.+3.*(k1**2.+k2**2.)*k3**2.+2.*k3**4.) + 7.*(2.*b2*k1**2.*k2**2. + bG2*(k1-k2-k3)*(k1+k2-k3)*(k1-k2+k3)*(k1+k2+k3)))*b1**2./28./k1**2./k2**2.
                        self.G2 = lambda k1,k2,k3: -((3*(k1**2-k2**2)**2+(k1**2+k2**2)*k3**2-4*k3**4)/(28 *k1**2 *k2**2))
                        self.Bbinl0 = lambda I01,I02,I21,I22,I41,I42,I61,I62,Im21,Im22,Im41,Im42,k32,k34,k36,k38,k3m2,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19,e20,e21: 2.*(e1*I22*k3m2*I01 + e2*I61*Im42*k3m2 + e3*k32*Im22*I01 + e4*k34*Im42*I01 + e5*I01*I02 + e6*I41*Im22*k3m2 + e7*I41*Im42+ e8*I21*I02*k3m2 + e9*I21*Im22 + e10*I21*k32*Im42 + Im41*(e11*I62*k3m2 + e12*k34*I02 + e13*k36*Im22 + e14*I42 + e15*k38*Im42 + e16*I22*k32) + Im21*(e17*I42*k3m2 +e18*k32*I02 + e19*I22 + e20*k36*Im42 +e21*k34*Im22) )
                        self.Bbin = lambda I01,I02,I21,I22,Im21,Im22,k32,k34,c1,c2,c3,c4,b1: 2.*(b1**2.)*(c1*I01*I02 + c2*(I21*Im22+I22*Im21)+c3*(I02*Im21+I01*Im22)*k32 +c4*(Im21*Im22)*k34)
                        self.j2 = lambda x: (3./x**2.-1.)*np.sin(x)/x - 3.*np.cos(x)/x**2.

        def loglkl(self, cosmo, data):
                """Compute the log-likelihood for a given set of cosmological and nuisance parameters. Note that this marginalizes over nuisance parameters that enter the model linearly."""

                # Load parameters
                h = cosmo.h()
                #norm = (data.mcmc_parameters['norm']['current'] * data.mcmc_parameters['norm']['scale'])
                norm = 1.
                c1 = 0.

                i_s=repr(3)
                b1 = (data.mcmc_parameters['b^{('+i_s+')}_1']['current'] *
                data.mcmc_parameters['b^{('+i_s+')}_1']['scale'])
                b2 = (data.mcmc_parameters['b^{('+i_s+')}_2']['current'] *
                data.mcmc_parameters['b^{('+i_s+')}_2']['scale'])
                bG2 = (data.mcmc_parameters['b^{('+i_s+')}_{G_2}']['current'] *
                data.mcmc_parameters['b^{('+i_s+')}_{G_2}']['scale'])
                fNL = (data.mcmc_parameters['f_{NL}']['current'] *
                        data.mcmc_parameters['f_{NL}']['scale'])

                print("FIXING B1 FOR TESTING")
                print(b1)
                b1 = 1.8682
                print(b1)

                dk2 = 0.005
                
                ## Define parameter mean and variances        
                # Means
                psh = 3500.
                bGamma3 = 23.*(b1-1.)/42.
                Pshot = 0.
                Bshot = 1.
                a0 = 0.
                a2 = 0.
                css4 = 0.
                css2 = 30.
                css0 = 0.
                b4 = 500.*1.
                # Standard deviations
                sigbGamma3 = 0.
                sigPshot = 1.*psh
                sigBshot = 1.*psh
                sigc1 = 5.
                siga0 = psh*1.
                sigcs0 = 30.
                sigcs2 = 30.
                sigcs4 = 30.
                sigb4 = 500.
                siga2 = psh*1.

                # Define local variables 
                dataset = self.dataset
                nP, nQ, nPQ, nB = dataset.nP, dataset.nQ, dataset.nPQ, dataset.nB
                kB, dkB = dataset.kB, dataset.dkB
                
                # Compute useful quantities for AP parameters
                z = self.z
                fz = cosmo.scale_independent_growth_factor_f(z)
                DA_th = cosmo.angular_distance(z)
                rs_th = cosmo.rs_drag()
                Hz_th = cosmo.Hubble(z)

                # Define k binning, depending on whether we include bin integration
                if self.bin_integration_Pk:
                        kint = np.linspace(log(1.e-4),log(max(dataset.kP)+0.01),100)
                        kint = np.exp(kint)
                else:
                        kint = dataset.kP

                # Create output arrays
                theory_minus_data = np.zeros(3*nP+nB+nQ+2)
                cov_bG3, cov_Pshot, cov_Bshot, cov_c1, cov_a0, cov_a2, cov_cs0, cov_cs2, cov_cs4, cov_b4 = [np.zeros(3*nP+nB+nQ+2) for _ in range(10)]

                # Run CLASS-PT
                all_theory = cosmo.get_pk_mult(kint*h,z,len(kint))

                #### Pk
                if self.use_P:

                        def bin_integrator(input_table):
                                """If bin-integration is included, integrate the function defined in `inttab' over the binned k-space. Else, return the input table."""
                                if self.bin_integration_Pk:
                                        f_int = interpolate.InterpolatedUnivariateSpline(kint,input_table,ext=3)
                                        integrand = lambda k: exp(3.*k)*f_int(exp(k))
                                        out = np.zeros(nPQ)
                                        for i in range(nPQ):
                                                out[i] = integrate.quad(integrand, log(dk2*i+self.kminP), log(dk2*(i+1)+self.kminP))[0]*3./((dk2*(i+1)+self.kminP)**3.-(dk2*i+self.kminP)**3.)
                                        return out
                                else:
                                        return input_table

                        P0tab = ((norm**2.*all_theory[15] +norm**4.*(all_theory[21])+ norm**1.*b1*all_theory[16] +norm**3.*b1*(all_theory[22]) + norm**0.*b1**2.*all_theory[17] +norm**2.*b1**2.*all_theory[23] + 0.25*norm**2.*b2**2.*all_theory[1] +b1*b2*norm**2.*all_theory[30]+ b2*norm**3.*all_theory[31] + b1*bG2*norm**2.*all_theory[32]+ bG2*norm**3.*all_theory[33] + b2*bG2*norm**2.*all_theory[4]+ bG2**2.*norm**2.*all_theory[5] + 2.*css0*norm**2.*all_theory[11]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**2.*(b1*all_theory[7]+norm*all_theory[8]))*h**3. + (psh)*Pshot + a0*(10**4)*(kint/0.5)**2.  + fz**2.*b4*kint**2.*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)*all_theory[13]*h + a2*(1./3.)*(10.**4.)*(kint/0.45)**2.)
                        
                        ## Compute P0, P2, P4 multipoles, integrating with respect to bins
                        P0 = bin_integrator((norm**2.*all_theory[15] +norm**4.*(all_theory[21])+ norm**1.*b1*all_theory[16] +norm**3.*b1*(all_theory[22]) + norm**0.*b1**2.*all_theory[17] +norm**2.*b1**2.*all_theory[23] + 0.25*norm**2.*b2**2.*all_theory[1] +b1*b2*norm**2.*all_theory[30]+ b2*norm**3.*all_theory[31] + b1*bG2*norm**2.*all_theory[32]+ bG2*norm**3.*all_theory[33] + b2*bG2*norm**2.*all_theory[4]+ bG2**2.*norm**2.*all_theory[5] + 2.*css0*norm**2.*all_theory[11]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**2.*(b1*all_theory[7]+norm*all_theory[8]))*h**3. + (psh)*Pshot + a0*(10**4)*(kint/0.5)**2.  + fz**2.*b4*kint**2.*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)*all_theory[13]*h + a2*(1./3.)*(10.**4.)*(kint/0.45)**2.)
                        P2 = bin_integrator((norm**2.*all_theory[18] +  norm**4.*(all_theory[24])+ norm**1.*b1*all_theory[19] +norm**3.*b1*(all_theory[25]) + b1**2.*norm**2.*all_theory[26] +b1*b2*norm**2.*all_theory[34]+ b2*norm**3.*all_theory[35] + b1*bG2*norm**2.*all_theory[36]+ bG2*norm**3.*all_theory[37]  + 2.*css2*norm**2.*all_theory[12]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**3.*all_theory[9])*h**3. + fz**2.*b4*kint**2.*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)*all_theory[13]*h + a2*(10.**4.)*(2./3.)*(kint/0.45)**2.)
                        P4 = bin_integrator((norm**2.*all_theory[20] + norm**4.*all_theory[27]+ b1*norm**3.*all_theory[28] + b1**2.*norm**2.*all_theory[29] + b2*norm**3.*all_theory[38] + bG2*norm**3.*all_theory[39]  +2.*css4*norm**2.*all_theory[13]/h**2.)*h**3. + fz**2.*b4*kint**2.*(norm**2.*fz**2.*210./143. + 30.*fz*b1*norm/11.+b1**2.)*all_theory[13]*h)

                        ## Compute derivatives with respect to parameters, integrating with respect to bins
                        deriv0_bG3 = bin_integrator((0.8*sigbGamma3*norm)*norm**2.*(b1*all_theory[7]+norm*all_theory[8])*h**3.)
                        deriv2_bG3 = bin_integrator((0.8*sigbGamma3*norm)*norm**3.*all_theory[9]*h**3.)
                        deriv_cs0 = bin_integrator(2.*norm**2.*all_theory[11]*h**1.)
                        deriv_cs2 = bin_integrator(2.*norm**2.*all_theory[12]*h**1.)
                        deriv_cs4 = bin_integrator(2.*norm**2.*all_theory[13]*h**1.)
                        deriv_b4 = bin_integrator(fz**2.*kint**2.*all_theory[13]*h)

                        ## Add to (theory - data)
                        theory_minus_data[:nP] = P0[:nP] - dataset.P0
                        theory_minus_data[nP:2*nP] = P2[:nP] - dataset.P2
                        theory_minus_data[2*nP:3*nP] = P4[:nP] - dataset.P4

                        ## Add to covariance matrix
                        # P0
                        cov_bG3[:nP] = deriv0_bG3[:nP]
                        cov_cs0[:nP] = deriv_cs0[:nP]
                        cov_b4[:nP] = deriv_b4[:nP]*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)
                        cov_Pshot[:nP] = 1.
                        cov_a0[:nP] = (dataset.kP[:nP]/0.45)**2.
                        cov_a2[:nP] = (1./3.)*(dataset.kP[:nP]/0.45)**2.
                        
                        # P2
                        cov_bG3[nP:2*nP] = deriv2_bG3[:nP]
                        cov_cs2[nP:2*nP] = deriv_cs2[:nP]
                        cov_b4[nP:2*nP] = deriv_b4[:nP]*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)
                        cov_a2[nP:2*nP] = (2./3.)*(dataset.kP[:nP]/0.45)**2.
                        
                        # P4
                        cov_cs4[2*nP:3*nP] = deriv_cs4[:nP]
                        cov_b4[2*nP:3*nP] = deriv_b4[:nP]*(norm**2.*fz**2.*210./143. + 30.*fz*b1*norm/11.+b1**2.)
                        
                #### Q0
                if self.use_Q:
                        ## Compute from Pk
                        Q0 = P0[nP:]-P2[nP:]/2.+3.*P4[nP:]/8.
                        theory_minus_data[3*nP:3*nP+nQ] = Q0 - dataset.Q0

                        ## Add to covariance matrix
                        cov_bG3[3*nP:3*nP+nQ] = deriv0_bG3[nP:] - 1./2.*deriv2_bG3[nP:]
                        cov_cs0[3*nP:3*nP+nQ] = deriv_cs0[nP:]
                        cov_cs2[3*nP:3*nP+nQ] = -1./2.*deriv_cs2[nP:]
                        cov_cs4[3*nP:3*nP+nQ] = 3./8.*deriv_cs4[nP:]
                        cov_b4[3*nP:3*nP+nQ] = deriv_b4[nP:]*((norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.) - ((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)/2. +3.*(norm**2.*fz**2.*210./143. + 30.*fz*b1*norm/11.+b1**2.)/8.)
                        cov_Pshot[3*nP:3*nP+nQ] = 1.
                        cov_a0[3*nP:3*nP+nQ] = (dataset.kP[nP:]/0.45)**2.

                #### AP
                if self.use_AP:  
                        # AP definitions  
                        A_par = self.rdHfid/(rs_th*Hz_th)
                        A_perp = self.rdDAfid/(rs_th/DA_th)
                        theory_minus_data[-2] = A_par - dataset.alphas[0]
                        theory_minus_data[-1] = A_perp - dataset.alphas[1]

                #### Bispectrum
                if self.use_B:

                        ### MESSY BELOW HERE!!
                        Ashot = 0.
                        c0 = 0.
                        c2 = 0.
                        beta = fz/b1
                        a0 = 1. + 2.*beta/3. + beta**2./5.
                        Plintab = -1.*norm**2.*(all_theory[10]/h**2./kint**2)*h**3
                        P2 = norm**2.*(all_theory[14])*h**3.

                        ng = (1.+Ashot)/psh

                        # IR resummation parameters
                        r_bao = cosmo.rs_drag()*h
                        ks_IR = 0.05

                        P0int = interpolate.InterpolatedUnivariateSpline(kint,Plintab,ext=3)
                        Sigma = integrate.quad(lambda k: (4*np.pi)*exp(1.*k)*P0int(exp(k))*(1.-3*(2*r_bao*exp(k)*cos(exp(k)*r_bao)+(-2+r_bao**2*exp(k)**2)*sin(r_bao*exp(k)))/(exp(k)*r_bao)**3)/(3*(2*np.pi)**3.), log(2.e-4), log(0.2))[0]
                        
                        # Wiggly power spectrum
                        Pw = (Plintab-P2)/(np.exp(-kint**2.*Sigma)-np.exp(-kint**2.*Sigma)*(1+kint**2.*Sigma))
                        Pwfunc = interpolate.InterpolatedUnivariateSpline(kint,Pw,ext=3)
                        # Non-Wiggly power spectrum
                        Pnw = Plintab - Pw*np.exp(-kint**2.*Sigma)
                        Pnwfunc = interpolate.InterpolatedUnivariateSpline(kint,Pnw,ext=3)

                        Sigma2 = integrate.quad(lambda k: (4*np.pi)*exp(1.*k)*P0int(exp(k))*(1.-3*(2*r_bao*exp(k)*cos(exp(k)*r_bao)+(-2+r_bao**2*exp(k)**2)*sin(r_bao*exp(k)))/(exp(k)*r_bao)**3)/(3*(2*np.pi)**3.), log(2.e-4), log(ks_IR))[0]
                        deltaSigma2 = integrate.quad(lambda k: (4*np.pi)*exp(1.*k)*P0int(exp(k))*(self.j2(exp(k)*r_bao))/((2*np.pi)**3.), log(2.e-4), log(ks_IR))[0]
                        
                        # IR resummed spectra
                        P_IR = lambda k, mu: Pnwfunc(k) +  np.exp(-k**2.*(Sigma2*(1.+2.*fz*mu**2.*(2.+fz)) + deltaSigma2*mu**2.*fz**2.*(mu**2.-1.)))*Pwfunc(k) -(c0+c1*mu**2.+c2*mu**4.)*(k/0.3)**2.*P0int(k)/(b1+fz*mu**2.)
                        P_IRC = lambda k, mu:Pnwfunc(k) +  np.exp(-k**2.*(Sigma2*(1.+2.*fz*mu**2.*(2.+fz)) + deltaSigma2*mu**2.*fz**2.*(mu**2.-1.)))*Pwfunc(k) -(mu**2.)*(k/0.3)**2.*P0int(k)/(b1+fz*mu**2.)

                        kmsMpc = 3.33564095198145e-6 # conversion factor

                        # Define coordinate rescaling parameters
                        DA=DA_th/(self.h_fid/h)
                        Hz=Hz_th*(self.h_fid/h)/kmsMpc
                        apar=self.Hz_fid/Hz
                        aperp=DA/self.DA_fid

                        B0 = np.zeros(nB)
                        new_triag = dataset.new_triag

                        Azeta = cosmo.A_s()*2.*np.pi**2.

                        Tfunc = lambda k: (P0int(k)/(Azeta*((k/0.05)**(cosmo.n_s()-1.))/k**3.))**0.5
                        BNG = lambda k1, k2, k3: Azeta**2.*(Tfunc(k1)*Tfunc(k2)*Tfunc(k3)*(18./5.)*(-1./k1**3./k2**3.-1./k3**3./k2**3.-1./k1**3./k3**3.-2./k1**2./k2**2./k3**2.+1/k1/k2**2./k3**3.+1/k1/k3**2./k2**3.+1/k2/k3**2./k1**3.+1/k2/k1**2./k3**3.+1/k3/k1**2./k2**3.+1/k3/k2**2./k1**3.))

                        def Bk_matrices(k1,k2,k3,mu1,phi,kc1=0,kc2=0,kc3=0,apar=1,aperp=1):
                                ddk1 = dk1/2.
                                ddk2 = dk2/2.
                                ddk3 = dk3/2.
                                kk1 = (kc1+k1*ddk1)
                                kk2 = (kc2+k2*ddk2)
                                kk3 = (kc3+k3*ddk3)
                                xxfunc = (kk3**2.-kk1**2.-kk2**2.)/(2.*kk1*kk2)
                                yyfunc = np.sqrt(np.abs(1.-xxfunc**2.))
                                mu2 = xxfunc*mu1 - np.sqrt(1.-mu1**2.)*yyfunc*np.cos(phi*2.*np.pi)
                                mu3 = -(kk2/kk3)*mu2-(kk1/kk3)*mu1

                                # Coordinate distortion on mu
                                nnu = lambda mu: mu/apar/(np.sqrt(np.abs(mu**2./apar**2. + (1-mu**2.)/aperp**2.)))
                                nnu1, nnu2, nnu3 = nnu(mu1), nnu(mu2), nnu(mu3)

                                # Coordinate distortion on length
                                qq = lambda mu: np.sqrt(np.abs(mu**2/apar**2 + (1.-mu**2)/aperp**2))
                                qq1, qq2, qq3 = qq(mu1), qq(mu2), qq(mu3)

                                # IR resummed spectra
                                PP_IR1, PP_IR2, PP_IR3 = P_IR(kk1*qq1,nnu1), P_IR(kk2*qq2,nnu2), P_IR(kk3*qq3,nnu3)
                                PP_IR1C, PP_IR2C, PP_IR3C = P_IRC(kk1*qq1,nnu1), P_IRC(kk2*qq2,nnu2), P_IRC(kk3*qq3,nnu3)

                                ### Bfunc3
                                zz21 = self.F2real(kk1*qq1,kk2*qq2,kk3*qq3,b1,b2,bG2)+b1**3.*beta*((nnu2*kk2*qq2+nnu1*kk1*qq1)/kk3/qq3)**2.*self.G2(kk1*qq1,kk2*qq2,kk3*qq3)+(b1**4.*beta/2.)*(nnu2*kk2*qq2+nnu1*kk1*qq1)*(nnu1*(1.+beta*nnu2**2.)/kk1/qq1 + nnu2*(1.+beta*nnu1**2.)/kk2/qq2)
                                zz22 = self.F2real(kk1*qq1,kk3*qq3,kk2*qq2,b1,b2,bG2)+b1**3.*beta*((nnu3*kk3*qq3+nnu1*kk1*qq1)/kk2/qq2)**2.*self.G2(kk1*qq1,kk3*qq3,kk2*qq2)+(b1**4.*beta/2.)*(nnu3*kk3*qq3+nnu1*kk1*qq1)*(nnu1*(1.+beta*nnu3**2.)/kk1/qq1 + nnu3*(1.+beta*nnu1**2.)/kk3/qq3)
                                zz23 = self.F2real(kk2*qq2,kk3*qq3,kk1*qq1,b1,b2,bG2)+b1**3.*beta*((nnu2*kk2*qq2+nnu3*kk3*qq3)/kk1/qq1)**2.*self.G2(kk2*qq2,kk3*qq3,kk1*qq1)+(b1**4.*beta/2.)*(nnu2*kk2*qq2+nnu3*kk3*qq3)*(nnu2*(1.+beta*nnu3**2.)/kk2/qq2 + nnu3*(1.+beta*nnu2**2.)/kk3/qq3)
                                
                                FF2func1 = zz21*(1+beta*nnu1**2)*(1.+beta*nnu2**2.)*PP_IR1*kk1*ddk1*PP_IR2*kk2*ddk2*kk3*ddk3 + 1.*0.5*(Bshot/ng)*b1**2.*PP_IR1*kk1*(1.+beta*nnu1**2.*(Bshot+2.*(1.+Pshot))/Bshot + beta**2.*nnu1**4.*2.*(1.+Pshot)/Bshot)*kk2*kk3*ddk1*ddk2*ddk3 + ((1.+Pshot)/ng)**2.*kk1*kk2*kk3*ddk1*ddk2*ddk3/2.
                                FF2func2 = zz22*(1+beta*nnu1**2)*(1.+beta*nnu3**2.)*PP_IR1*kk1*ddk1*PP_IR3*kk3*ddk3*kk2*ddk2 + 1.*0.5*(Bshot/ng)*b1**2.*PP_IR2*kk2*(1.+beta*nnu2**2.*(Bshot+2.+2.*Pshot)/Bshot + beta**2.*nnu2**4.*2.*(1.+Pshot)/Bshot)*kk1*kk3*ddk1*ddk2*ddk3 + 0.*(1/ng)**2.*kk1*kk2*kk3*ddk1*ddk2*ddk3/6.
                                FF2func3 = zz23*(1+beta*nnu2**2)*(1.+beta*nnu3**2.)*PP_IR2*kk2*ddk2*PP_IR3*kk3*ddk3*kk1*ddk1 + 1.*0.5*(Bshot/ng)*b1**2.*PP_IR3*kk3*(1.+beta*nnu3**2.*(Bshot+2.+2.*Pshot)/Bshot + beta**2.*nnu3**4.*2.*(1.+Pshot)/Bshot)*kk2*kk1*ddk1*ddk2*ddk3 + 0.*(1/ng)**2.*kk1*kk2*kk3*ddk1*ddk2*ddk3/6.
                                
                                FF2func1C = zz21*(1+beta*nnu1**2)*(1.+beta*nnu2**2.)*PP_IR1C*kk1*ddk1*PP_IR2C*kk2*ddk2*kk3*ddk3 + 1.*0.5*(Bshot/ng)*b1**2.*PP_IR1C*kk1*(1.+beta*nnu1**2.*(Bshot+2.*(1.+Pshot))/Bshot + beta**2.*nnu1**4.*2.*(1.+Pshot)/Bshot)*kk2*kk3*ddk1*ddk2*ddk3 + ((1.+Pshot)/ng)**2.*kk1*kk2*kk3*ddk1*ddk2*ddk3/2.
                                FF2func2C = zz22*(1+beta*nnu1**2)*(1.+beta*nnu3**2.)*PP_IR1C*kk1*ddk1*PP_IR3C*kk3*ddk3*kk2*ddk2 + 1.*0.5*(Bshot/ng)*b1**2.*PP_IR2C*kk2*(1.+beta*nnu2**2.*(Bshot+2.+2.*Pshot)/Bshot + beta**2.*nnu2**4.*2.*(1.+Pshot)/Bshot)*kk1*kk3*ddk1*ddk2*ddk3 + 0.*(1/ng)**2.*kk1*kk2*kk3*ddk1*ddk2*ddk3/6.
                                FF2func3C = zz23*(1+beta*nnu2**2)*(1.+beta*nnu3**2.)*PP_IR2C*kk2*ddk2*PP_IR3C*kk3*ddk3*kk1*ddk1 + 1.*0.5*(Bshot/ng)*b1**2.*PP_IR3C*kk3*(1.+beta*nnu3**2.*(Bshot+2.+2.*Pshot)/Bshot + beta**2.*nnu3**4.*2.*(1.+Pshot)/Bshot)*kk2*kk1*ddk1*ddk2*ddk3 + 0.*(1/ng)**2.*kk1*kk2*kk3*ddk1*ddk2*ddk3/6.

                                FFnlfunc = fNL*BNG(kk1*qq1,kk2*qq2,kk3*qq3)*b1**3.*(1+beta*nnu1**2)*(1.+beta*nnu3**2.)*(1+beta*nnu2**2)*kk1*kk2*kk3*ddk1*ddk2*ddk3

                                Bfunc3 = (2.*FF2func1 + 2.*FF2func2 + 2.*FF2func3 + FFnlfunc)/apar**2./aperp**4.
                                
                                EBfunc = b1**2.*(((1.+beta*nnu1**2.)*PP_IR1+PP_IR2*(1.+beta*nnu2**2.)+ PP_IR3*(1.+beta*nnu3**2.))*kk1*kk2*kk3*ddk1*ddk2*ddk3)/apar**2./aperp**4.

                                EPfunc = (b1*(2.*beta*nnu1**2.*(1.+beta*nnu1**2.)*PP_IR1+PP_IR2*(beta*nnu2**2.*2.)*(1.+beta*nnu2**2.)+ PP_IR3*(2.*beta*nnu3**2.)*(1.+beta*nnu3**2.) + 2.*psh)*kk1*kk2*kk3*ddk1*ddk2*ddk3)/apar**2./aperp**4.

                                Bfunc4 = (2.*FF2func1C + 2.*FF2func2C + 2.*FF2func3C - 2.*FF2func1 - 2.*FF2func2 - 2.*FF2func3)/apar**2./aperp**4.
                                
                                return Bfunc3, EBfunc, EPfunc, Bfunc4

                        for j in range(int(nB)):
                                kc1 = kB[new_triag[0][j]]
                                kc2 = kB[new_triag[1][j]]
                                kc3 = kB[new_triag[2][j]]
                                dk1 = dkB
                                dk2 = dkB
                                dk3 = dkB
                                if (kB[new_triag[0][j]]<dkB):
                                        kc1 = 0.0058
                                        dk1  = 0.0084
                                if (kB[new_triag[1][j]]<dkB):
                                        kc2 = 0.0058
                                        dk2  = 0.0084
                                if (kB[new_triag[2][j]]<dkB):
                                        kc3 = 0.0058
                                        dk3  = 0.0084

                                # Idealized bin volume
                                Nk123 = ((kc1+dk1/2.)**2. - (kc1-dk1/2.)**2.)*((kc2+dk2/2.)**2. - (kc2-dk2/2.)**2.)*((kc3+dk3/2.)**2. - (kc3-dk3/2.)**2.)/8.
                                
                                # Compute matrices
                                mat4, mat5, mat6, mat7 = Bk_matrices(*self.mesh_mu,kc1=kc1,kc2=kc2,kc3=kc3,apar=apar,aperp=aperp)
                                
                                # Sum over angles to compute B0
                                B0[j] = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(mat4,self.gauss_w2)/2.,self.gauss_w2)/2.,self.gauss_w),self.gauss_w),self.gauss_w)/Nk123

                                # Add to output array
                                theory_minus_data[3*nP + nQ + j] = B0[j]*dataset.discreteness_weights[j] - dataset.B0[j]
                                
                                # Update nuisance parameter covariance
                                derivB_Pshot = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(mat6,self.gauss_w2)/2.,self.gauss_w2)/2.,self.gauss_w),self.gauss_w),self.gauss_w)/Nk123
                                derivB_Bshot = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(mat5,self.gauss_w2)/2.,self.gauss_w2)/2.,self.gauss_w),self.gauss_w),self.gauss_w)/Nk123
                                derivB_c1 = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(mat7,self.gauss_w2)/2.,self.gauss_w2)/2.,self.gauss_w),self.gauss_w),self.gauss_w)/Nk123

                                cov_Pshot[3*nP + nQ + j] = derivB_Pshot
                                cov_Bshot[3*nP + nQ + j] = derivB_Bshot		
                                cov_c1[3*nP + nQ + j] = derivB_c1

                ### COMBINE AND COMPUTE LIKELIHOOD

                # Assemble full covariance including nuisance parameter marginalizations
                marg_cov = dataset.cov + np.outer(cov_bG3,cov_bG3) + sigPshot**2.*np.outer(cov_Pshot,cov_Pshot) + siga0**2.*np.outer(cov_a0,cov_a0) + siga2**2.*np.outer(cov_a2,cov_a2) + sigcs4**2.*np.outer(cov_cs4,cov_cs4)+sigcs2**2.*np.outer(cov_cs2,cov_cs2)+sigcs0**2.*np.outer(cov_cs0,cov_cs0) + sigb4**2.*np.outer(cov_b4,cov_b4) + sigBshot**2.*np.outer(cov_Bshot,cov_Bshot) + sigc1**2.*np.outer(cov_c1,cov_c1)
                
                # Compute chi2
                chi2 = np.inner(theory_minus_data,np.inner(np.linalg.inv(marg_cov),theory_minus_data))
                
                # Correct normalizations
                chi2 +=np.linalg.slogdet(marg_cov)[1] - dataset.logdetcov
                
                # Add parameter priors
                chi2 += (Pshot-0.)**2./1.**2. + (Bshot-1.)**2. + (c1-0.)**2./5.**2. + (b2-0.)**2./1.**2. + (bG2-0.)**2./1.**2.
                
                return -0.5*chi2
