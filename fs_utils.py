#### A collection of classes used in the full_shape_spectra likelihood

import numpy as np
import os
from scipy import interpolate
import scipy.integrate as integrate
from scipy.special.orthogonal import p_roots

class Datasets(object):
        def __init__(self, options):
                """Load Pk, Q0 and B0 data from file, as well as covariance matrix. The `options' argument is a dictionary of options specifying file names etc."""
                
                # Load datasets
                if options.use_Q and not options.use_P:
                        raise Exception("Cannot use Q0 without power spectra!")
                if options.use_P:
                        self.load_power_spectrum(options)
                if options.use_B:
                        self.load_bispectrum(options)
                else:
                        self.nB = 0
                if options.use_AP:
                        self.load_AP(options)
                else:
                        self.nAP = 0

                # Load covariance
                self.load_covariance(options)

        def load_power_spectrum(self, options):
                """Load power spectrum multipole dataset, optionally including Q0"""
                
                # Load raw Pk measurements
                k_init,P0_init,P2_init,P4_init=np.loadtxt(os.path.join(options.data_directory, options.P_measurements), skiprows = 0, unpack=True)

                # Count number of P bins (nP) and Q bins (nQ)
                self.nP_init = len(k_init)
                if options.use_Q:
                        self.nPQ = np.sum((k_init<options.kmaxQ)&(k_init>=options.kminP)) 
                        self.nQ = np.sum((k_init<options.kmaxQ)&(k_init>=options.kmaxP))
                        self.nP = self.nPQ - self.nQ
                        self.omitP = np.sum((k_init<options.kminP)) # bins to omit at start of Pk array              
                        self.omitQ = self.nP + self.omitP # bins to omit at start of Q0 array
                else:
                        self.nP = np.sum((k_init<options.kmaxP)&(k_init>=options.kminP)) 
                        self.nPQ = self.nP
                        self.nQ = 0
                        self.omitP = np.sum((k_init<options.kminP)) # bins to omit at start of Pk array              
                
                # Filter k and P_ell to correct bins
                self.kPQ = k_init[self.omitP:self.omitP+self.nPQ]
                self.dkPQ = self.kPQ[1]-self.kPQ[0] # bin width
                P0 = P0_init[self.omitP:self.omitP+self.nPQ]
                P2 = P2_init[self.omitP:self.omitP+self.nPQ]
                P4 = P4_init[self.omitP:self.omitP+self.nPQ]

                # Define data vectors
                self.P0 = P0[:self.nP]
                self.P2 = P2[:self.nP]
                self.P4 = P4[:self.nP]

                # Compute Q0 from Pk0 measurements
                if options.use_Q:
                        self.Q0 = P0[self.nP:]-1./2.*P2[self.nP:]+3./8.*P4[self.nP:]

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
                self.nAP = 2

        def load_covariance(self, options):
                """Load in the covariance matrix, filtered to the required bins and datasets [with the ordering P0, P2, P4, Q0, B0, AP]."""
                
                # Load full covariance matrix
                cov1 = np.loadtxt(os.path.join(options.data_directory, options.covmat_file),dtype=np.float64)

                # Define which bins we use
                filt = []
                if options.use_P:
                        filt.append(np.arange(self.omitP,self.omitP+self.nP)) # P0
                        filt.append(np.arange(self.omitP+self.nP_init,self.omitP+self.nP_init+self.nP)) # P2
                        filt.append(np.arange(self.omitP+2*self.nP_init,self.omitP+2*self.nP_init+self.nP)) # P4
                if options.use_Q:
                        filt.append(np.arange(self.omitQ+3*self.nP_init,self.omitQ+3*self.nP_init+self.nQ)) # Q0
                if options.use_B:
                        filt.append(np.arange(4*self.nP_init,4*self.nP_init+self.nB)) # B0
                if options.use_AP:
                        filt.append([-2,-1])
                filt= np.concatenate(filt)
                
                # Filter to the correct bins we want
                self.cov = np.zeros((len(filt),len(filt)),dtype='float64')
                for i,index in enumerate(filt):
                        for j,jndex in enumerate(filt):
                                self.cov[i,j] = cov1[index,jndex]

                # Compute matrix determinant for later use
                self.logdetcov = np.linalg.slogdet(self.cov)[1]

class PkTheory(object):
    def __init__(self, options, all_theory, h, As, fNL, fNL2, norm, fz, k_grid, kPQ, nP, nQ, Tk):
            """Compute the theoretical power spectrum P(k) and parameter derivatives for a given cosmology and set of nuisance parameters."""
            self.all_theory = all_theory
            self.h = h
            self.As = As
            self.norm = norm
            self.fNL = fNL
            self.fNL2 = fNL2
            self.k_grid = k_grid
            self.kPQ = kPQ
            self.fz = fz
            self.nP = nP
            self.nQ = nQ
            self.Tk = Tk
            self.options = options
            self.dataset = options.dataset
    
    def bin_integrator(self, input_table):
            """If bin-integration is included, integrate the function defined in `inttab' over the binned k-space. Else, return the input table."""
            
            k_grid = self.k_grid

            if self.options.bin_integration_P:
                    f_int = interpolate.InterpolatedUnivariateSpline(k_grid,input_table,ext=3)
                    integrand = lambda k: np.exp(3.*k)*f_int(np.exp(k))
                    out = np.zeros(len(self.kPQ))
                    for i in range(len(self.kPQ)):
                            kmin = self.dataset.dkPQ*i+self.options.kminP
                            kmax = self.dataset.dkPQ*(i+1)+self.options.kminP
                            out[i] = integrate.quad(integrand, np.log(kmin), np.log(kmax))[0]*3./(kmax**3.-kmin**3.)
                    return out
            else:
                    return input_table
    
    def compute_Pl_oneloop(self, b1, b2, bG2, bGamma3, cs0, cs2, cs4, b4, a0, a2, inv_nbar, Pshot, bphi):
            """Compute the 1-loop power spectrum multipoles, given the bias parameters."""
            
            if not hasattr(self, 'P0'):
                    self._load_P_oneloop_all(b1, b2, bG2, bGamma3, cs0, cs2, cs4, b4, a0, a2, inv_nbar, Pshot, bphi)
            
            P0 = self.P0[:self.nP]
            P2 = self.P2[:self.nP]
            P4 = self.P4[:self.nP]
            
            return P0, P2, P4

    def compute_Q0_oneloop(self, b1, b2, bG2, bGamma3, cs0, cs2, cs4, b4, a0, a2, inv_nbar, Pshot, bphi):
            """Compute the 1-loop Q0 theory, given the bias parameters."""
            
            if not hasattr(self, 'P0'):
                    self._load_P_oneloop_all(b1, b2, bG2, bGamma3, cs0, cs2, cs4, b4, a0, a2, inv_nbar, Pshot, bphi)
            
            Q0 = self.P0[self.nP:]-1./2.*self.P2[self.nP:]+3./8.*self.P4[self.nP:]
            
            return Q0

    def _load_P_oneloop_all(self, b1, b2, bG2, bGamma3, cs0, cs2, cs4, b4, a0, a2, inv_nbar, Pshot, bphi):
            """Internal function to compute the 1-loop power spectrum multipoles for all k, given the bias parameters."""
            
            # Load quantities
            all_theory = self.all_theory
            norm = self.norm
            h = self.h
            fz = self.fz
            fNL = self.fNL
            fNL2 = self.fNL2
            k_grid = self.k_grid

            # fNL parameters
            if not hasattr(self, 'phif'):
                self.phif = (fNL+fNL2)*(18./5.)*(b1-1.)*1.686*bphi*((k_grid/0.45)**2./self.Tk)
                self.phif1 = (fNL+fNL2)*(18./5.)*(b1-1.)*1.686*((k_grid/0.45)**2./self.Tk)
                self.fnlc = (self.As**0.5)*1944./625.*np.pi**4.

            if fNL==0 and fNL2==0:
                self.P0 = self.bin_integrator((norm**2.*all_theory[15]+norm**4.*(all_theory[21])+norm**1.*b1*all_theory[16]+norm**3.*b1*(all_theory[22]) + norm**0.*b1**2.*all_theory[17] +norm**2.*b1**2.*all_theory[23] + 0.25*norm**2.*b2**2.*all_theory[1] +b1*b2*norm**2.*all_theory[30]+ b2*norm**3.*all_theory[31] + b1*bG2*norm**2.*all_theory[32]+ bG2*norm**3.*all_theory[33] + b2*bG2*norm**2.*all_theory[4]+ bG2**2.*norm**2.*all_theory[5] + 2.*cs0*norm**2.*all_theory[11]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**2.*(b1*all_theory[7]+norm*all_theory[8]))*h**3. + (inv_nbar)*Pshot + a0*inv_nbar*(k_grid/0.5)**2.  + fz**2.*b4*k_grid**2.*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)*all_theory[13]*h + a2*(1./3.)*(10.**4.)*(k_grid/0.45)**2.)
                self.P2 = self.bin_integrator((norm**2.*all_theory[18]+norm**4.*(all_theory[24])+norm**1.*b1*all_theory[19]+norm**3.*b1*(all_theory[25]) + b1**2.*norm**2.*all_theory[26] +b1*b2*norm**2.*all_theory[34]+ b2*norm**3.*all_theory[35] + b1*bG2*norm**2.*all_theory[36]+ bG2*norm**3.*all_theory[37] + 0.25*b2**2.*all_theory[42] + b2*bG2*all_theory[43] + (bG2**2.)*all_theory[44] + 2.*cs2*norm**2.*all_theory[12]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**3.*all_theory[9])*h**3. + fz**2.*b4*k_grid**2.*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)*all_theory[13]*h + a2*(10.**4.)*(2./3.)*(k_grid/0.45)**2.)
                self.P4 = self.bin_integrator((norm**2.*all_theory[20]+norm**4.*all_theory[27]+b1*norm**3.*all_theory[28]+b1**2.*norm**2.*all_theory[29] + b2*norm**3.*all_theory[38] + bG2*norm**3.*all_theory[39] + b1*b2*all_theory[40] + b1*bG2*all_theory[41] + 0.25*b2**2.*all_theory[45] + b2*bG2*all_theory[46] + (bG2**2.)*all_theory[46] +2.*cs4*norm**2.*all_theory[13]/h**2.)*h**3. + fz**2.*b4*k_grid**2.*(norm**2.*fz**2.*210./143. + 30.*fz*b1*norm/11.+b1**2.)*all_theory[13]*h)
            else:
                self.P0 = self.bin_integrator((norm**2.*all_theory[15]+norm**4.*(all_theory[21])+norm**1.*b1*all_theory[16]+norm**3.*b1*(all_theory[22]) + norm**0.*b1**2.*all_theory[17] +norm**2.*b1**2.*all_theory[23] + 0.25*norm**2.*b2**2.*all_theory[1] +b1*b2*norm**2.*all_theory[30]+ b2*norm**3.*all_theory[31] + b1*bG2*norm**2.*all_theory[32]+ bG2*norm**3.*all_theory[33] + b2*bG2*norm**2.*all_theory[4]+ bG2**2.*norm**2.*all_theory[5] + 2.*cs0*norm**2.*all_theory[11]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**2.*(b1*all_theory[7]+norm*all_theory[8]))*h**3. + (inv_nbar)*Pshot + a0*inv_nbar*(k_grid/0.5)**2.  + fz**2.*b4*k_grid**2.*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)*all_theory[13]*h + a2*(1./3.)*(10.**4.)*(k_grid/0.45)**2. + self.fnlc*fNL*(h**3.)*(all_theory[51]+b1*all_theory[52]+b1**2.*all_theory[53]+b1*b2*all_theory[60]+b2*all_theory[61]+b1*bG2*all_theory[62]+bG2*all_theory[63]) + 1.*(2.*b1*self.phif+self.phif**2.)*all_theory[17]*(h**3.) + 1.*self.phif*all_theory[16]*(h**3.) + self.fnlc*fNL2*(h**3.)*(all_theory[75]+b1*all_theory[76]+b1**2.*all_theory[77]+b1*b2*all_theory[84]+b2*all_theory[85]+b1*bG2*all_theory[86]+bG2*all_theory[87]))
                self.P2 = self.bin_integrator((norm**2.*all_theory[18]+norm**4.*(all_theory[24])+norm**1.*b1*all_theory[19]+norm**3.*b1*(all_theory[25]) + b1**2.*norm**2.*all_theory[26] +b1*b2*norm**2.*all_theory[34]+ b2*norm**3.*all_theory[35] + b1*bG2*norm**2.*all_theory[36]+ bG2*norm**3.*all_theory[37] + 0.25*b2**2.*all_theory[42] + b2*bG2*all_theory[43] + (bG2**2.)*all_theory[44] + 2.*cs2*norm**2.*all_theory[12]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**3.*all_theory[9])*h**3. + fz**2.*b4*k_grid**2.*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)*all_theory[13]*h + a2*(10.**4.)*(2./3.)*(k_grid/0.45)**2.+ self.fnlc*fNL*(h**3.)*(all_theory[54]+b1*all_theory[55]+b1**2.*all_theory[56]+b1*b2*all_theory[64]+b2*all_theory[65]+b1*bG2*all_theory[66]+bG2*all_theory[67]) + 1.*self.phif*all_theory[19]*(h**3.) + self.fnlc*fNL2*(h**3.)*(all_theory[78]+b1*all_theory[79]+b1**2.*all_theory[80]+b1*b2*all_theory[88]+b2*all_theory[89]+b1*bG2*all_theory[90]+bG2*all_theory[91]))
                self.P4 = self.bin_integrator((norm**2.*all_theory[20]+norm**4.*all_theory[27]+b1*norm**3.*all_theory[28]+b1**2.*norm**2.*all_theory[29] + b2*norm**3.*all_theory[38] + bG2*norm**3.*all_theory[39] + b1*b2*all_theory[40] + b1*bG2*all_theory[41] + 0.25*b2**2.*all_theory[45] + b2*bG2*all_theory[46] + (bG2**2.)*all_theory[46] +2.*cs4*norm**2.*all_theory[13]/h**2.)*h**3. + fz**2.*b4*k_grid**2.*(norm**2.*fz**2.*210./143. + 30.*fz*b1*norm/11.+b1**2.)*all_theory[13]*h+self.fnlc*fNL*(h**3.)*(all_theory[57]+b1*all_theory[58]+b1**2.*all_theory[59]+b1*b2*all_theory[68]+b2*all_theory[69]+b1*bG2*all_theory[70]+bG2*all_theory[71]) + self.fnlc*fNL2*(h**3.)*(all_theory[81]+b1*all_theory[82]+b1**2.*all_theory[83]+b1*b2*all_theory[92]+b2*all_theory[93]+b1*bG2*all_theory[94]+bG2*all_theory[95]))
            
    def _load_individual_derivatives(self, b1):
            """Compute individual derivatives needed to construct Pl and Q0 derivatives. This preloads the quantities requiring bin integration."""

            # Load quantities
            all_theory = self.all_theory
            norm = self.norm
            h = self.h
            fz = self.fz
            k_grid = self.k_grid
            
            self.deriv0_bGamma3 = self.bin_integrator((0.8*norm)*norm**2.*(b1*all_theory[7]+norm*all_theory[8])*h**3.)
            self.deriv2_bGamma3 = self.bin_integrator((0.8*norm)*norm**3.*all_theory[9]*h**3.)
            self.deriv0_cs0 = self.bin_integrator(2.*norm**2.*all_theory[11]*h**1.)
            self.deriv2_cs2 = self.bin_integrator(2.*norm**2.*all_theory[12]*h**1.)
            self.deriv4_cs4 = self.bin_integrator(2.*norm**2.*all_theory[13]*h**1.)
            self.derivN_b4 = self.bin_integrator(fz**2.*k_grid**2.*all_theory[13]*h)
            self.deriv0_bphi = self.bin_integrator((2.*b1*self.phif1+2.*self.phif1*self.phif)*all_theory[17]*(h**3.) + self.phif1*all_theory[16]*(h**3.))
            self.deriv2_bphi = self.bin_integrator(self.phif1*all_theory[19]*(h**3.))
            
    def compute_Pl_derivatives(self, b1):
            """Compute the derivatives of the power spectrum multipoles with respect to parameters entering the model linearly"""
            
            # Load quantities
            norm = self.norm
            fz = self.fz
            kPQ = self.kPQ
            nP = self.nP
            
            # Compute individual derivatives
            if not hasattr(self, 'deriv0_bGamma3'):
                    self._load_individual_derivatives(b1)
            
            # Initialize arrays
            deriv_bGamma3P, deriv_cs0P, deriv_cs2P, deriv_cs4P, deriv_b4P, deriv_PshotP, deriv_a0P, deriv_a2P, deriv_bphiP = [np.zeros(3*nP) for _ in range(9)]

            # Assemble stacked derivatives
            deriv_bGamma3P[:nP] = self.deriv0_bGamma3[:nP]
            deriv_bGamma3P[nP:2*nP] = self.deriv2_bGamma3[:nP]

            deriv_cs0P[:nP] = self.deriv0_cs0[:nP]
            deriv_cs2P[nP:2*nP] = self.deriv2_cs2[:nP]
            deriv_cs4P[2*nP:3*nP] = self.deriv4_cs4[:nP]

            deriv_b4P[:nP] = self.derivN_b4[:nP]*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)
            deriv_b4P[nP:2*nP] = self.derivN_b4[:nP]*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)
            deriv_b4P[2*nP:3*nP] = self.derivN_b4[:nP]*(norm**2.*fz**2.*210./143. + 30.*fz*b1*norm/11.+b1**2.)

            deriv_PshotP[:nP] = 1.
            
            deriv_a0P[:nP] = (kPQ[:nP]/0.45)**2.
            
            deriv_a2P[:nP] = (1./3.)*(kPQ[:nP]/0.45)**2.
            deriv_a2P[nP:2*nP] = (2./3.)*(kPQ[:nP]/0.45)**2.

            deriv_bphiP[:nP] = self.deriv0_bphi[:nP]
            deriv_bphiP[nP:2*nP] = self.deriv2_bphi[:nP]

            return deriv_bGamma3P, deriv_cs0P, deriv_cs2P, deriv_cs4P, deriv_b4P, deriv_PshotP, deriv_a0P, deriv_a2P, deriv_bphiP

    def compute_Q0_derivatives(self, b1):
            """Compute the derivatives of Q0 with respect to parameters entering the model linearly"""
            
            # Load quantities
            norm = self.norm
            h = self.h
            fz = self.fz
            kPQ = self.kPQ
            nP = self.nP
            
            # Compute individual derivatives
            if not hasattr(self, 'deriv0_bGamma3'):
                    self._load_individual_derivatives(b1)
            
            # Initialize arrays
            deriv_bGamma3Q, deriv_cs0Q, deriv_cs2Q, deriv_cs4Q, deriv_b4Q, deriv_PshotQ, deriv_a0Q, deriv_a2Q = [np.zeros(self.nQ) for _ in range(8)]

            # Assemble stacked derivatives
            deriv_bGamma3Q = self.deriv0_bGamma3[nP:] - 1./2.*self.deriv2_bGamma3[nP:]
            deriv_cs0Q = self.deriv0_cs0[nP:]
            deriv_cs2Q = -1./2.*self.deriv2_cs2[nP:]
            deriv_cs4Q = 3./8.*self.deriv4_cs4[nP:]
            deriv_b4Q = self.derivN_b4[nP:]*((norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.) - ((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)/2. +3.*(norm**2.*fz**2.*210./143. + 30.*fz*b1*norm/11.+b1**2.)/8.)
            deriv_PshotQ = 1.
            deriv_a0Q = (kPQ[nP:]/0.45)**2.
            deriv_bphiQ = self.deriv0_bphi[nP:] - 1./2.*self.deriv2_bphi[nP:]

            return deriv_bGamma3Q, deriv_cs0Q, deriv_cs2Q, deriv_cs4Q, deriv_b4Q, deriv_PshotQ, deriv_a0Q, deriv_a2Q, deriv_bphiQ


class BkUtils(object):
        def __init__(self):
                """Load a number of utility functions for the bispectrum likelihood computation"""

                # Load angular integration grid
                if not hasattr(self, 'mesh_mu'):
                        self.compute_angular_grid()

                # Selection of other bispectrum functions dumped from Mathematica
                self.F2 = lambda k1,k2,k3,b1,b2,bG2: (b1*(-5.*(k1**2.-k2**2.)**2.+3.*(k1**2.+k2**2.)*k3**2.+2.*k3**4.) + 7.*(2.*b2*k1**2.*k2**2. + bG2*(k1-k2-k3)*(k1+k2-k3)*(k1-k2+k3)*(k1+k2+k3)))*b1**2./28./k1**2./k2**2.
                self.G2 = lambda k1,k2,k3: -((3*(k1**2-k2**2)**2+(k1**2+k2**2)*k3**2-4*k3**4)/(28 *k1**2 *k2**2))
                self.j2 = lambda x: (3./x**2.-1.)*np.sin(x)/x - 3.*np.cos(x)/x**2.

        def compute_angular_grid(self, n_gauss=3, n_gauss2 = 8):
                """Load the 5D angular grid used for Monte Carlo integration"""

                [gauss_mu,self.gauss_w], [gauss_mu2,self.gauss_w2] = p_roots(n_gauss), p_roots(n_gauss2)
                self.mesh_mu = np.meshgrid(gauss_mu,gauss_mu,gauss_mu,gauss_mu2,gauss_mu2, sparse=True, indexing='ij')

