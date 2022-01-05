#### A collection of classes used in the full_shape_spectra likelihood

import numpy as np
import os
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

