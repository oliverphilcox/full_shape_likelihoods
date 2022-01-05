#### A collection of classes used in the full_shape_spectra likelihood

import numpy as np
import os
from scipy import interpolate
import scipy.integrate as integrate

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
                self.triangle_indices = [np.asarray(kk/self.dkB,dtype=int)-int(options.kminB/self.dkB) for kk in [khere, khere2, khere3]]
        
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
        def __init__(self, options, all_theory, h, As, fNL_eq, fNL_orth, norm, fz, k_grid, kPQ, nP, nQ, Tk):
                """Compute the theoretical power spectrum P(k) and parameter derivatives for a given cosmology and set of nuisance parameters."""
                
                # Read in input parameters
                self.all_theory = all_theory
                self.h = h
                self.As = As
                self.norm = norm
                self.fNL_eq = fNL_eq
                self.fNL_orth = fNL_orth
                self.k_grid = k_grid
                self.kPQ = kPQ
                self.fz = fz
                self.nP = nP
                self.nQ = nQ
                self.Tk = Tk
                self.options = options
                self.dataset = options.dataset
        
        def bin_integrator(self, input_table1):
                """If bin-integration is included, integrate the function defined in `inttab' over the binned k-space. Else, return the input table1."""
                
                if self.options.bin_integration_P:
                        f_int = interpolate.InterpolatedUnivariateSpline(self.k_grid,input_table1,ext=3)
                        integrand = lambda k: np.exp(3.*k)*f_int(np.exp(k))
                        out = np.zeros(len(self.kPQ))
                        for i in range(len(self.kPQ)):
                                kmin = self.dataset.dkPQ*i+self.options.kminP
                                kmax = self.dataset.dkPQ*(i+1)+self.options.kminP
                                out[i] = integrate.quad(integrand, np.log(kmin), np.log(kmax))[0]*3./(kmax**3.-kmin**3.)
                        return out
                else:
                        return input_table1
        
        def compute_Pl_oneloop(self, b1, b2, bG2, bGamma3, cs0, cs2, cs4, b4, a0, a2, inv_nbar, Pshot, bphi):
                """Compute the 1-loop power spectrum multipoles, given the bias parameters."""
                
                # Run the main code
                if not hasattr(self, 'P0'):
                        self._load_P_oneloop_all(b1, b2, bG2, bGamma3, cs0, cs2, cs4, b4, a0, a2, inv_nbar, Pshot, bphi)
                
                # Extract the power spectrum multipoles
                P0 = self.P0[:self.nP]
                P2 = self.P2[:self.nP]
                P4 = self.P4[:self.nP]
                
                return P0, P2, P4

        def compute_Q0_oneloop(self, b1, b2, bG2, bGamma3, cs0, cs2, cs4, b4, a0, a2, inv_nbar, Pshot, bphi):
                """Compute the 1-loop Q0 theory, given the bias parameters."""
                
                # Run the main code
                if not hasattr(self, 'P0'):
                        self._load_P_oneloop_all(b1, b2, bG2, bGamma3, cs0, cs2, cs4, b4, a0, a2, inv_nbar, Pshot, bphi)
                
                # Extract Q0
                Q0 = self.P0[self.nP:]-1./2.*self.P2[self.nP:]+3./8.*self.P4[self.nP:]
                
                return Q0

        def _load_P_oneloop_all(self, b1, b2, bG2, bGamma3, cs0, cs2, cs4, b4, a0, a2, inv_nbar, Pshot, bphi):
                """Internal function to compute the 1-loop power spectrum multipoles for all k, given the bias parameters."""
                
                # Load variables
                all_theory = self.all_theory
                norm = self.norm
                h = self.h
                fz = self.fz
                fNL_eq = self.fNL_eq
                fNL_orth = self.fNL_orth
                k_grid = self.k_grid

                # Compute fNL factors
                if not hasattr(self, 'phif') and not (fNL_eq==0 and fNL_orth==0):
                        self.phif = (fNL_eq+fNL_orth)*(18./5.)*(b1-1.)*1.686*bphi*((k_grid/0.45)**2./self.Tk)
                        self.phif1 = (fNL_eq+fNL_orth)*(18./5.)*(b1-1.)*1.686*((k_grid/0.45)**2./self.Tk)
                        self.fnlc = (self.As**0.5)*1944./625.*np.pi**4.

                # Compute the power spectrum multipoles, including bin integration if requested
                if fNL_eq==0 and fNL_orth==0:
                        self.P0 = self.bin_integrator((norm**2.*all_theory[15]+norm**4.*(all_theory[21])+norm**1.*b1*all_theory[16]+norm**3.*b1*(all_theory[22]) + norm**0.*b1**2.*all_theory[17] +norm**2.*b1**2.*all_theory[23] + 0.25*norm**2.*b2**2.*all_theory[1] +b1*b2*norm**2.*all_theory[30]+ b2*norm**3.*all_theory[31] + b1*bG2*norm**2.*all_theory[32]+ bG2*norm**3.*all_theory[33] + b2*bG2*norm**2.*all_theory[4]+ bG2**2.*norm**2.*all_theory[5] + 2.*cs0*norm**2.*all_theory[11]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**2.*(b1*all_theory[7]+norm*all_theory[8]))*h**3. + (inv_nbar)*Pshot + a0*inv_nbar*(k_grid/0.5)**2.  + fz**2.*b4*k_grid**2.*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)*all_theory[13]*h + a2*(1./3.)*(10.**4.)*(k_grid/0.45)**2.)
                        self.P2 = self.bin_integrator((norm**2.*all_theory[18]+norm**4.*(all_theory[24])+norm**1.*b1*all_theory[19]+norm**3.*b1*(all_theory[25]) + b1**2.*norm**2.*all_theory[26] +b1*b2*norm**2.*all_theory[34]+ b2*norm**3.*all_theory[35] + b1*bG2*norm**2.*all_theory[36]+ bG2*norm**3.*all_theory[37] + 0.25*b2**2.*all_theory[42] + b2*bG2*all_theory[43] + (bG2**2.)*all_theory[44] + 2.*cs2*norm**2.*all_theory[12]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**3.*all_theory[9])*h**3. + fz**2.*b4*k_grid**2.*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)*all_theory[13]*h + a2*(10.**4.)*(2./3.)*(k_grid/0.45)**2.)
                        self.P4 = self.bin_integrator((norm**2.*all_theory[20]+norm**4.*all_theory[27]+b1*norm**3.*all_theory[28]+b1**2.*norm**2.*all_theory[29] + b2*norm**3.*all_theory[38] + bG2*norm**3.*all_theory[39] + b1*b2*all_theory[40] + b1*bG2*all_theory[41] + 0.25*b2**2.*all_theory[45] + b2*bG2*all_theory[46] + (bG2**2.)*all_theory[46] +2.*cs4*norm**2.*all_theory[13]/h**2.)*h**3. + fz**2.*b4*k_grid**2.*(norm**2.*fz**2.*210./143. + 30.*fz*b1*norm/11.+b1**2.)*all_theory[13]*h)
                else:
                        self.P0 = self.bin_integrator((norm**2.*all_theory[15]+norm**4.*(all_theory[21])+norm**1.*b1*all_theory[16]+norm**3.*b1*(all_theory[22]) + norm**0.*b1**2.*all_theory[17] +norm**2.*b1**2.*all_theory[23] + 0.25*norm**2.*b2**2.*all_theory[1] +b1*b2*norm**2.*all_theory[30]+ b2*norm**3.*all_theory[31] + b1*bG2*norm**2.*all_theory[32]+ bG2*norm**3.*all_theory[33] + b2*bG2*norm**2.*all_theory[4]+ bG2**2.*norm**2.*all_theory[5] + 2.*cs0*norm**2.*all_theory[11]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**2.*(b1*all_theory[7]+norm*all_theory[8]))*h**3. + (inv_nbar)*Pshot + a0*inv_nbar*(k_grid/0.5)**2.  + fz**2.*b4*k_grid**2.*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)*all_theory[13]*h + a2*(1./3.)*(10.**4.)*(k_grid/0.45)**2. + self.fnlc*fNL_eq*(h**3.)*(all_theory[51]+b1*all_theory[52]+b1**2.*all_theory[53]+b1*b2*all_theory[60]+b2*all_theory[61]+b1*bG2*all_theory[62]+bG2*all_theory[63]) + 1.*(2.*b1*self.phif+self.phif**2.)*all_theory[17]*(h**3.) + 1.*self.phif*all_theory[16]*(h**3.) + self.fnlc*fNL_orth*(h**3.)*(all_theory[75]+b1*all_theory[76]+b1**2.*all_theory[77]+b1*b2*all_theory[84]+b2*all_theory[85]+b1*bG2*all_theory[86]+bG2*all_theory[87]))
                        self.P2 = self.bin_integrator((norm**2.*all_theory[18]+norm**4.*(all_theory[24])+norm**1.*b1*all_theory[19]+norm**3.*b1*(all_theory[25]) + b1**2.*norm**2.*all_theory[26] +b1*b2*norm**2.*all_theory[34]+ b2*norm**3.*all_theory[35] + b1*bG2*norm**2.*all_theory[36]+ bG2*norm**3.*all_theory[37] + 0.25*b2**2.*all_theory[42] + b2*bG2*all_theory[43] + (bG2**2.)*all_theory[44] + 2.*cs2*norm**2.*all_theory[12]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**3.*all_theory[9])*h**3. + fz**2.*b4*k_grid**2.*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)*all_theory[13]*h + a2*(10.**4.)*(2./3.)*(k_grid/0.45)**2.+ self.fnlc*fNL_eq*(h**3.)*(all_theory[54]+b1*all_theory[55]+b1**2.*all_theory[56]+b1*b2*all_theory[64]+b2*all_theory[65]+b1*bG2*all_theory[66]+bG2*all_theory[67]) + 1.*self.phif*all_theory[19]*(h**3.) + self.fnlc*fNL_orth*(h**3.)*(all_theory[78]+b1*all_theory[79]+b1**2.*all_theory[80]+b1*b2*all_theory[88]+b2*all_theory[89]+b1*bG2*all_theory[90]+bG2*all_theory[91]))
                        self.P4 = self.bin_integrator((norm**2.*all_theory[20]+norm**4.*all_theory[27]+b1*norm**3.*all_theory[28]+b1**2.*norm**2.*all_theory[29] + b2*norm**3.*all_theory[38] + bG2*norm**3.*all_theory[39] + b1*b2*all_theory[40] + b1*bG2*all_theory[41] + 0.25*b2**2.*all_theory[45] + b2*bG2*all_theory[46] + (bG2**2.)*all_theory[46] +2.*cs4*norm**2.*all_theory[13]/h**2.)*h**3. + fz**2.*b4*k_grid**2.*(norm**2.*fz**2.*210./143. + 30.*fz*b1*norm/11.+b1**2.)*all_theory[13]*h+self.fnlc*fNL_eq*(h**3.)*(all_theory[57]+b1*all_theory[58]+b1**2.*all_theory[59]+b1*b2*all_theory[68]+b2*all_theory[69]+b1*bG2*all_theory[70]+bG2*all_theory[71]) + self.fnlc*fNL_orth*(h**3.)*(all_theory[81]+b1*all_theory[82]+b1**2.*all_theory[83]+b1*b2*all_theory[92]+b2*all_theory[93]+b1*bG2*all_theory[94]+bG2*all_theory[95]))
                
        def _load_individual_derivatives(self, b1):
                """Compute individual derivatives needed to construct Pl and Q0 derivatives. This preloads the quantities requiring bin integration."""

                # Load quantities
                all_theory = self.all_theory
                norm = self.norm
                h = self.h
                fz = self.fz
                k_grid = self.k_grid
                
                # Compute derivatives, including bin integration if requested
                self.deriv0_bGamma3 = self.bin_integrator((0.8*norm)*norm**2.*(b1*all_theory[7]+norm*all_theory[8])*h**3.)
                self.deriv2_bGamma3 = self.bin_integrator((0.8*norm)*norm**3.*all_theory[9]*h**3.)
                self.deriv0_cs0 = self.bin_integrator(2.*norm**2.*all_theory[11]*h**1.)
                self.deriv2_cs2 = self.bin_integrator(2.*norm**2.*all_theory[12]*h**1.)
                self.deriv4_cs4 = self.bin_integrator(2.*norm**2.*all_theory[13]*h**1.)
                self.derivN_b4 = self.bin_integrator(fz**2.*k_grid**2.*all_theory[13]*h)

                # Derivatives involving fNL
                if self.fNL_eq==0 and self.fNL_orth==0:
                        self.deriv0_bphi = np.zeros(len(self.kPQ))
                        self.deriv2_bphi = np.zeros(len(self.kPQ))
                else:
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

class BkTheory(object):
        def __init__(self, options, As, fNL_eq, fNL_orth, apar, aperp, fz, r_bao, k_grid, Pk_lin_table1, Pk_lin_table2, gauss_w, gauss_w2, mesh_mu, nB):
                """Compute the theoretical power spectrum P(k) and parameter derivatives for a given cosmology and set of nuisance parameters."""
                
                # Load variables
                self.options = options
                self.dataset = options.dataset
                self.fNL_eq = fNL_eq
                self.fNL_orth = fNL_orth
                self.apar = apar
                self.aperp = aperp
                self.fz = fz
                self.r_bao = r_bao
                self.gauss_w = gauss_w
                self.gauss_w2 = gauss_w2
                self.mesh_mu = mesh_mu
                self.nB = nB
                self.k_grid = k_grid
                
                # Load fNL variables
                self.Azeta = As*2.*np.pi**2.
                self.Pk_lin_table1 = Pk_lin_table1
                self.Pk_lin_table2 = Pk_lin_table2
                self.P0int = interpolate.InterpolatedUnivariateSpline(k_grid,Pk_lin_table1,ext=3)

                # Function definitions
                self.F2 = lambda k1,k2,k3,b1,b2,bG2: (b1*(-5.*(k1**2.-k2**2.)**2.+3.*(k1**2.+k2**2.)*k3**2.+2.*k3**4.) + 7.*(2.*b2*k1**2.*k2**2. + bG2*(k1-k2-k3)*(k1+k2-k3)*(k1-k2+k3)*(k1+k2+k3)))*b1**2./28./k1**2./k2**2.
                self.G2 = lambda k1,k2,k3: -((3*(k1**2-k2**2)**2+(k1**2+k2**2)*k3**2-4*k3**4)/(28 *k1**2 *k2**2))
                self.j2 = lambda x: (3./x**2.-1.)*np.sin(x)/x - 3.*np.cos(x)/x**2.

        def _bin_integrate(self, B_matrix):
                """Function to integrate over bin lengths and angles via matrix multiplication"""

                return np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(B_matrix,self.gauss_w2)/2.,self.gauss_w2)/2.,self.gauss_w),self.gauss_w),self.gauss_w)

        def _Bk_eq(self, k1, k2, k3): 
                """Equilateral fNL bispectrum template"""

                return self.Azeta**2.*(self.Tfunc(k1)*self.Tfunc(k2)*self.Tfunc(k3)*(18./5.)*(-1./k1**3./k2**3.-1./k3**3./k2**3.-1./k1**3./k3**3.-2./k1**2./k2**2./k3**2.+1/k1/k2**2./k3**3.+1/k1/k3**2./k2**3.+1/k2/k3**2./k1**3.+1/k2/k1**2./k3**3.+1/k3/k1**2./k2**3.+1/k3/k2**2./k1**3.))

        def _Bk_orth(self, k1, k2, k3):
                """Orthogonal fNL bispectrum template"""

                p_here=27./(743./(7.*(20.*np.pi**2.-193.))-21.)
                ktot=k1+k2+k3
                e2=k1*k2+k1*k3+k3*k2
                e3=k1*k2*k3
                D_here=(ktot-2.*k1)*(ktot-2.*k2)*(ktot-2.*k3)
                G_here=2.*e2/3.-(k1**2.+k2**2.+k3**2.)/3.
                N_orth = (840.*np.pi**2.-7363.-189.*(20.*np.pi**2.-193.))/(29114. - 2940.*np.pi**2.)
                Bfunc = self.Azeta**2.*(18./5.)*self.Tfunc(k1)*self.Tfunc(k2)*self.Tfunc(k3)*(1./(k1**2.*k2**2.*k3**2.))*((1.+p_here)*D_here/e3 -p_here*G_here**3./e3**2.)/N_orth
                return Bfunc

        def _load_IR_resummation(self, b1, c1):
                """Load quantities relevant to IR resummation of bispectrum"""

                # IR resummation parameters
                ks_IR = 0.05
                r_bao = self.r_bao
                k_grid = self.k_grid
                P0int = self.P0int
                fz = self.fz

                # Compute IR Sigma
                Sigma = integrate.quad(lambda k: (4*np.pi)*np.exp(1.*k)*self.P0int(np.exp(k))*(1.-3*(2*r_bao*np.exp(k)*np.cos(np.exp(k)*r_bao)+(-2+r_bao**2*np.exp(k)**2)*np.sin(r_bao*np.exp(k)))/(np.exp(k)*r_bao)**3)/(3*(2*np.pi)**3.), np.log(2.e-4), np.log(0.2))[0]

                # Wiggly power spectrum
                Pw = (self.Pk_lin_table1-self.Pk_lin_table2)/(np.exp(-k_grid**2.*Sigma)-np.exp(-k_grid**2.*Sigma)*(1+k_grid**2.*Sigma))
                Pwfunc = interpolate.InterpolatedUnivariateSpline(k_grid,Pw,ext=3)
                
                # Non-Wiggly power spectrum
                Pnw = self.Pk_lin_table1 - Pw*np.exp(-k_grid**2.*Sigma)
                Pnwfunc = interpolate.InterpolatedUnivariateSpline(k_grid,Pnw,ext=3)

                # Recompute IR Sigma
                Sigma2 = integrate.quad(lambda k: (4*np.pi)*np.exp(1.*k)*P0int(np.exp(k))*(1.-3*(2*r_bao*np.exp(k)*np.cos(np.exp(k)*r_bao)+(-2+r_bao**2*np.exp(k)**2)*np.sin(r_bao*np.exp(k)))/(np.exp(k)*r_bao)**3)/(3*(2*np.pi)**3.), np.log(2.e-4), np.log(ks_IR))[0]
                deltaSigma2 = integrate.quad(lambda k: (4*np.pi)*np.exp(1.*k)*P0int(np.exp(k))*(self.j2(np.exp(k)*r_bao))/((2*np.pi)**3.), np.log(2.e-4), np.log(ks_IR))[0]

                # IR resummed spectra
                self.P_IR = lambda k, mu: Pnwfunc(k) +  np.exp(-k**2.*(Sigma2*(1.+2.*fz*mu**2.*(2.+fz)) + deltaSigma2*mu**2.*fz**2.*(mu**2.-1.)))*Pwfunc(k) -(c1*mu**2.)*(k/0.3)**2.*P0int(k)/(b1+fz*mu**2.)
                self.P_IRC = lambda k, mu:Pnwfunc(k) +  np.exp(-k**2.*(Sigma2*(1.+2.*fz*mu**2.*(2.+fz)) + deltaSigma2*mu**2.*fz**2.*(mu**2.-1.)))*Pwfunc(k) -(mu**2.)*(k/0.3)**2.*P0int(k)/(b1+fz*mu**2.)
                
        def _compute_B_matrices(self,beta,b1,b2,bG2,Pshot,Bshot,kc1,kc2,kc3,dk1,dk2,dk3,k1,k2,k3,mu1,phi):
                """Load the bispectrum matrices for a given set of k bins. These will later be integrated over bins to form the bispectrum monopole and derivatives"""
                
                # Define local variables
                fNL_eq = self.fNL_eq
                fNL_orth = self.fNL_orth
                apar = self.apar
                aperp = self.aperp
                inv_nbar = self.options.inv_nbar

                # Bin centers
                ddk1 = dk1/2.
                ddk2 = dk2/2.
                ddk3 = dk3/2.
                kk1 = (kc1+k1*ddk1)
                kk2 = (kc2+k2*ddk2)
                kk3 = (kc3+k3*ddk3)

                # Define mu1, mu2, mu3 angles
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
                P_IR1, P_IR2, P_IR3 = self.P_IR(kk1*qq1,nnu1), self.P_IR(kk2*qq2,nnu2), self.P_IR(kk3*qq3,nnu3)
                P_IR1C, P_IR2C, P_IR3C = self.P_IRC(kk1*qq1,nnu1), self.P_IRC(kk2*qq2,nnu2), self.P_IRC(kk3*qq3,nnu3)

                # Compute underlying bispectrum matrices
                zz21 = self.F2(kk1*qq1,kk2*qq2,kk3*qq3,b1,b2,bG2)+b1**3.*beta*((nnu2*kk2*qq2+nnu1*kk1*qq1)/kk3/qq3)**2.*self.G2(kk1*qq1,kk2*qq2,kk3*qq3)+(b1**4.*beta/2.)*(nnu2*kk2*qq2+nnu1*kk1*qq1)*(nnu1*(1.+beta*nnu2**2.)/kk1/qq1 + nnu2*(1.+beta*nnu1**2.)/kk2/qq2)
                zz22 = self.F2(kk1*qq1,kk3*qq3,kk2*qq2,b1,b2,bG2)+b1**3.*beta*((nnu3*kk3*qq3+nnu1*kk1*qq1)/kk2/qq2)**2.*self.G2(kk1*qq1,kk3*qq3,kk2*qq2)+(b1**4.*beta/2.)*(nnu3*kk3*qq3+nnu1*kk1*qq1)*(nnu1*(1.+beta*nnu3**2.)/kk1/qq1 + nnu3*(1.+beta*nnu1**2.)/kk3/qq3)
                zz23 = self.F2(kk2*qq2,kk3*qq3,kk1*qq1,b1,b2,bG2)+b1**3.*beta*((nnu2*kk2*qq2+nnu3*kk3*qq3)/kk1/qq1)**2.*self.G2(kk2*qq2,kk3*qq3,kk1*qq1)+(b1**4.*beta/2.)*(nnu2*kk2*qq2+nnu3*kk3*qq3)*(nnu2*(1.+beta*nnu3**2.)/kk2/qq2 + nnu3*(1.+beta*nnu2**2.)/kk3/qq3)

                FF2func1 = zz21*(1+beta*nnu1**2)*(1.+beta*nnu2**2.)*P_IR1*kk1*ddk1*P_IR2*kk2*ddk2*kk3*ddk3 + 1.*0.5*(Bshot*inv_nbar)*b1**2.*P_IR1*kk1*(1.+beta*nnu1**2.*(Bshot+1.*(1.+Pshot))/Bshot + beta**2.*nnu1**4.*1.*(1.+Pshot)/Bshot)*kk2*kk3*ddk1*ddk2*ddk3 + ((1.+Pshot)*inv_nbar)**2.*kk1*kk2*kk3*ddk1*ddk2*ddk3/2.
                FF2func2 = zz22*(1+beta*nnu1**2)*(1.+beta*nnu3**2.)*P_IR1*kk1*ddk1*P_IR3*kk3*ddk3*kk2*ddk2 + 1.*0.5*(Bshot*inv_nbar)*b1**2.*P_IR2*kk2*(1.+beta*nnu2**2.*(Bshot+1.+1.*Pshot)/Bshot + beta**2.*nnu2**4.*1.*(1.+Pshot)/Bshot)*kk1*kk3*ddk1*ddk2*ddk3 + 0.*(1*inv_nbar)**2.*kk1*kk2*kk3*ddk1*ddk2*ddk3/6.
                FF2func3 = zz23*(1+beta*nnu2**2)*(1.+beta*nnu3**2.)*P_IR2*kk2*ddk2*P_IR3*kk3*ddk3*kk1*ddk1 + 1.*0.5*(Bshot*inv_nbar)*b1**2.*P_IR3*kk3*(1.+beta*nnu3**2.*(Bshot+1.+1.*Pshot)/Bshot + beta**2.*nnu3**4.*1.*(1.+Pshot)/Bshot)*kk2*kk1*ddk1*ddk2*ddk3 + 0.*(1*inv_nbar)**2.*kk1*kk2*kk3*ddk1*ddk2*ddk3/6.
                
                FF2func1C = zz21*(1+beta*nnu1**2)*(1.+beta*nnu2**2.)*P_IR1C*kk1*ddk1*P_IR2C*kk2*ddk2*kk3*ddk3 + 1.*0.5*(Bshot*inv_nbar)*b1**2.*P_IR1C*kk1*(1.+beta*nnu1**2.*(Bshot+1.*(1.+Pshot))/Bshot + beta**2.*nnu1**4.*1.*(1.+Pshot)/Bshot)*kk2*kk3*ddk1*ddk2*ddk3 + ((1.+Pshot)*inv_nbar)**2.*kk1*kk2*kk3*ddk1*ddk2*ddk3/2.
                FF2func2C = zz22*(1+beta*nnu1**2)*(1.+beta*nnu3**2.)*P_IR1C*kk1*ddk1*P_IR3C*kk3*ddk3*kk2*ddk2 + 1.*0.5*(Bshot*inv_nbar)*b1**2.*P_IR2C*kk2*(1.+beta*nnu2**2.*(Bshot+1.+1.*Pshot)/Bshot + beta**2.*nnu2**4.*1.*(1.+Pshot)/Bshot)*kk1*kk3*ddk1*ddk2*ddk3 + 0.*(1*inv_nbar)**2.*kk1*kk2*kk3*ddk1*ddk2*ddk3/6.
                FF2func3C = zz23*(1+beta*nnu2**2)*(1.+beta*nnu3**2.)*P_IR2C*kk2*ddk2*P_IR3C*kk3*ddk3*kk1*ddk1 + 1.*0.5*(Bshot*inv_nbar)*b1**2.*P_IR3C*kk3*(1.+beta*nnu3**2.*(Bshot+1.+1.*Pshot)/Bshot + beta**2.*nnu3**4.*1.*(1.+Pshot)/Bshot)*kk2*kk1*ddk1*ddk2*ddk3 + 0.*(1*inv_nbar)**2.*kk1*kk2*kk3*ddk1*ddk2*ddk3/6.

                # Add matrices for primordial non-Gaussianity
                FFnlfunc = 0.
                if fNL_eq!=0:
                        FFnlfunc += fNL_eq*self._Bk_eq(kk1*qq1,kk2*qq2,kk3*qq3)*b1**3.*(1+beta*nnu1**2)*(1.+beta*nnu3**2.)*(1+beta*nnu2**2)*kk1*kk2*kk3*ddk1*ddk2*ddk3
                if fNL_orth!=0:
                        FFnlfunc += fNL_orth*self._Bk_orth(kk1*qq1,kk2*qq2,kk3*qq3)*b1**3.*(1+beta*nnu1**2)*(1.+beta*nnu3**2.)*(1+beta*nnu2**2)*kk1*kk2*kk3*ddk1*ddk2*ddk3

                # Assemble output bispectrum matrices
                B0_matrix = (2.*FF2func1 + 2.*FF2func2 + 2.*FF2func3 + FFnlfunc)/apar**2./aperp**4.
                deriv_Pshot_matrix = (b1**2.*(1.*beta*nnu1**2.*(1.+beta*nnu1**2.)*P_IR1+P_IR2*(beta*nnu2**2.)*(1.+beta*nnu2**2.)+ P_IR3*(beta*nnu3**2.)*(1.+beta*nnu3**2.)) + 2.*inv_nbar*(1.+Pshot))*kk1*kk2*kk3*ddk1*ddk2*ddk3/apar**2./aperp**4.
                deriv_Bshot_matrix = b1**2.*(((1.+beta*nnu1**2.)*P_IR1+P_IR2*(1.+beta*nnu2**2.)+ P_IR3*(1.+beta*nnu3**2.))*kk1*kk2*kk3*ddk1*ddk2*ddk3)/apar**2./aperp**4.
                deriv_c1_matrix = (2.*FF2func1C + 2.*FF2func2C + 2.*FF2func3C - 2.*FF2func1 - 2.*FF2func2 - 2.*FF2func3)/apar**2./aperp**4.
                
                return B0_matrix, deriv_Pshot_matrix, deriv_Bshot_matrix, deriv_c1_matrix

        def compute_B0_tree_theory_derivs(self, b1, b2, bG2, c1, Pshot, Bshot):
                """Compute the tree-level bispectrum, given the bias parameters. This computes both the theory and the derivatives with respect to linear parameters."""

                # Define local variables                                        
                kB, dkB = self.dataset.kB, self.dataset.dkB
                fNL_eq = self.fNL_eq
                fNL_orth = self.fNL_orth
                fz = self.fz
                nB = self.nB
                beta = fz/b1
                triangle_indices = self.dataset.triangle_indices

                # Pre-compute IR resummation quantities
                if not hasattr(self,'P_IR'):
                        self._load_IR_resummation(b1, c1)

                # Iterate over bispectrum bins and compute B0
                B0 = np.zeros(nB)
                deriv_PshotB, deriv_BshotB, deriv_c1B = [np.zeros(nB) for _ in range(3)]

                for j in range(int(nB)):
                        # Bin-centers
                        kc1, kc2, kc3 = kB[triangle_indices[0][j]], kB[triangle_indices[1][j]], kB[triangle_indices[2][j]]
                        # Bin-widths
                        dk1, dk2, dk3 = dkB, dkB, dkB
                        
                        # Check bin edges
                        if (kB[triangle_indices[0][j]]<dkB) or (kB[triangle_indices[1][j]]<dkB) or (kB[triangle_indices[2][j]]<dkB): 
                                raise Exception('Lowest bispectrum bin center is below than dk; alternative binning must be specified!')
                        
                        # Idealized bin volume
                        Nk123 = ((kc1+dk1/2.)**2.-(kc1-dk1/2.)**2.)*((kc2+dk2/2.)**2.-(kc2-dk2/2.)**2.)*((kc3+dk3/2.)**2.-(kc3-dk3/2.)**2.)/8.
                        
                        # Compute matrices
                        B0_matrix, deriv_Pshot_matrix, deriv_Bshot_matrix, deriv_c1_matrix = self._compute_B_matrices(beta,b1,b2,bG2,Pshot,Bshot,kc1,kc2,kc3,dk1,dk2,dk3,*self.mesh_mu)

                        # Integrate over bins to compute B0
                        B0[j] = self._bin_integrate(B0_matrix)/Nk123*self.dataset.discreteness_weights[j]
                        
                        # Update nuisance parameter covariance
                        deriv_PshotB[j] = self._bin_integrate(deriv_Pshot_matrix)/Nk123
                        deriv_BshotB[j] = self._bin_integrate(deriv_Bshot_matrix)/Nk123
                        deriv_c1B[j] = self._bin_integrate(deriv_c1_matrix)/Nk123

                return B0, deriv_PshotB, deriv_BshotB, deriv_c1B

