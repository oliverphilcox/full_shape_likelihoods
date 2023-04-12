#### A collection of classes used in the full_shape_spectra likelihood

import numpy as np
import os
from scipy import interpolate
import scipy.integrate as integrate
import h5py
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as sparse_inv   
from scipy.sparse.linalg import splu                               

class Datasets(object):
        def __init__(self, options):
                """Load Pk, Q0 and B0 data from file, as well as covariance matrix. The `options' argument is a dictionary of options specifying file names etc."""
                
                # Count number of redshift bins
                self.nz = options.nz
                
                # Load datasets
                if options.use_Q and not options.use_P:
                        raise Exception("Cannot use Q0 without power spectra!")
                if options.use_P:
                        self.load_power_spectrum(options)
                else:
                        self.nP = np.zeros(options.nz)
                        self.nQ = np.zeros(options.nz)
                if options.use_B:
                        self.load_bispectrum(options)
                        if options.oneloop_B:
                                self.initialize_oneloop_B(options)
                else:
                        self.nB = np.zeros(options.nz)
                if options.use_AP:
                        self.load_AP(options)
                else:
                        self.nAP = 0

                # Load covariance
                self.load_covariance(options)

        def load_power_spectrum(self, options):
                """Load power spectrum multipole dataset, optionally including Q0"""
                
                print("Loading power spectra...")
                
                # Load raw Pk measurements
                self.P0, self.P2, self.P4, self.Q0 = [],[],[],[]
                self.kPQ, self.dkPQ = [],[]
                self.nP_init, self.nPQ, self.nP, self.nQ,self.omitP,self.omitQ = [],[],[],[],[],[]
                
                for zi in range(self.nz):
                        data=np.loadtxt(os.path.join(options.data_directory, options.P_measurements[zi]), skiprows = 0, unpack=True)
                        k_init = data[0]
                        P0_init=data[1]
                        P2_init=data[2]
                        P4_init=data[3]
                        
                        # Count number of P bins (nP) and Q bins (nQ)
                        self.nP_init.append(len(k_init))
                        if options.use_Q:
                            self.nPQ.append(np.sum((k_init<options.kmaxQ[zi])&(k_init>=options.kminP[zi])))
                            self.nQ.append(np.sum((k_init<options.kmaxQ[zi])&(k_init>=options.kmaxP[zi])))
                            self.nP.append(self.nPQ[zi] - self.nQ[zi])         
                            self.omitP.append(np.sum((k_init<options.kminP[zi]))) # bins to omit at start of Pk array
                            self.omitQ.append(self.nP[zi] + self.omitP[zi]) # bins to omit at start of Q0 array
                        else:
                            self.nP.append(np.sum((k_init<options.kmaxP[zi])&(k_init>=options.kminP[zi])))
                            self.nPQ.append(self.nP[zi])
                            self.nQ.append(0)
                            self.omitP.append(np.sum((k_init<options.kminP[zi]))) # bins to omit at start of Pk array              
                        # Filter k and P_ell to correct bins
                        self.kPQ.append(k_init[self.omitP[zi]:self.omitP[zi]+self.nPQ[zi]])
                        self.dkPQ.append(self.kPQ[zi][1]-self.kPQ[zi][0]) # bin width
                        P0 = P0_init[self.omitP[zi]:self.omitP[zi]+self.nPQ[zi]]
                        P2 = P2_init[self.omitP[zi]:self.omitP[zi]+self.nPQ[zi]]
                        P4 = P4_init[self.omitP[zi]:self.omitP[zi]+self.nPQ[zi]]
                        
                        # Define data vectors
                        self.P0.append(P0[:self.nP[zi]])
                        self.P2.append(P2[:self.nP[zi]])
                        self.P4.append(P4[:self.nP[zi]])

                        # Compute Q0 from Pk0 measurements
                        if options.use_Q:
                                self.Q0.append(P0[self.nP[zi]:]-1./2.*P2[self.nP[zi]:]+3./8.*P4[self.nP[zi]:])

        def load_bispectrum(self, options):
                """Load bispectrum dataset."""
                
                print("Loading bispectra...")
                
                # Load discreteness weights and bispectra from file
                self.discreteness_weights, self.B0, self.nB = [],[],[]
                self.kB, self.dkB, self.triangle_indices = [],[],[]
                
                for zi in range(self.nz):
                        self.discreteness_weights.append(np.asarray(np.loadtxt(os.path.join(options.data_directory, options.discreteness_weights_file[zi]), dtype=np.float64)))
                        data = np.loadtxt(os.path.join(options.data_directory, options.B_measurements[zi]), dtype=np.float64, unpack=True)
                        khere, khere2, khere3 = data[:3]
                        self.B0.append(data[3])
                
                        assert len(self.B0[zi])==len(self.discreteness_weights[zi]), "Number of bispectra bins must match number of weights!"
                        self.nB.append(len(self.B0[zi]))

                        # 1D triangle centers
                        self.kB.append(np.arange(options.kminB[zi],options.kmaxB[zi],options.dkB[zi]))
                        self.dkB.append(self.kB[zi][1]-self.kB[zi][0])
                
                        # Indices labelling bispectrum bins
                        self.triangle_indices.append([np.asarray(kk/self.dkB[zi],dtype=int)-int(options.kminB[zi]/self.dkB[zi]) for kk in [khere, khere2, khere3]])

        def initialize_oneloop_B(self, options):
                """Initialize the one-loop bispectrum computation. The primary function of this is to load the interpolation grid from file."""

                print("Interpolating one-loop bispectra...")
                
                ## Load interpolation grid
                try:
                        infile = h5py.File(options.oneloop_shapes,'r')  
                except:
                        raise IOError('Could not open file %s'%options.oneloop_shapes)
                b222 = infile['222']
                b321I = infile['321I']
                b321II = infile['321II']
                b411 = infile['411']
                kk = infile['k']
                xx = infile['x']
                yy = infile['y']
                
                ## Define mesh of grid points
                kt = np.meshgrid(kk,xx)[0].T.ravel()
                xt = np.meshgrid(kk,xx)[1].T.ravel()
                yt = np.meshgrid(kk,yy)[1].T.ravel()
                kxy = np.vstack([kt,xt,yt]).T

                def Interpolate3D(bkTable):
                        """Reconstruct full 3D interpolation table for all biases and angles simultaneously. 
                        If this is used for vector k1,k2,k3, they should be *ordered* such that k1 > k2 > k3. This will *not* be checked at runtime.
                        This function will return NaNs if the inputs do not obey the triangle conditions.
                        """

                        bkVals = np.asarray([[bkTable[i,:,:,j].T.ravel() for i in range(len(bkTable))] for j in range(len(bkTable[0,0,0]))]).T
                        BkInt1 = interpolate.LinearNDInterpolator(kxy,bkVals)
                        
                        def BkInt(k1,k2,k3):
                                if type(k1)==np.ndarray:
                                        # Note no bounds-checking here!
                                        k = k1
                                        x = (k3/k1)**2;
                                        y = (k2/k1)**2;
                                        return BkInt1(k, x, y)
                                else:
                                        k123 = np.sort([k1,k2,k3])
                                        k = k123[2]
                                        x = (k123[0]/k)**2;
                                        y = (k123[1]/k)**2;
                                        return BkInt1(k, x, y)
                        return BkInt

                ## Define interpolators
                # These return a grid of all biases + mu coefficients simultaneously
                print("Loading bispectrum interpolators...\n")
                self.b222int = Interpolate3D(b222)
                self.b321Iint = Interpolate3D(b321I)
                self.b321IIint = Interpolate3D(b321II)
                self.b411int = Interpolate3D(b411)
        
        def load_AP(self, options):
                """Load Alcock-Paczynski dataset."""
                
                alphas = []
                for zi in range(self.nz):
                        alphas.append(np.loadtxt(os.path.join(options.data_directory, options.AP_measurements[zi]), dtype=np.float64))
                self.alphas = np.asarray(alphas)
                self.nAP = 2

        def load_covariance(self, options):
                """Load in the covariance matrix, filtered to the required bins and datasets [with the ordering P0, P2, P4, Q0, B0, AP]."""
                
                # Load full covariance matrix
                self.cov = []
                self.icov = []
                self.logdetcov = []
                print("Loading covariances...")
                for zi in range(self.nz):
                        cov1 = np.loadtxt(os.path.join(options.data_directory, options.covmat_file[zi]),dtype=np.float64)
                        
                        # Define which bins we use
                        filt = []
                        if options.use_P:
                                filt.append(np.arange(self.omitP[zi],self.omitP[zi]+self.nP[zi])) # P0
                                if options.lmax>0:
                                        filt.append(np.arange(self.omitP[zi]+self.nP_init[zi],self.omitP[zi]+self.nP_init[zi]+self.nP[zi])) # P2
                                if options.lmax>2:
                                        filt.append(np.arange(self.omitP[zi]+2*self.nP_init[zi],self.omitP[zi]+2*self.nP_init[zi]+self.nP[zi])) # P4
                        if options.use_Q:
                                filt.append(np.arange(self.omitQ[zi]+3*self.nP_init[zi],self.omitQ[zi]+3*self.nP_init[zi]+self.nQ[zi])) # Q0
                        if options.use_B:
                                filt.append(np.arange(4*self.nP_init[zi],4*self.nP_init[zi]+self.nB[zi])) # B0
                        if options.use_AP:
                                filt.append([-2,-1])
                        filt= np.concatenate(filt)

                        # Filter to the correct bins we want
                        cov = np.zeros((len(filt),len(filt)),dtype='float64')
                        for i,index in enumerate(filt):
                                for j,jndex in enumerate(filt):
                                        cov[i, j] = cov1[index, jndex]
                        self.cov.append(cov)

                        # Invert covariance (using sparse inverse for speed)
                        sparse_cov = csc_matrix(cov)
                        icov = sparse_inv(sparse_cov).toarray()
                        self.icov.append(icov)

                        # Compute matrix determinant (again sparsely)
                        lu = splu(sparse_cov)
                        detcov = (np.log(lu.L.diagonal().astype(np.complex128)).sum() + np.log(lu.U.diagonal().astype(np.complex128)).sum()).real
                        
                        self.logdetcov.append(detcov)

class PkTheory(object):
        def __init__(self, options, zi, all_theory, h, As, fNL_eq, fNL_orth, norm, fz, k_grid, kPQ, nP, nQ, Tk):
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
                self.dkPQ = options.dataset.dkPQ[zi]
                self.kminP = options.kminP[zi]
                self.kmaxP = options.kmaxP[zi]
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
                                kmin = self.dkPQ*i+self.kminP
                                kmax = self.dkPQ*(i+1)+self.kminP
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
                
                # Compute the power spectrum multipoles, including bin integration if requested
                if fNL_eq==0 and fNL_orth==0:
                        self.P0 = self.bin_integrator((norm**2.*all_theory[15]+norm**4.*(all_theory[21])+norm**1.*b1*all_theory[16]+norm**3.*b1*(all_theory[22]) + norm**0.*b1**2.*all_theory[17] +norm**2.*b1**2.*all_theory[23] + 0.25*norm**2.*b2**2.*all_theory[1] +b1*b2*norm**2.*all_theory[30]+ b2*norm**3.*all_theory[31] + b1*bG2*norm**2.*all_theory[32]+ bG2*norm**3.*all_theory[33] + b2*bG2*norm**2.*all_theory[4]+ bG2**2.*norm**2.*all_theory[5] + 2.*cs0*norm**2.*all_theory[11]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**2.*(b1*all_theory[7]+norm*all_theory[8]))*h**3. + (inv_nbar)*Pshot + a0*inv_nbar*(k_grid/0.5)**2.  + fz**2.*b4*k_grid**2.*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)*all_theory[13]*h + a2*(1./3.)*(10.**4.)*(k_grid/0.45)**2.)
                        
                        self.P2 = self.bin_integrator((norm**2.*all_theory[18]+norm**4.*(all_theory[24])+norm**1.*b1*all_theory[19]+norm**3.*b1*(all_theory[25]) + b1**2.*norm**2.*all_theory[26] +b1*b2*norm**2.*all_theory[34]+ b2*norm**3.*all_theory[35] + b1*bG2*norm**2.*all_theory[36]+ bG2*norm**3.*all_theory[37] + 0.25*b2**2.*all_theory[42] + b2*bG2*all_theory[43] + (bG2**2.)*all_theory[44] + 2.*cs2*norm**2.*all_theory[12]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**3.*all_theory[9])*h**3. + fz**2.*b4*k_grid**2.*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)*all_theory[13]*h + a2*(10.**4.)*(2./3.)*(k_grid/0.45)**2.)
                        self.P4 = self.bin_integrator((norm**2.*all_theory[20]+norm**4.*all_theory[27]+b1*norm**3.*all_theory[28]+b1**2.*norm**2.*all_theory[29] + b2*norm**3.*all_theory[38] + bG2*norm**3.*all_theory[39] + b1*b2*all_theory[40] + b1*bG2*all_theory[41] + 0.25*b2**2.*all_theory[45] + b2*bG2*all_theory[46] + (bG2**2.)*all_theory[46] +2.*cs4*norm**2.*all_theory[13]/h**2.)*h**3. + fz**2.*b4*k_grid**2.*(norm**2.*fz**2.*210./143. + 30.*fz*b1*norm/11.+b1**2.)*all_theory[13]*h)
                        
                else:
                        self.P0 = self.bin_integrator((norm**2.*all_theory[15]+norm**4.*(all_theory[21])+norm**1.*b1*all_theory[16]+norm**3.*b1*(all_theory[22]) + norm**0.*b1**2.*all_theory[17] +norm**2.*b1**2.*all_theory[23] + 0.25*norm**2.*b2**2.*all_theory[1] +b1*b2*norm**2.*all_theory[30]+ b2*norm**3.*all_theory[31] + b1*bG2*norm**2.*all_theory[32]+ bG2*norm**3.*all_theory[33] + b2*bG2*norm**2.*all_theory[4]+ bG2**2.*norm**2.*all_theory[5] + 2.*cs0*norm**2.*all_theory[11]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**2.*(b1*all_theory[7]+norm*all_theory[8]))*h**3. + (inv_nbar)*Pshot + a0*inv_nbar*(k_grid/0.5)**2.  + fz**2.*b4*k_grid**2.*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)*all_theory[13]*h + a2*(1./3.)*(10.**4.)*(k_grid/0.45)**2. + fNL_eq*(h**3.)*(all_theory[51]+b1*all_theory[52]+b1**2.*all_theory[53]+b1*b2*all_theory[60]+b2*all_theory[61]+b1*bG2*all_theory[62]+bG2*all_theory[63]) + 1.*(2.*b1*self.phif+self.phif**2.)*all_theory[17]*(h**3.) + 1.*self.phif*all_theory[16]*(h**3.) + fNL_orth*(h**3.)*(all_theory[75]+b1*all_theory[76]+b1**2.*all_theory[77]+b1*b2*all_theory[84]+b2*all_theory[85]+b1*bG2*all_theory[86]+bG2*all_theory[87]))
                        self.P2 = self.bin_integrator((norm**2.*all_theory[18]+norm**4.*(all_theory[24])+norm**1.*b1*all_theory[19]+norm**3.*b1*(all_theory[25]) + b1**2.*norm**2.*all_theory[26] +b1*b2*norm**2.*all_theory[34]+ b2*norm**3.*all_theory[35] + b1*bG2*norm**2.*all_theory[36]+ bG2*norm**3.*all_theory[37] + 0.25*b2**2.*all_theory[42] + b2*bG2*all_theory[43] + (bG2**2.)*all_theory[44] + 2.*cs2*norm**2.*all_theory[12]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**3.*all_theory[9])*h**3. + fz**2.*b4*k_grid**2.*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)*all_theory[13]*h + a2*(10.**4.)*(2./3.)*(k_grid/0.45)**2.+ fNL_eq*(h**3.)*(all_theory[54]+b1*all_theory[55]+b1**2.*all_theory[56]+b1*b2*all_theory[64]+b2*all_theory[65]+b1*bG2*all_theory[66]+bG2*all_theory[67]) + 1.*self.phif*all_theory[19]*(h**3.) + fNL_orth*(h**3.)*(all_theory[78]+b1*all_theory[79]+b1**2.*all_theory[80]+b1*b2*all_theory[88]+b2*all_theory[89]+b1*bG2*all_theory[90]+bG2*all_theory[91]))
                        self.P4 = self.bin_integrator((norm**2.*all_theory[20]+norm**4.*all_theory[27]+b1*norm**3.*all_theory[28]+b1**2.*norm**2.*all_theory[29] + b2*norm**3.*all_theory[38] + bG2*norm**3.*all_theory[39] + b1*b2*all_theory[40] + b1*bG2*all_theory[41] + 0.25*b2**2.*all_theory[45] + b2*bG2*all_theory[46] + (bG2**2.)*all_theory[46] +2.*cs4*norm**2.*all_theory[13]/h**2.)*h**3. + fz**2.*b4*k_grid**2.*(norm**2.*fz**2.*210./143. + 30.*fz*b1*norm/11.+b1**2.)*all_theory[13]*h+fNL_eq*(h**3.)*(all_theory[57]+b1*all_theory[58]+b1**2.*all_theory[59]+b1*b2*all_theory[68]+b2*all_theory[69]+b1*bG2*all_theory[70]+bG2*all_theory[71]) + fNL_orth*(h**3.)*(all_theory[81]+b1*all_theory[82]+b1**2.*all_theory[83]+b1*b2*all_theory[92]+b2*all_theory[93]+b1*bG2*all_theory[94]+bG2*all_theory[95]))
                
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
        def __init__(self, options, zi, As, fNL_eq, fNL_orth, apar, aperp, fz, sigma8, r_bao, k_grid, Tfunc, Pk_lin_table1, Pk_lin_table2, inv_nbar, gauss_w, gauss_w2, mesh_mu, nB):
                """Compute the theoretical power spectrum P(k) and parameter derivatives for a given cosmology and set of nuisance parameters."""
                
                # Load variables
                self.options = options
                self.dataset = options.dataset
                self.fNL_eq = fNL_eq
                self.fNL_orth = fNL_orth
                self.apar = apar
                self.aperp = aperp
                self.fz = fz
                self.sigma8 = sigma8
                self.r_bao = r_bao
                self.inv_nbar = inv_nbar
                self.gauss_w = gauss_w
                self.gauss_w2 = gauss_w2
                self.mesh_mu = mesh_mu
                self.nB = nB
                self.oneloop_B = options.oneloop_B
                self.k_grid = k_grid
                self.triangle_indices = options.dataset.triangle_indices[zi]
                self.kB = options.dataset.kB[zi]
                self.dkB = options.dataset.dkB[zi]
                
                # Load fNL variables
                self.Tfunc = Tfunc
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
        
        def _reorder_kmu(self,k1AP,k2AP,k3AP,mu1AP,mu2AP,mu3AP):
                """Switch k and mu vectors into the ordering with k1 >= k2 >= k3, as needed for the interpolators."""
                
                # Order k vectors
                args = np.argsort([k1AP,k2AP,k3AP],axis=0)

                # Define ordered set
                k1APord = np.zeros_like(k1AP)
                k2APord = np.zeros_like(k2AP)
                k3APord = np.zeros_like(k3AP)
                mu1APord = np.zeros_like(mu1AP)
                mu2APord = np.zeros_like(mu2AP)
                mu3APord = np.zeros_like(mu3AP)

                # biggest k
                k1APord[args[2]==0] = k1AP[args[2]==0]
                k1APord[args[2]==1] = k2AP[args[2]==1]
                k1APord[args[2]==2] = k3AP[args[2]==2]
                mu1APord[args[2]==0] = mu1AP[args[2]==0]
                mu1APord[args[2]==1] = mu2AP[args[2]==1]
                mu1APord[args[2]==2] = mu3AP[args[2]==2]

                # middle k
                k2APord[args[1]==0] = k1AP[args[1]==0]
                k2APord[args[1]==1] = k2AP[args[1]==1]
                k2APord[args[1]==2] = k3AP[args[1]==2]
                mu2APord[args[1]==0] = mu1AP[args[1]==0]
                mu2APord[args[1]==1] = mu2AP[args[1]==1]
                mu2APord[args[1]==2] = mu3AP[args[1]==2]

                # smallest k
                k3APord[args[0]==0] = k1AP[args[0]==0]
                k3APord[args[0]==1] = k2AP[args[0]==1]
                k3APord[args[0]==2] = k3AP[args[0]==2]
                mu3APord[args[0]==0] = mu1AP[args[0]==0]
                mu3APord[args[0]==1] = mu2AP[args[0]==1]
                mu3APord[args[0]==2] = mu3AP[args[0]==2]
                
                return k1APord, k2APord, k3APord, mu1APord, mu2APord, mu3APord
                
        def _compute_mu_vectors(self,mu1,mu2):
                """Compute a list of mu vectors for the one-loop bispectrum computation."""
                out = []
                for i in range(11):
                        for j in range(11):
                                if i+j>12: continue
                                if (i+j)%2==1: continue
                                out.append(mu1**i*mu2**j)
                return np.asarray(out)

        def _compute_B_matrices_tree(self,beta,b1,b2,bG2,Pshot,Bshot,kc1,kc2,kc3,dk1,dk2,dk3,k1,k2,k3,mu1,phi):
                """Load the tree-level bispectrum matrices for a given set of k bins. These will later be integrated over bins to form the bispectrum monopole and derivatives"""
                
                # Define local variables
                fNL_eq = self.fNL_eq
                fNL_orth = self.fNL_orth
                apar = self.apar
                aperp = self.aperp
                inv_nbar = self.inv_nbar

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

                # Add additional one-loop pieces
                if self.oneloop_B:
                        # Stochasticity
                        deriv_eps2_matrix = (qq1**2+qq2**2+qq3**2)*kk1*kk2*kk3*ddk1*ddk2*ddk3/apar**2./aperp**4.
                        deriv_eta21_matrix = (qq1**2*P_IR1+qq2**2*P_IR2+qq3**2*P_IR3)*kk1*kk2*kk3*ddk1*ddk2*ddk3/apar**2./aperp**4.
                        deriv_eta22_matrix = ((qq2**2+qq3**2)*P_IR1+(qq1**2+qq3**2)*P_IR2+(qq1**2+qq2**2)*P_IR3)*kk1*kk2*kk3*ddk1*ddk2*ddk3/apar**2./aperp**4.

                        # Counterterms / derivative operators
                        k12 = ((qq3**2-qq1**2-qq2**2)**2./(2.*qq1*qq2)**2.-1.)*P_IR1*P_IR2
                        k23 = ((qq1**2-qq2**2-qq3**2)**2./(2.*qq2*qq3)**2.-1.)*P_IR2*P_IR3
                        k31 = ((qq2**2-qq1**2-qq3**2)**2./(2.*qq3*qq1)**2.-1.)*P_IR3*P_IR1
                        deriv_betaBa_matrix = -((qq1**2+qq2**2)*zz21+(qq1**2+qq3**2)*zz22+(qq2**2+qq3**2)*zz23)*kk1*kk2*kk3*ddk1*ddk2*ddk3/apar**2./aperp**4.
                        deriv_betaBb_matrix = -(qq3**2*zz21+qq2**2*zz22+qq1**2*zz23)*kk1*kk2*kk3*ddk1*ddk2*ddk3/apar**2./aperp**4.
                        deriv_betaBc_matrix = -((qq1**2+qq2**2)*k12+(qq2**2+qq3**2)*k23+(qq1**2+qq3**2)*k31)*kk1*kk2*kk3*ddk1*ddk2*ddk3/apar**2./aperp**4.
                        deriv_betaBd_matrix = -(qq3**2*k12+qq1**2*k23+qq2**2*k31)*kk1*kk2*kk3*ddk1*ddk2*ddk3/apar**2./aperp**4.
                        deriv_betaBe_matrix = -((qq3**2-qq1**2-qq2**2)/2.*P_IR1*P_IR2+(qq1**2-qq2**2-qq3**2)/2.*P_IR2*P_IR3+(qq2**2-qq1**2-qq3**2)/2.*P_IR1*P_IR3)*kk1*kk2*kk3*ddk1*ddk2*ddk3/apar**2./aperp**4.
                        
                        return B0_matrix, deriv_Pshot_matrix, deriv_Bshot_matrix, deriv_c1_matrix, deriv_eps2_matrix, deriv_eta21_matrix, deriv_eta22_matrix, deriv_betaBa_matrix, deriv_betaBb_matrix, deriv_betaBc_matrix, deriv_betaBd_matrix, deriv_betaBe_matrix
                else:
                        return B0_matrix, deriv_Pshot_matrix, deriv_Bshot_matrix, deriv_c1_matrix

        def compute_B_oneloop(self,s8,b1,b2,b3,g2,g3,g21,g2x,g22,g21x,g31,g211,f,kc1,kc2,kc3,dk1,dk2,dk3,k1,k2,k3,mu1,phi):
                """Load the one-loop bispectrum for a given k-bin."""
   
                # Define local variables
                apar = self.apar
                aperp = self.aperp

                # Define lists of bias parameters
                biasListB222 = [b1**3,b1**2*b2,b1*b2**2,b2**3,b1**2*f,b1**3*f,b1*b2*f,b1**2*b2*f,b2**2*f,b1*b2**2*f,b1*f**2,b1**2*f**2,b1**3*f**2,b2*f**2,
                                b1*b2*f**2,b1**2*b2*f**2,b2**2*f**2,f**3,b1*f**3,b1**2*f**3,b1**3*f**3,b2*f**3,b1*b2*f**3,f**4,b1*f**4,b1**2*f**4,b2*f**4,f**5,
                                b1*f**5,f**6,b1**2*g2,b1*b2*g2,b2**2*g2,b1*f*g2,b1**2*f*g2,b2*f*g2,b1*b2*f*g2,f**2*g2,b1*f**2*g2,b1**2*f**2*g2,b2*f**2*g2,
                                f**3*g2,b1*f**3*g2,f**4*g2,b1*g2**2,b2*g2**2,f*g2**2,b1*f*g2**2,f**2*g2**2,g2**3]
                biasListB321I = [b1**3,b1**2*b2,b1*b2**2,b1**2*b3,b1*b2*b3,b1**2*f,b1**3*f,b1*b2*f,b1**2*b2*f,b2**2*f,b1*b2**2*f,b1*b3*f,b1**2*b3*f,b2*b3*f,
                                b1*f**2,b1**2*f**2,b1**3*f**2,b2*f**2,b1*b2*f**2,b1**2*b2*f**2,b2**2*f**2,b3*f**2,b1*b3*f**2,f**3,b1*f**3,b1**2*f**3,b1**3*f**3,
                                b2*f**3,b1*b2*f**3,b3*f**3,f**4,b1*f**4,b1**2*f**4,b2*f**4,f**5,b1*f**5,f**6,b1**2*g2,b1*b2*g2,b1*b3*g2,b1*f*g2,b1**2*f*g2,
                                b2*f*g2,b1*b2*f*g2,b3*f*g2,f**2*g2,b1*f**2*g2,b1**2*f**2*g2,b2*f**2*g2,f**3*g2,b1*f**3*g2,f**4*g2,b1*g2**2,f*g2**2,b1*f*g2**2,
                                f**2*g2**2,b1**2*g21,b1*b2*g21,b1*f*g21,b1**2*f*g21,b2*f*g21,f**2*g21,b1*f**2*g21,f**3*g21,b1*g2*g21,f*g2*g21,b1**2*g2x,
                                b1*b2*g2x,b1*f*g2x,b1**2*f*g2x,b2*f*g2x,f**2*g2x,b1*f**2*g2x,f**3*g2x,b1*g2*g2x,f*g2*g2x,b1**2*g3,b1*b2*g3,b1*f*g3,b1**2*f*g3,
                                b2*f*g3,f**2*g3,b1*f**2*g3,f**3*g3,b1*g2*g3,f*g2*g3]
                biasListB321II = [b1**3,b1**2*b2,b1**2*f,b1**3*f,b1*b2*f,b1*f**2,b1**2*f**2,b1**3*f**2,b2*f**2,b1*b2*f**2,b1**2*b2*f**2,f**3,b1*f**3,
                                b1**2*f**3,b1**3*f**3,b2*f**3,b1*b2*f**3,f**4,b1*f**4,b1**2*f**4,b2*f**4,f**5,b1*f**5,f**6,b1**2*g2,b1*b2*g2,b1*f*g2,b1**2*f*g2,
                                b2*f*g2,f**2*g2,b1*f**2*g2,b1**2*f**2*g2,f**3*g2,b1*f**3*g2,f**4*g2,b1*g2**2,f*g2**2,b1**2*g21,b1*b2*g21,b1*f*g21,b1**2*f*g21,
                                b2*f*g21,f**2*g21,b1*f**2*g21,f**3*g21,b1*g2*g21,f*g2*g21]
                biasListB411 = [b1**3,b1**2*b2,b1**2*b3,b1**2*f,b1**3*f,b1*b2*f,b1**2*b2*f,b1*b3*f,b1**2*b3*f,b1*f**2,b1**2*f**2,b1**3*f**2,b2*f**2,
                                b1*b2*f**2,b1**2*b2*f**2,b3*f**2,b1*b3*f**2,f**3,b1*f**3,b1**2*f**3,b1**3*f**3,b2*f**3,b1*b2*f**3,b3*f**3,f**4,b1*f**4,
                                b1**2*f**4,b2*f**4,f**5,b1*f**5,f**6,b1**2*g2,b1*f*g2,b1**2*f*g2,f**2*g2,b1*f**2*g2,b1**2*f**2*g2,f**3*g2,b1*f**3*g2,f**4*g2,
                                b1**2*g21,b1*f*g21,b1**2*f*g21,f**2*g21,b1*f**2*g21,f**3*g21,b1**2*g211,b1*f*g211,f**2*g211,b1**2*g21x,b1*f*g21x,f**2*g21x,
                                b1**2*g22,b1*f*g22,f**2*g22,b1**2*g2x,b1*f*g2x,b1**2*f*g2x,f**2*g2x,b1*f**2*g2x,f**3*g2x,b1**2*g3,b1*f*g3,b1**2*f*g3,f**2*g3,
                                b1*f**2*g3,f**3*g3,b1**2*g31,b1*f*g31,f**2*g31]

                # Define derivatives with respect to bias parameters
                biasListB321I_b3 = [0,0,0,b1**2,b1*b2,0,0,0,0,0,0,b1*f,b1**2*f,b2*f,0,0,0,0,0,0,0,f**2,b1*f**2,0,0,0,0,0,0,f**3,0,0,0,0,0,0,0,0,0,b1*g2,0,0,0,0,f*g2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                biasListB411_b3 = [0,0,b1**2,0,0,0,0,b1*f,b1**2*f,0,0,0,0,0,0,f**2,b1*f**2,0,0,0,0,0,0,f**3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                
                biasListB321I_g3 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,b1**2,b1*b2,b1*f,b1**2*f,b2*f,f**2,b1*f**2,f**3,b1*g2,f*g2]
                biasListB411_g3 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,b1**2,b1*f,b1**2*f,f**2,b1*f**2,f**3,0,0,0]

                biasListB321I_g21 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,b1**2,b1*b2,b1*f,b1**2*f,b2*f,f**2,b1*f**2,f**3,b1*g2,f*g2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                biasListB321II_g21 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,b1**2,b1*b2,b1*f,b1**2*f,b2*f,f**2,b1*f**2,f**3,b1*g2,f*g2]
                biasListB411_g21 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,b1**2,b1*f,b1**2*f,f**2,b1*f**2,f**3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

                biasListB321I_g2x = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,b1**2,b1*b2,b1*f,b1**2*f,b2*f,f**2,b1*f**2,f**3,b1*g2,f*g2,0,0,0,0,0,0,0,0,0,0]
                biasListB411_g2x = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,b1**2,b1*f,b1**2*f,f**2,b1*f**2,f**3,0,0,0,0,0,0,0,0,0] 

                biasListB411_g22 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,b1**2,b1*f,f**2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                
                biasListB411_g21x = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,b1**2,b1*f,f**2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

                biasListB411_g31 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,b1**2,b1*f,f**2]

                biasListB411_g211 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,b1**2,b1*f,f**2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

                # Define bin parameters
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

                ## Define new angles
                shape = np.ones(nnu2.shape)
                k1AP = (kk1*qq1)*shape
                k2AP = (kk2*qq2)*shape
                k3AP = (kk3*qq3)*shape
                mu1AP = nnu1*shape
                mu2AP = nnu2*shape
                mu3AP = nnu3*shape
                                
                # Permute to get correct ordering               
                k1APord, k2APord, k3APord, mu1APord, mu2APord, mu3APord = self._reorder_kmu(k1AP,k2AP,k3AP,mu1AP,mu2AP,mu3AP)                                

                # Define mu array
                muVec = np.moveaxis(self._compute_mu_vectors(mu1APord,mu2APord),0,-1)#[:,:,:,:,:,:,None]
                
                # Define volume element
                volEl = (kk1*kk2*kk3*ddk1*ddk2*ddk3*shape)

                def sum_mu(interp):
                        """Sum the interpolated quantities over the mu bins"""
                        return np.sum(muVec*interp,axis=5)
    
                # Perform the (vectorized) interpolation, including all angular components and biases
                b222int = self.dataset.b222int(k1APord,k2APord,k3APord)
                b321Iint = self.dataset.b321Iint(k1APord,k2APord,k3APord)
                b321IIint = self.dataset.b321IIint(k1APord,k2APord,k3APord)
                b411int = self.dataset.b411int(k1APord,k2APord,k3APord)
                
                b_matrix = sum_mu(np.matmul(b222int,biasListB222))
                b_matrix += sum_mu(np.matmul(b321Iint,biasListB321I))
                b_matrix += sum_mu(np.matmul(b321IIint,biasListB321II))
                b_matrix += sum_mu(np.matmul(b411int,biasListB411))
                
                # Contribution bias derivatives
                b_matrix_b3 = sum_mu(np.matmul(b321Iint,biasListB321I_b3))+sum_mu(np.matmul(b411int,biasListB411_b3))
                b_matrix_g3 = sum_mu(np.matmul(b321Iint,biasListB321I_g3))+sum_mu(np.matmul(b411int,biasListB411_g3))
                b_matrix_g21 = sum_mu(np.matmul(b321Iint,biasListB321I_g21))+sum_mu(np.matmul(b321IIint,biasListB321II_g21))+sum_mu(np.matmul(b411int,biasListB411_g21))
                b_matrix_g2x = sum_mu(np.matmul(b321Iint,biasListB321I_g2x))+sum_mu(np.matmul(b411int,biasListB411_g2x))
                b_matrix_g22 = sum_mu(np.matmul(b411int,biasListB411_g22))
                b_matrix_g21x = sum_mu(np.matmul(b411int,biasListB411_g21x))
                b_matrix_g31 = sum_mu(np.matmul(b411int,biasListB411_g31))
                b_matrix_g211 = sum_mu(np.matmul(b411int,biasListB411_g211))

                # Create list of derivatives
                b_matrix_derivs = [b_matrix_b3,b_matrix_g3,b_matrix_g21,b_matrix_g2x,b_matrix_g22,b_matrix_g21x,b_matrix_g31,b_matrix_g211]

                # Filter out elements that do not obey triangle conditions 
                tol = 0. # 0.0025
                triangle_conditions = ((np.abs(k1APord-k2APord)<=k3APord-tol)&(k3APord-tol<=(k1APord+k2APord)))
                b_matrix = np.where(triangle_conditions,b_matrix,0.)
                b_matrix_derivs = [np.where(triangle_conditions,b_matrix_derivs[i],0.) for i in range(len(b_matrix_derivs))]
                
                # Define bin volume (using only good triangles)
                Nk123 = self._bin_integrate(volEl*triangle_conditions)

                # Compute bin-integrated spectrum and normalize
                b_integ = s8**6*self._bin_integrate(b_matrix*volEl*triangle_conditions)/Nk123/apar**2/aperp**4
                b_integ_derivs = [s8**6*self._bin_integrate(b_matrix_derivs[i]*volEl*triangle_conditions)/Nk123/apar**2/aperp**4 for i in range(len(b_matrix_derivs))]

                return b_integ, b_integ_derivs

        def compute_B0_theory_derivs(self, bias_list):
                """Compute the bispectrum (at tree- or one-loop order), given the bias parameters. This computes both the theory and the derivatives with respect to linear parameters."""

                # Define local variables                                        
                kB, dkB = self.kB, self.dkB
                fNL_eq = self.fNL_eq
                fNL_orth = self.fNL_orth
                fz = self.fz
                nB = self.nB
                if self.oneloop_B:
                        # Ratio of sigma8 to rescale fiducial 1-loop templates
                        s8 = self.sigma8/self.options.sigma8_fid
                
                triangle_indices = self.triangle_indices

                import time
                start = time.time()

                # Load in bias parameters
                if not self.oneloop_B:
                        b1, b2, bG2, c1, Pshot, Bshot = bias_list
                else:
                        b1, b2, bG2, bGamma3, b3, g3, g2x, g22, g21x, g31, g211, c1, Pshot, Bshot, eps2, eta21, eta22, betaBa, betaBb, betaBc, betaBd, betaBe = bias_list
                        # Redefine to Eggemeier conventions
                        g2 = bG2
                        g21 = -4./7.*(bG2+bGamma3)
                beta = fz/b1
                
                # Pre-compute IR resummation quantities
                if not hasattr(self,'P_IR'):
                        self._load_IR_resummation(b1, c1)

                # Iterate over bispectrum bins and compute B0
                B0 = np.zeros(nB)
                deriv_PshotB, deriv_BshotB, deriv_c1B = [np.zeros(nB) for _ in range(3)]
                if self.oneloop_B:
                        deriv_b3, deriv_g3, deriv_g21, deriv_g2x, deriv_g22, deriv_g21x, deriv_g31, deriv_g211, deriv_eps2, deriv_eta21, deriv_eta22, deriv_betaBa, deriv_betaBb, deriv_betaBc, deriv_betaBd, deriv_betaBe = [np.zeros(nB) for _ in range(16)]

                for j in range(int(nB)):
                        # Bin-centers
                        kc1, kc2, kc3 = kB[triangle_indices[0][j]], kB[triangle_indices[1][j]], kB[triangle_indices[2][j]]
                        # Bin-widths
                        dk1, dk2, dk3 = dkB, dkB, dkB
                        
                        # Check bin edges
                        if (kB[triangle_indices[0][j]]<dkB) or (kB[triangle_indices[1][j]]<dkB) or (kB[triangle_indices[2][j]]<dkB): 
                                raise Exception('Lowest bispectrum bin center is below dk; alternative binning must be specified!')
                        
                        # Idealized bin volume
                        Nk123 = ((kc1+dk1/2.)**2.-(kc1-dk1/2.)**2.)*((kc2+dk2/2.)**2.-(kc2-dk2/2.)**2.)*((kc3+dk3/2.)**2.-(kc3-dk3/2.)**2.)/8.
                        
                        # Compute matrices
                        if self.oneloop_B:
                                B0_matrix_tree, deriv_Pshot_matrix_tree, deriv_Bshot_matrix_tree, deriv_c1_matrix_tree, deriv_eps2_matrix, deriv_eta21_matrix, deriv_eta22_matrix, deriv_betaBa_matrix, deriv_betaBb_matrix, deriv_betaBc_matrix, deriv_betaBd_matrix, deriv_betaBe_matrix = self._compute_B_matrices_tree(beta,b1,b2,bG2,Pshot,Bshot,kc1,kc2,kc3,dk1,dk2,dk3,*self.mesh_mu)
                        else:
                                B0_matrix_tree, deriv_Pshot_matrix_tree, deriv_Bshot_matrix_tree, deriv_c1_matrix_tree = self._compute_B_matrices_tree(beta,b1,b2,bG2,Pshot,Bshot,kc1,kc2,kc3,dk1,dk2,dk3,*self.mesh_mu)
                        
                        # Integrate over bins to compute B0
                        B0[j] = self._bin_integrate(B0_matrix_tree)/Nk123
                        
                        # Update nuisance parameter covariance
                        deriv_PshotB[j] = self._bin_integrate(deriv_Pshot_matrix_tree)/Nk123
                        deriv_BshotB[j] = self._bin_integrate(deriv_Bshot_matrix_tree)/Nk123
                        deriv_c1B[j] = self._bin_integrate(deriv_c1_matrix_tree)/Nk123

                        if self.oneloop_B:
                                B0_oneloop, B0_oneloop_derivs = self.compute_B_oneloop(s8,b1,b2,b3,g2,g3,g21,g2x,g22,g21x,g31,g211,fz,kc1,kc2,kc3,dk1,dk2,dk3,*self.mesh_mu)
                                B0[j] += B0_oneloop
                                # Also add on the signal from counterterms and stochasticity
                                B0[j] += self._bin_integrate(deriv_eps2_matrix*eps2+deriv_eta21_matrix*eta21+deriv_eta22_matrix*eta22+deriv_betaBa_matrix*betaBa+deriv_betaBb_matrix*betaBb+deriv_betaBc_matrix*betaBc+deriv_betaBd_matrix*betaBd+deriv_betaBe_matrix*betaBe)/Nk123
                                # Assemble derivatives
                                deriv_b3[j] = B0_oneloop_derivs[0]
                                deriv_g3[j] = B0_oneloop_derivs[1]
                                deriv_g21[j] = B0_oneloop_derivs[2]
                                deriv_g2x[j] = B0_oneloop_derivs[3]
                                deriv_g22[j] = B0_oneloop_derivs[4]
                                deriv_g21x[j] = B0_oneloop_derivs[5]
                                deriv_g31[j] = B0_oneloop_derivs[6]
                                deriv_g211[j] = B0_oneloop_derivs[7]
                                deriv_eps2[j] = self._bin_integrate(deriv_eps2_matrix)/Nk123
                                deriv_eta21[j] = self._bin_integrate(deriv_eta21_matrix)/Nk123
                                deriv_eta22[j] = self._bin_integrate(deriv_eta22_matrix)/Nk123
                                deriv_betaBa[j] = self._bin_integrate(deriv_betaBa_matrix)/Nk123
                                deriv_betaBb[j] = self._bin_integrate(deriv_betaBb_matrix)/Nk123
                                deriv_betaBc[j] = self._bin_integrate(deriv_betaBc_matrix)/Nk123
                                deriv_betaBd[j] = self._bin_integrate(deriv_betaBd_matrix)/Nk123
                                deriv_betaBe[j] = self._bin_integrate(deriv_betaBe_matrix)/Nk123
                        
                        
                if self.oneloop_B:
                        derivs = [deriv_PshotB, deriv_BshotB, deriv_c1B, deriv_b3, deriv_g3, deriv_g21, deriv_g2x, deriv_g22, deriv_g21x, deriv_g31, deriv_g211, deriv_eps2, deriv_eta21, deriv_eta22, deriv_betaBa, deriv_betaBb, deriv_betaBc, deriv_betaBd, deriv_betaBe]
                else:
                        derivs = [deriv_PshotB, deriv_BshotB, deriv_c1B]

                return B0, derivs

