#### A collection of classes used in the full_shape_spectra likelihood

import numpy as np
import os
from scipy import interpolate
import scipy.integrate as integrate
import h5py
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as sparse_inv
from scipy.sparse.linalg import splu
from scipy.special import legendre

class Datasets(object):
        def __init__(self, options):
                """Load Pk, Q0 and B0 data from file, as well as covariance matrix. The `options' argument is a dictionary of options specifying file names etc."""

                # Count number of redshift bins
                self.nz = options.nz

                # Check lmax
                assert options.lmaxP in [0,2,4]
                assert options.lmaxB in [0,2,4]
                self.nlP = options.lmaxP//2+1
                self.nlB = options.lmaxB//2+1

                # Load datasets
                if options.use_Q and not options.use_P:
                        raise Exception("Cannot use Q0 without power spectra!")
                if options.use_P or options.use_B:
                        self.load_power_spectrum(options) # load to get dimensions!
                if not options.use_P:
                        self.nP = np.zeros(options.nz)
                        self.nQ = np.zeros(options.nz)
                if options.use_B:
                        self.load_bispectrum(options)
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

                Pl_init = []
                self.kPQ, self.dkPQ = [],[]
                self.nP_init, self.nPQ, self.nP, self.nQ,self.omitP,self.omitQ = [],[],[],[],[],[]

                for zi in range(self.nz):
                        data=np.loadtxt(os.path.join(options.data_directory, options.P_measurements[zi]), skiprows = 0, unpack=True)
                        k_init = data[0]
                        Pl_init.append(data[1:4].T)
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

                Pl_init = np.asarray(Pl_init)

                # Filter k and P_ell to correct bins
                Pl = np.asarray([Pl_init[zi,self.omitP[zi]:self.omitP[zi]+self.nPQ[zi],:] for zi in range(self.nz)])

                # Define data vectors
                self.Pl = np.asarray([Pl[zi,:self.nP[zi],:self.nlP] for zi in range(self.nz)])

                # Compute Q0 from Pk0 measurements
                if options.use_Q:
                        self.Q0 = np.asarray([Pl[zi,self.nP[zi]:,0]-1./2.*Pl[zi,self.nP[zi]:,1]+3./8.*Pl[zi,self.nP[zi]:,2] for zi in range(self.nz)])

        def load_bispectrum(self, options):
                """Load bispectrum dataset."""

                # Load discreteness weights and bispectra from file
                self.discreteness_weights, Bl, self.nB = [],[],[]
                self.kB, self.dkB, self.triangle_indices = [],[],[]

                for zi in range(self.nz):
                        self.discreteness_weights.append(np.asarray(np.loadtxt(os.path.join(options.data_directory, options.discreteness_weights_file[zi]), dtype=np.float64)))
                        data = np.loadtxt(os.path.join(options.data_directory, options.B_measurements[zi]), dtype=np.float64, unpack=True)
                        khere, khere2, khere3 = data[:3]

                        assert np.shape(data[3:].T)==np.shape(self.discreteness_weights[zi]), "Number of bispectrum multipole bins must match number of weights!"
                        self.nB.append(np.shape(self.discreteness_weights[zi])[0])

                        # Define bispectrum multipoles
                        Bl.append(data[3:].T)

                        # 1D triangle centers
                        self.kB.append(np.arange(options.kminB[zi],options.kmaxB[zi]+1.e-12,options.dkB[zi]))
                        self.dkB.append(self.kB[zi][1]-self.kB[zi][0])

                        # Indices labelling bispectrum bins
                        self.triangle_indices.append([np.asarray(kk/self.dkB[zi],dtype=int)-int(options.kminB[zi]/self.dkB[zi]) for kk in [khere, khere2, khere3]])

                # Define data vectors
                self.Bl = np.asarray(Bl)[:,:,:self.nlB]

        def load_AP(self, options):
                """Load Alcock-Paczynski dataset."""

                alphas = []
                for zi in range(self.nz):
                        alphas.append(np.loadtxt(os.path.join(options.data_directory, options.AP_measurements[zi]), dtype=np.float64))
                self.alphas = np.asarray(alphas)
                self.nAP = 2

        def load_covariance(self, options):
                """Load in the covariance matrix, filtered to the required bins and datasets [with the ordering P0, P2, P4, Q0, B0, B2, B4, AP]."""

                # Load full covariance matrix
                self.cov = []
                self.icov = []
                #self.logdetcov = []
                print("Loading covariances...")
                for zi in range(self.nz):
                        cov1 = np.loadtxt(os.path.join(options.data_directory, options.covmat_file[zi]),dtype=np.float64)

                        # Define which bins we use
                        filt = []
                        if options.use_P:
                                filt.append(np.arange(self.omitP[zi],self.omitP[zi]+self.nP[zi])) # P0
                                if options.lmaxP>0:
                                        filt.append(np.arange(self.omitP[zi]+self.nP_init[zi],self.omitP[zi]+self.nP_init[zi]+self.nP[zi])) # P2
                                if options.lmaxP>2:
                                        filt.append(np.arange(self.omitP[zi]+2*self.nP_init[zi],self.omitP[zi]+2*self.nP_init[zi]+self.nP[zi])) # P4
                        if options.use_Q:
                                filt.append(np.arange(self.omitQ[zi]+3*self.nP_init[zi],self.omitQ[zi]+3*self.nP_init[zi]+self.nQ[zi])) # Q0
                        if options.use_B:
                                filt.append(np.arange(4*self.nP_init[zi],4*self.nP_init[zi]+self.nB[zi])) # B0
                                if options.lmaxB>0:
                                        filt.append(np.arange(4*self.nP_init[zi]+self.nB[zi],4*self.nP_init[zi]+2*self.nB[zi])) # B2
                                if options.lmaxB>2:
                                        filt.append(np.arange(4*self.nP_init[zi]+2*self.nB[zi],4*self.nP_init[zi]+3*self.nB[zi])) # B4
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
                        #lu = splu(sparse_cov)
                        #detcov = (np.log(lu.L.diagonal().astype(np.complex128)).sum() + np.log(lu.U.diagonal().astype(np.complex128)).sum()).real

                        #self.logdetcov.append(detcov)

        def _load_collider_power_spectra(self, options):
            """Load interpolation functions for the cosmological collider power spectra"""

            # List of quantities to interpolate
            columns = ['v_v_0','v_d_0','d_d_0','b2over2_0','bG2_0','b1_b2over2_0','b1_bG2_0',
                       'v_v_2','v_d_2','d_d_2','b2over2_2','bG2_2','b1_b2over2_2','b1_bG2_2',
                       'v_v_4','v_d_4','d_d_4','b2over2_4','bG2_4','b1_b2over2_4','b1_bG2_4']

            def load_interpolators(infile_name):

                # Load file and coordinate grids
                infile  = h5py.File(infile_name,'r')
                k_list  = np.asarray(infile['x'])
                mu_list = np.asarray(infile['y'])
                cs_list = np.asarray(infile['z'])

                # Create a stack of all the interpolators and return
                stack_dat = np.stack([np.asarray(infile[column]) for column in columns],axis=-1)
                return interpolate.RegularGridInterpolator((k_list, mu_list, cs_list), stack_dat, method="linear", bounds_error=False, fill_value = 0.)

            # Load interpolators for each redshift
            self.pk_dotpi2_interp = {}
            self.pk_nablapi2_interp = {}
            for zi,z in enumerate(options.collider_pk_redshifts):
                self.pk_dotpi2_interp[z] = load_interpolators(options.collider_pk_dotpi2[zi])
                self.pk_nablapi2_interp[z] = load_interpolators(options.collider_pk_nablapi2[zi])

        def _load_collider_shapes(self, options):
            """Load interpolation functions for the cosmological collider bispectra"""

            # Load file and coordinate grids
            infile  = h5py.File(options.collider_shapes,'r')
            u_list  = np.asarray(infile['x'])
            mu_list = np.asarray(infile['y'])
            cs_list = np.asarray(infile['z'])

            # Create a stack of all the interpolators and create function which outputs only the two combinations we need
            stack_dat = np.stack([np.asarray(infile['I'])+1./u_list[:,None,None]*np.asarray(infile['u2dI']),
                                  np.asarray(infile['u4ddI'])+2.0*np.asarray(infile['u3dI'])],
                                 axis=-1)
            interp = interpolate.RegularGridInterpolator((u_list, mu_list, cs_list), stack_dat, method="linear", bounds_error=True)

            # Define function for bispectrum shapes and normalization
            def collider_shapes(k2_k1,k3_k1,mu,cs):

                sq_1 = (1.0-k3_k1**2-k2_k1**2)
                sq_2 = (k3_k1**2-k2_k1**2-1.0)
                sq_3 = (k2_k1**2-1.0-k3_k1**2)
                cs2 = cs**2

                # Run interpolators
                shape = np.zeros_like(k3_k1)
                interp_1 = interp(np.stack([1.0/(cs*(k3_k1+k2_k1)),mu+shape,cs+shape],axis=-1))
                interp_2 = interp(np.stack([k3_k1/(cs*(k2_k1+1.0)),mu+shape,cs+shape],axis=-1))
                interp_3 = interp(np.stack([k2_k1/(cs*(1.0+k3_k1)),mu+shape,cs+shape],axis=-1))

                # dotpi2 shape
                dot =  k3_k1*k2_k1*interp_1[...,1]
                dot += k2_k1/(k3_k1**2)*interp_2[...,1]
                dot += k3_k1/(k2_k1**2)*interp_3[...,1]

                # nablapi2 shape
                nabla =  sq_1*cs2*interp_1[...,1]
                nabla += sq_2*cs2/(k3_k1**2)*interp_2[...,1]
                nabla += sq_3*cs2/(k2_k1**2)*interp_3[...,1]
                nabla += sq_1/(k3_k1*k2_k1)*interp_1[...,0]
                nabla += sq_2/k2_k1*interp_2[...,0]
                nabla += sq_3/k3_k1*interp_3[...,0]

                return dot, nabla

            # Define function for normalization
            def collider_norm(mu,cs):

                # Run interpolators
                interp_1 = interp([1.0/(2.*cs),mu,cs])[0]

                # Return normalized shapes
                return 3*interp_1[1]*(-10./9.)*cs, (-3*cs**2*interp_1[1]-3*interp_1[0])*(5./9.)/cs

            self.collider_shapes = collider_shapes
            self.collider_norm = collider_norm

class PkTheory(object):
        def __init__(self, options, all_theory, h, As, zi, z, ng_params, norm, fz, k_grid, Tk):
                """Compute the theoretical power spectrum P(k) and parameter derivatives for a given cosmology and set of non-Gaussianity and nuisance parameters."""

                # Read in input parameters
                self.all_theory = all_theory
                self.h = h
                self.As = As
                self.norm = norm
                self.k_grid = k_grid
                self.fz = fz
                self.kPQ = options.dataset.kPQ[zi]
                self.dkPQ = options.dataset.dkPQ[zi]
                self.kminP = options.kminP[zi]
                self.kmaxP = options.kmaxP[zi]
                self.nP = options.dataset.nP[zi]
                self.nQ = options.dataset.nQ[zi]
                self.nlP = options.dataset.nlP
                self.Tk = Tk
                self.options = options
                self.use_eq_orth = options.use_eq_orth
                self.use_collider = options.use_collider
                self.dataset = options.dataset

                # Non-Gaussianity parameters
                if self.use_eq_orth:
                    self.fNL_eq = ng_params['fNL_eq']
                    self.fNL_orth = ng_params['fNL_orth']
                if self.use_collider:
                    self.beta_dotpi2 = ng_params['beta_dotpi2']
                    self.beta_nablapi2 = ng_params['beta_nablapi2']
                    self.mu = ng_params['mu']
                    self.cs = ng_params['cs']

                # Collider interpolating functions
                if self.use_collider:
                    assert z in options.collider_pk_redshifts, "Requested redshift %.2f has not been computed for the collider!"%z

                    # Define collider normalization
                    self.norm_count_dotpi2, self.norm_count_nablapi2 = self.dataset.collider_norm(self.mu,self.cs)

                    # Assemble collider grids
                    self.Pl_dotpi2 = self.norm_count_dotpi2*np.reshape(self._Pl_dotpi2(z,k_grid*h,self.mu,self.cs),(len(k_grid),3,7))
                    self.Pl_nablapi2 = self.norm_count_nablapi2*np.reshape(self._Pl_nablapi2(z,k_grid*h,self.mu,self.cs),(len(k_grid),3,7))

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

        def compute_Pl_oneloop(self, biases):
                """Compute the 1-loop power spectrum multipoles, given the bias parameters."""

                # Run the main code
                if not hasattr(self, 'P0'):
                        self._load_P_oneloop_all(biases)

                # Extract the power spectrum multipoles
                Pl = self.Pl[:self.nP,:self.nlP]

                return Pl

        def compute_Q0_oneloop(self, biases):
                """Compute the 1-loop Q0 theory, given the bias parameters."""

                # Run the main code
                if not hasattr(self, 'P0'):
                        self._load_P_oneloop_all(biases)

                # Extract Q0
                Q0 = self.Pl[self.nP:,0]-1./2.*self.Pl[self.nP:,1]+3./8.*self.Pl[self.nP:,2]

                return Q0

        def _Pl_dotpi2(self,z,k_in_1_over_Mpc,mu,cs):
            """Compute P_ell(k) for the collider dot(pi)^2 shape for all k at some (precomputed) redshift, z"""
            return self.dataset.pk_dotpi2_interp[z](np.asarray([k_in_1_over_Mpc, mu*np.ones(len(k_in_1_over_Mpc)), cs*np.ones(len(k_in_1_over_Mpc))]).T)

        def _Pl_nablapi2(self,z,k_in_1_over_Mpc,mu,cs):
            """Compute P_ell(k) for the collider nabla(pi)^2 shape for all k at some (precomputed) redshift, z"""
            return self.dataset.pk_nablapi2_interp[z](np.asarray([k_in_1_over_Mpc, mu*np.ones(len(k_in_1_over_Mpc)), cs*np.ones(len(k_in_1_over_Mpc))]).T)

        def _load_P_oneloop_all(self, biases):
                """Internal function to compute the 1-loop power spectrum multipoles for all k, given the bias parameters. Note that we always use lmax = 4 here, since higher-multipoles are needed for Q0."""

                # Load variables
                all_theory = self.all_theory
                norm = self.norm
                h = self.h
                fz = self.fz
                k_grid = self.k_grid

                # Load biases for simplicity
                b1 = biases['b1']
                b2 = biases['b2']
                bG2 = biases['bG2']
                bGamma3 = biases['bGamma3']
                cs0 = biases['cs0']
                cs2 = biases['cs2']
                cs4 = biases['cs4']
                b4 = biases['b4']
                a0 = biases['a0']
                a2 = biases['a2']
                Pshot = biases['Pshot']
                inv_nbar = biases['inv_nbar']

                # Load non-Gaussianity bias if needed
                if self.use_eq_orth:
                    bphi = biases['bphi']
                if self.use_collider:
                    bphi_coll_1 = biases['bphi_coll_1']
                    bphi_coll_2 = biases['bphi_coll_2']
                    # mu is needed as well for counterterm
                    mu = self.mu

                # Compute fNL factors
                if not hasattr(self, 'phif') and self.use_eq_orth:
                    prefactor = (18./5.)*(b1-1.)*1.686/self.Tk
                    self.phif1 = (self.fNL_eq+self.fNL_orth)*prefactor*(k_grid/0.45)**2.

                if not hasattr(self, 'phifcoll1_1') and self.use_collider:
                    prefactor = (18./5.)*(b1-1.)*1.686/self.Tk
                    fNL_val_dotpi2   = self.norm_count_dotpi2*self.beta_dotpi2
                    fNL_val_nablapi2 = self.norm_count_nablapi2*self.beta_nablapi2

                    self.phifcoll1_1  = (fNL_val_dotpi2 + fNL_val_nablapi2)*prefactor*(k_grid/0.45)**1.5*np.cos(mu*np.log(k_grid/0.45))
                    self.phifcoll1_2  = (fNL_val_dotpi2 + fNL_val_nablapi2)*prefactor*(k_grid/0.45)**1.5*np.sin(mu*np.log(k_grid/0.45))

                # Initialize arrays
                theory_Pl = np.zeros((len(k_grid),3))

                # Compute the power spectrum multipoles for Gaussian initial conditions
                theory_Pl[:,0] = (norm**2.*all_theory[15]+norm**4.*(all_theory[21])+norm**1.*b1*all_theory[16]+norm**3.*b1*(all_theory[22]) + norm**0.*b1**2.*all_theory[17] +norm**2.*b1**2.*all_theory[23] + 0.25*norm**2.*b2**2.*all_theory[1] +b1*b2*norm**2.*all_theory[30]+ b2*norm**3.*all_theory[31] + b1*bG2*norm**2.*all_theory[32]+ bG2*norm**3.*all_theory[33] + b2*bG2*norm**2.*all_theory[4]+ bG2**2.*norm**2.*all_theory[5] + 2.*cs0*norm**2.*all_theory[11]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**2.*(b1*all_theory[7]+norm*all_theory[8]))*h**3. + (inv_nbar)*Pshot + a0*inv_nbar*(k_grid/0.5)**2.  + fz**2.*b4*k_grid**2.*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)*all_theory[13]*h + a2*(1./3.)*(10.**4.)*(k_grid/0.45)**2.
                theory_Pl[:,1] = (norm**2.*all_theory[18]+norm**4.*(all_theory[24])+norm**1.*b1*all_theory[19]+norm**3.*b1*(all_theory[25]) + b1**2.*norm**2.*all_theory[26] +b1*b2*norm**2.*all_theory[34]+ b2*norm**3.*all_theory[35] + b1*bG2*norm**2.*all_theory[36]+ bG2*norm**3.*all_theory[37] + 0.25*b2**2.*all_theory[42] + b2*bG2*all_theory[43] + (bG2**2.)*all_theory[44] + 2.*cs2*norm**2.*all_theory[12]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**3.*all_theory[9])*h**3. + fz**2.*b4*k_grid**2.*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)*all_theory[13]*h + a2*(10.**4.)*(2./3.)*(k_grid/0.45)**2.
                theory_Pl[:,2] = (norm**2.*all_theory[20]+norm**4.*all_theory[27]+b1*norm**3.*all_theory[28]+b1**2.*norm**2.*all_theory[29] + b2*norm**3.*all_theory[38] + bG2*norm**3.*all_theory[39] + b1*b2*all_theory[40] + b1*bG2*all_theory[41] + 0.25*b2**2.*all_theory[45] + b2*bG2*all_theory[46] + (bG2**2.)*all_theory[46] +2.*cs4*norm**2.*all_theory[13]/h**2.)*h**3. + fz**2.*b4*k_grid**2.*(norm**2.*fz**2.*210./143. + 30.*fz*b1*norm/11.+b1**2.)*all_theory[13]*h

                # Add fNL-eq/orth terms
                if self.use_eq_orth:

                    # Eq
                    theory_Pl[:,0] += self.fNL_eq*(h**3.)*(all_theory[51]+b1*all_theory[52]+b1**2.*all_theory[53]+b1*b2*all_theory[60]+b2*all_theory[61]+b1*bG2*all_theory[62]+bG2*all_theory[63])
                    theory_Pl[:,1] += self.fNL_eq*(h**3.)*(all_theory[54]+b1*all_theory[55]+b1**2.*all_theory[56]+b1*b2*all_theory[64]+b2*all_theory[65]+b1*bG2*all_theory[66]+bG2*all_theory[67])
                    theory_Pl[:,2] += self.fNL_eq*(h**3.)*(all_theory[57]+b1*all_theory[58]+b1**2.*all_theory[59]+b1*b2*all_theory[68]+b2*all_theory[69]+b1*bG2*all_theory[70]+bG2*all_theory[71])

                    # Orth
                    theory_Pl[:,0] += self.fNL_orth*(h**3.)*(all_theory[75]+b1*all_theory[76]+b1**2.*all_theory[77]+b1*b2*all_theory[84]+b2*all_theory[85]+b1*bG2*all_theory[86]+bG2*all_theory[87])
                    theory_Pl[:,1] += self.fNL_orth*(h**3.)*(all_theory[78]+b1*all_theory[79]+b1**2.*all_theory[80]+b1*b2*all_theory[88]+b2*all_theory[89]+b1*bG2*all_theory[90]+bG2*all_theory[91])
                    theory_Pl[:,2] += self.fNL_orth*(h**3.)*(all_theory[81]+b1*all_theory[82]+b1**2.*all_theory[83]+b1*b2*all_theory[92]+b2*all_theory[93]+b1*bG2*all_theory[94]+bG2*all_theory[95])

                    # Counterterms
                    theory_Pl[:,0] +=  (2.*b1*bphi*self.phif1+bphi**2*self.phif1**2.)*all_theory[17]*(h**3.) + 1.*bphi*self.phif1*all_theory[16]*(h**3.)
                    theory_Pl[:,1] += bphi*self.phif1*all_theory[19]*(h**3.)

                # Add collider terms
                if self.use_collider:
                    for i in range(3):
                        theory_Pl[:,i] += (self.beta_dotpi2*(h**3.)*(self.Pl_dotpi2[:,i,0]+b1*self.Pl_dotpi2[:,i,1]+b1*b1*self.Pl_dotpi2[:,i,2]+(b2/2.)*self.Pl_dotpi2[:,i,3]+bG2*self.Pl_dotpi2[:,i,4]+(b1*b2/2.)*self.Pl_dotpi2[:,i,5]+(b1*bG2)*self.Pl_dotpi2[:,i,6])).ravel()
                        theory_Pl[:,i] += (self.beta_nablapi2*(h**3.)*(self.Pl_nablapi2[:,i,0]+b1*self.Pl_nablapi2[:,i,1]+b1*b1*self.Pl_nablapi2[:,i,2]+(b2/2.)*self.Pl_nablapi2[:,i,3]+bG2*self.Pl_nablapi2[:,i,4]+(b1*b2/2.)*self.Pl_nablapi2[:,i,5]+(b1*bG2)*self.Pl_nablapi2[:,i,6])).ravel()

                    # Counterterms
                    theory_Pl[:,0] += (2.*b1*bphi_coll_1*self.phifcoll1_1+bphi_coll_1**2*self.phifcoll1_1**2.)*all_theory[17]*(h**3.) + 1.*bphi_coll_1*self.phifcoll1_1*all_theory[16]*(h**3.)+(2.*b1*bphi_coll_2*self.phifcoll1_2+bphi_coll_2**2*self.phifcoll1_2**2.)*all_theory[17]*(h**3.) + 1.*bphi_coll_2*self.phifcoll1_2*all_theory[16]*(h**3.)
                    theory_Pl[:,1] += bphi_coll_1*self.phifcoll1_1*all_theory[19]*(h**3.)+bphi_coll_2*self.phifcoll1_2*all_theory[19]*(h**3.)

                # Apply bin integration if requested
                self.Pl = np.zeros((len(self.kPQ),3))
                for i in range(3):
                    self.Pl[:,i] = self.bin_integrator(theory_Pl[:,i])

        def _load_individual_derivatives(self, biases):
                """Compute individual derivatives needed to construct Pl and Q0 derivatives. This preloads the quantities requiring bin integration."""

                # Load quantities
                all_theory = self.all_theory
                norm = self.norm
                h = self.h
                fz = self.fz
                k_grid = self.k_grid
                b1 = biases['b1']

                # Compute derivatives, including bin integration if requested
                zeros = np.zeros(len(self.kPQ))
                self.deriv0_bGamma3 = self.bin_integrator((0.8*norm)*norm**2.*(b1*all_theory[7]+norm*all_theory[8])*h**3.)
                self.deriv2_bGamma3 = self.bin_integrator((0.8*norm)*norm**3.*all_theory[9]*h**3.)
                self.deriv0_cs0 = self.bin_integrator(2.*norm**2.*all_theory[11]*h**1.)
                self.deriv2_cs2 = self.bin_integrator(2.*norm**2.*all_theory[12]*h**1.)
                self.deriv4_cs4 = self.bin_integrator(2.*norm**2.*all_theory[13]*h**1.)
                self.derivN_b4 = self.bin_integrator(fz**2.*k_grid**2.*all_theory[13]*h)

                # Add derivatives for bphi parameters
                self.deriv0_bphi = np.zeros(len(self.kPQ))
                self.deriv2_bphi = np.zeros(len(self.kPQ))
                self.deriv0_bphi_coll = np.zeros(len(self.kPQ))
                self.deriv2_bphi_coll = np.zeros(len(self.kPQ))

                if self.use_eq_orth:
                    bphi = biases['bphi']
                    self.deriv0_bphi      = self.bin_integrator((2.*b1*self.phif1    +2.*bphi*self.phif1**2)*all_theory[17]*(h**3.)         + self.phif1*all_theory[16]*(h**3.))
                    self.deriv2_bphi      = self.bin_integrator(self.phif1*all_theory[19]*(h**3.))

                if self.use_collider:
                    bphi_coll_1 = biases['bphi_coll_1']
                    bphi_coll_2 = biases['bphi_coll_2']
                    self.deriv0_bphi_coll_1 = self.bin_integrator((2.*b1*self.phifcoll1_1+2.*bphi_coll_1*self.phifcoll1_1**2)*all_theory[17]*(h**3.) + self.phifcoll1_1*all_theory[16]*(h**3.))
                    self.deriv2_bphi_coll_1 = self.bin_integrator(self.phifcoll1_1*all_theory[19]*(h**3.))
                    self.deriv0_bphi_coll_2 = self.bin_integrator((2.*b1*self.phifcoll1_2+2.*bphi_coll_2*self.phifcoll1_2**2)*all_theory[17]*(h**3.) + self.phifcoll1_2*all_theory[16]*(h**3.))
                    self.deriv2_bphi_coll_2 = self.bin_integrator(self.phifcoll1_2*all_theory[19]*(h**3.))

        def compute_Pl_derivatives(self, biases):
                """Compute the derivatives of the power spectrum multipoles with respect to parameters entering the model linearly"""

                # Load quantities
                norm = self.norm
                fz = self.fz
                kPQ = self.kPQ
                nP = self.nP
                nlP = self.nlP
                b1 = biases['b1']

                # Compute individual derivatives
                if not hasattr(self, 'deriv0_bGamma3'):
                        self._load_individual_derivatives(biases)

                # Assemble stacked derivatives
                zeros = np.zeros(nP)
                derivP = {}
                derivP['bGamma3'] = np.concatenate([self.deriv0_bGamma3[:nP],self.deriv2_bGamma3[:nP],zeros])[:nP*nlP]
                derivP['cs0'] = np.concatenate([self.deriv0_cs0[:nP],zeros,zeros])[:nP*nlP]
                derivP['cs2'] = np.concatenate([zeros,self.deriv2_cs2[:nP],zeros])[:nP*nlP]
                derivP['cs4'] = np.concatenate([zeros,zeros,self.deriv4_cs4[:nP]])[:nP*nlP]
                derivP['b4'] = np.concatenate([self.derivN_b4[:nP]*(norm**2.*fz**2./9.+2.*fz*b1*norm/7. + b1**2./5)*(35./8.),
                                               self.derivN_b4[:nP]*((norm**2.*fz**2.*70.+165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.),
                                               self.derivN_b4[:nP]*(norm**2.*fz**2.*210./143.+30.*fz*b1*norm/11.+b1**2.)])[:nP*nlP]
                derivP['Pshot'] = np.concatenate([1.+zeros, zeros, zeros])[:nP*nlP]
                derivP['a0'] = np.concatenate([(kPQ[:nP]/0.45)**2., zeros, zeros])[:nP*nlP]
                derivP['a2'] = np.concatenate([(1./3.)*(kPQ[:nP]/0.45)**2., (2./3.)*(kPQ[:nP]/0.45)**2., zeros])[:nP*nlP]

                if self.use_eq_orth:
                    derivP['bphi'] = np.concatenate([self.deriv0_bphi[:nP], self.deriv2_bphi[:nP], zeros])[:nP*nlP]

                if self.use_collider:
                    derivP['bphi_coll_1'] = np.concatenate([self.deriv0_bphi_coll_1[:nP],self.deriv2_bphi_coll_1[:nP], zeros])[:nP*nlP]
                    derivP['bphi_coll_2'] = np.concatenate([self.deriv0_bphi_coll_2[:nP],self.deriv2_bphi_coll_2[:nP], zeros])[:nP*nlP]

                return derivP

        def compute_Q0_derivatives(self, biases):
                """Compute the derivatives of Q0 with respect to parameters entering the model linearly"""

                # Load quantities
                norm = self.norm
                h = self.h
                fz = self.fz
                kPQ = self.kPQ
                nP = self.nP
                b1 = biases['b1']

                # Compute individual derivatives
                if not hasattr(self, 'deriv0_bGamma3'):
                        self._load_individual_derivatives(biases)

                # Assemble stacked derivatives
                derivQ = {}
                derivQ['bGamma3'] = self.deriv0_bGamma3[nP:] - 1./2.*self.deriv2_bGamma3[nP:]
                derivQ['cs0'] = self.deriv0_cs0[nP:]
                derivQ['cs2'] = -1./2.*self.deriv2_cs2[nP:]
                derivQ['cs4'] = 3./8.*self.deriv4_cs4[nP:]
                derivQ['b4'] = self.derivN_b4[nP:]*((norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.) - ((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)/2. +3.*(norm**2.*fz**2.*210./143. + 30.*fz*b1*norm/11.+b1**2.)/8.)
                derivQ['Pshot'] = 1.
                derivQ['a0'] = (kPQ[nP:]/0.45)**2.
                derivQ['a2'] = 0.*derivQ['a0']
                if self.use_eq_orth:
                    derivQ['bphi'] = self.deriv0_bphi[nP:] - 1./2.*self.deriv2_bphi[nP:]

                if self.use_collider:
                    derivQ['bphi_coll_1'] = self.deriv0_bphi_coll_1[nP:] - 1./2.*self.deriv2_bphi_coll_1[nP:]
                    derivQ['bphi_coll_2'] = self.deriv0_bphi_coll_2[nP:] - 1./2.*self.deriv2_bphi_coll_2[nP:]

                return derivQ

class BkTheory(object):
        def __init__(self, options, As, zi, ng_params, apar, aperp, fz, sigma8, r_bao, k_grid, Tfunc, Pk_lin_table1, Pk_lin_table2, inv_nbar, gauss_w, gauss_w2, mesh_mu):
                """Compute the theoretical power spectrum P(k) and parameter derivatives for a given cosmology and set of nuisance parameters."""

                # Load variables
                self.options = options
                self.dataset = options.dataset
                self.apar = apar
                self.aperp = aperp
                self.fz = fz
                self.sigma8 = sigma8
                self.r_bao = r_bao
                self.inv_nbar = inv_nbar
                self.gauss_w = gauss_w
                self.gauss_w2 = gauss_w2
                self.mesh_mu = mesh_mu
                self.k_grid = k_grid

                self.triangle_indices = self.dataset.triangle_indices[zi]
                self.kB = self.dataset.kB[zi]
                self.dkB = self.dataset.dkB[zi]
                self.nB = self.dataset.nB[zi]
                self.nlB = self.dataset.nlB

                self.use_eq_orth = options.use_eq_orth
                self.use_collider = options.use_collider

                # Non-Gaussianity parameters
                if self.use_eq_orth:
                    self.fNL_eq = ng_params['fNL_eq']
                    self.fNL_orth = ng_params['fNL_orth']
                if self.use_collider:
                    self.beta_dotpi2 = ng_params['beta_dotpi2']
                    self.beta_nablapi2 = ng_params['beta_nablapi2']
                    self.mu = ng_params['mu']
                    self.cs = ng_params['cs']

                # Load functions and tables
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

        def _Bk_collider(self,k1,k2,k3,mu,cs):

            norm_0 = (-10./9.)*cs, (5./9.)/cs
            shapes = self.dataset.collider_shapes(k2/k1,k3/k1,mu,cs)

            prefactor = (18./5.)*np.power(self.Azeta,2.0)/(k1*k2*k3)**2.0*self.Tfunc(k1)*self.Tfunc(k2)*self.Tfunc(k3)

            dotpi2 = prefactor*norm_0[0]*shapes[0]
            nablapi2 = prefactor*norm_0[1]*shapes[1]
            return dotpi2, nablapi2

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

        def _compute_B_matrices_tree(self,biases,kc1,kc2,kc3,dk1,dk2,dk3,k1,k2,k3,mu1,phi):
                """Load the tree-level bispectrum matrices for a given set of k bins. These will later be integrated over bins to form the bispectrum monopole and derivatives"""

                # Define local variables
                apar = self.apar
                aperp = self.aperp
                inv_nbar = self.inv_nbar
                b1 = biases['b1']
                b2 = biases['b2']
                bG2 = biases['bG2']
                Pshot = biases['Pshot']
                Bshot = biases['Bshot']
                beta = self.fz/biases['b1']

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

                # Equilateral / orthogonal
                if self.use_eq_orth:
                    FFnlfunc += self.fNL_eq*self._Bk_eq(kk1*qq1,kk2*qq2,kk3*qq3)*b1**3.*(1+beta*nnu1**2)*(1.+beta*nnu3**2.)*(1+beta*nnu2**2)*kk1*kk2*kk3*ddk1*ddk2*ddk3
                    FFnlfunc += self.fNL_orth*self._Bk_orth(kk1*qq1,kk2*qq2,kk3*qq3)*b1**3.*(1+beta*nnu1**2)*(1.+beta*nnu3**2.)*(1+beta*nnu2**2)*kk1*kk2*kk3*ddk1*ddk2*ddk3

                # Collider shapes
                if self.use_collider:
                    Bk_collider = self._Bk_collider(kk1*qq1,kk2*qq2,kk3*qq3,self.mu,self.cs)
                    FFnlfunc += self.beta_dotpi2*Bk_collider[0]*b1**3.*(1+beta*nnu1**2)*(1.+beta*nnu3**2.)*(1+beta*nnu2**2)*kk1*kk2*kk3*ddk1*ddk2*ddk3
                    FFnlfunc += self.beta_nablapi2*Bk_collider[1]*b1**3.*(1+beta*nnu1**2)*(1.+beta*nnu3**2.)*(1+beta*nnu2**2)*kk1*kk2*kk3*ddk1*ddk2*ddk3

                # Assemble output bispectrum matrices
                Bl_matrix = (2.*FF2func1 + 2.*FF2func2 + 2.*FF2func3 + FFnlfunc)/apar**2./aperp**4.

                deriv_Pshot_matrix = (b1**2.*(1.*beta*nnu1**2.*(1.+beta*nnu1**2.)*P_IR1+P_IR2*(beta*nnu2**2.)*(1.+beta*nnu2**2.)+ P_IR3*(beta*nnu3**2.)*(1.+beta*nnu3**2.)) + 2.*inv_nbar*(1.+Pshot))*kk1*kk2*kk3*ddk1*ddk2*ddk3/apar**2./aperp**4.
                deriv_Bshot_matrix = b1**2.*(((1.+beta*nnu1**2.)*P_IR1+P_IR2*(1.+beta*nnu2**2.)+ P_IR3*(1.+beta*nnu3**2.))*kk1*kk2*kk3*ddk1*ddk2*ddk3)/apar**2./aperp**4.
                deriv_c1_matrix = (2.*FF2func1C + 2.*FF2func2C + 2.*FF2func3C - 2.*FF2func1 - 2.*FF2func2 - 2.*FF2func3)/apar**2./aperp**4.

                return Bl_matrix, deriv_Pshot_matrix, deriv_Bshot_matrix, deriv_c1_matrix

        def compute_Bl_theory_derivs(self, biases):
            """Compute the bispectrum at tree-level, given the bias parameters. This computes both the theory and the derivatives with respect to linear parameters."""

            # Define local variables
            fz = self.fz

            # Load in bias parameters
            b1 = biases['b1']
            b2 = biases['b2']
            bG2 = biases['bG2']
            c1 = biases['c1']
            Pshot = biases['Pshot']
            Bshot = biases['Bshot']
            beta = fz/b1

            # Pre-compute IR resummation quantities
            if not hasattr(self,'P_IR'):
                    self._load_IR_resummation(b1, c1)

           # Iterate over bispectrum bins and compute B0
            Bl = np.zeros((self.nB,self.nlB))
            zeros = np.zeros(self.nB*self.nlB)
            derivB = {'Pshot':zeros.copy(), 'Bshot':zeros.copy(), 'c1':zeros.copy()}

            for j in range(int(self.nB)):
                    # Bin-centers
                    kc1, kc2, kc3 = self.kB[self.triangle_indices[0][j]], self.kB[self.triangle_indices[1][j]], self.kB[self.triangle_indices[2][j]]
                    # Bin-widths
                    dk1, dk2, dk3 = self.dkB, self.dkB, self.dkB

                    # Check bin edges
                    if (self.kB[self.triangle_indices[0][j]]<self.dkB) or (self.kB[self.triangle_indices[1][j]]<self.dkB) or (self.kB[self.triangle_indices[2][j]]<self.dkB):
                            raise Exception('Lowest bispectrum bin center is below dk; alternative binning must be specified!')

                    # Idealized bin volume
                    Nk123 = ((kc1+dk1/2.)**2.-(kc1-dk1/2.)**2.)*((kc2+dk2/2.)**2.-(kc2-dk2/2.)**2.)*((kc3+dk3/2.)**2.-(kc3-dk3/2.)**2.)/8.

                    # Compute matrices
                    B_matrix_tree, deriv_Pshot_matrix_tree, deriv_Bshot_matrix_tree, deriv_c1_matrix_tree = self._compute_B_matrices_tree(biases,kc1,kc2,kc3,dk1,dk2,dk3,*self.mesh_mu)

                    for li in range(self.nlB):

                        # Compute Legendre weighting
                        leg_factor = (4*li+1)*legendre(2*li)(self.mesh_mu[-2])

                        # Integrate over bins to compute B0
                        Bl[j,li] = self._bin_integrate(B_matrix_tree*leg_factor)/Nk123

                        # Update nuisance parameter covariance
                        derivB['Pshot'][li*self.nB+j] = self._bin_integrate(deriv_Pshot_matrix_tree*leg_factor)/Nk123
                        derivB['Bshot'][li*self.nB+j] = self._bin_integrate(deriv_Bshot_matrix_tree*leg_factor)/Nk123
                        derivB['c1'][li*self.nB+j] = self._bin_integrate(deriv_c1_matrix_tree*leg_factor)/Nk123

            return Bl, derivB
