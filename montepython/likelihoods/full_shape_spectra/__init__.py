import numpy as np
from montepython.likelihood_class import Likelihood_prior
from scipy import interpolate
from scipy.special.orthogonal import p_roots
import os,sys
path=os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)
from fs_utils import Datasets, PkTheory, BkTheory

import classy

class full_shape_spectra(Likelihood_prior):

        # initialisation of the class is done within the parent Likelihood_prior. For
        # this case, it does not differ, actually, from the __init__ method in
        # Likelihood class.

        def __init__(self,path,data,command_line,extra_inputs=[]):
                """Initialize the full shape likelihood. This loads the data-set and pre-computes a number of useful quantities."""

                # Initialize the  likelihood
                Likelihood_prior.__init__(self,path,data,command_line)

                # Load the data
                self.dataset = Datasets(self)

                # Define nuisance parameter mean and standard deviations
                # mean of b2 and bG2 based on b2(b1), bG2(b1) relations from Kaz's paper
                # same for variance, if they vary...
                # NB: different convention for mean and variance of shot noise in P and B
                shape = np.ones(self.dataset.nz)

                self.prior_means = {#'b2': lambda b1: 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3 - 8./21.*(b1 - 1),
                                    #'bG2': lambda b1: -2./7.*(b1-1),
                                    'bGamma3': lambda b1: 23.*(b1-1.)/42.,
                                    'cs0': 0.*shape,
                                    'cs2': 30.*shape,
                                    'cs4': 0.*shape,
                                    'b4':500.*shape,
                                    'c1': 0.*shape,
                                    'Pshot': 0.*shape,
                                    'Bshot': 1.*shape,
                                    'a0': 0.*shape,
                                    'a2': 0.*shape,
                                    'bphi': 1.*shape,
                                    'bphi_coll_1': 1.*shape,
                                    'bphi_coll_2': 1.*shape} # last three only relevant for non-Gaussianity!
                self.prior_stds = {#'b2': 1.*shape,
                                   #'bG2': 1.*shape,
                                   'bGamma3': 1.*shape,
                                   'cs0': 30.*shape,
                                   'cs2': 30.*shape,
                                   'cs4': 30.*shape,
                                   'b4':500.*shape,
                                   'c1': 5.*shape,
                                   'Pshot': self.inv_nbar,
                                   'Bshot': self.inv_nbar,
                                   'a0': self.inv_nbar,
                                   'a2': self.inv_nbar,
                                   'bphi': 5.*shape,
                                   'bphi_coll_1': 5.*shape,
                                   'bphi_coll_2': 5.*shape} # last three only relevant for non-Gaussianity!

                if not self.bias_relations_varying and not self.bias_relations_fixed:
                    print('~> you are not using HOD-inspired relations for b2 and bG2...')

                if self.bias_relations_varying:
                        # Replace quadratic biases mean and variance with bias relation predictions from HOD fitting of Kaz
                        assert not self.bias_relations_fixed, '~> you are asking to fix biases and vary them at the same time!'
                        print("~> careful! You are using a fit to variance of b2 and bG2 in terms of b1. Be sure that b1 is sampled in the range where these are positive...")
                        self.prior_means |= {'b2': lambda b1:  -0.38 - 0.15*b1 - 0.021*b1**2. - 0.047*b1**3.,
                                             'bG2': lambda b1: 0.18 - 0.11*b1 - 0.015*b1**2.}
                        self.prior_stds  |= {'b2': lambda b1:  0.06*b1 + 0.24*b1**2. + 0.02*b1**3. - 0.003*b1**4.,
                                             'bG2': lambda b1: 0.11*b1 - 0.012*b1**2. - 0.001*b1**3.}
                else:
                        self.prior_means |= {'b2': lambda b1: 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3 - 8./21.*(b1 - 1),
                                             'bG2': lambda b1: -2./7.*(b1-1)}
                        self.prior_stds  |= {'b2': 1.*shape,
                                             'bG2': 1.*shape}

                # Pre-load useful quantities for bispectra
                if self.use_B:
                        [gauss_mu,self.gauss_w], [gauss_mu2,self.gauss_w2] = p_roots(self.n_gauss), p_roots(self.n_gauss2)
                        self.mesh_mu = np.meshgrid(gauss_mu,gauss_mu,gauss_mu,gauss_mu2,gauss_mu2, sparse=True, indexing='ij')

                # Load collider interpolation tables
                if hasattr(self,'use_collider'):
                    if self.use_collider and (self.use_P or self.use_B or self.use_Q):
                        print("Loading collider interpolators...")
                        self.dataset._load_collider_shapes(self)
                        self.dataset._load_collider_power_spectra(self)

                # Summarize input parameters
                print("#################################################\n")
                print("### Full shape likelihood\n")
                print("Redshifts: %s"%(['%.2f'%zz for zz in self.z[:self.nz]]))
                print("Power Spectrum: %s"%self.use_P)
                print("Q0: %s"%self.use_Q)
                if self.use_B:
                        print("Bispectra: Tree-Level")
                else:
                        print("Bispectra: False")
                print("AP Parameters: %s"%self.use_AP)
                print("")
                for zi in range(self.nz):
                        print("# z-bin %d #"%zi)
                        if self.use_P:
                                print("k-min (Pk): %.3f"%self.kminP[zi])
                                print("k-max (Pk): %.3f"%self.kmaxP[zi])
                                print("N-bins (Pk): %d"%(len(self.dataset.Pl[zi])))
                                print("l-max (Pk): %d"%self.lmaxP)
                                print("")
                        if self.use_Q:
                                print("k-min (Q0): %.3f"%self.kmaxP[zi])
                                print("k-max (Q0): %.3f"%self.kmaxQ[zi])
                                print("N-bins (Q0): %d"%len(self.dataset.Q0[zi]))
                                print("")
                        if self.use_B:
                                print("k-min (Bk): %.3f"%self.kminB[zi])
                                print("k-max (Bk): %.3f"%self.kmaxB[zi])
                                print("N-bins (Bk): %d"%len(self.dataset.Bl[zi]))
                                print("l-max (Bk): %d"%self.lmaxB)
                                print("")
                print("#################################################\n")

        def _sm_inv_det(self,imat,logdet,x_list):
                """Sherman-Morrison matrix inversion for A + x x^T + y y^T + ...
                This computes both the inverse and the determinant simultaneously."""
                for x in x_list:
                    logdet += np.log(1.+np.inner(x,np.inner(imat,x)))
                    out = np.outer(x,x)
                    imat = imat - np.matmul(imat,np.matmul(out,imat))/(1.+np.inner(x,np.inner(imat,x)))
                return imat, logdet

        def loglkl(self, cosmo, data):
                """Compute the log-likelihood for a given set of cosmological and nuisance parameters. Note that this marginalizes over nuisance parameters that enter the model linearly and that have a Gaussian prior."""

                # Load cosmological parameters
                h = cosmo.h()
                As = cosmo.A_s()
                norm = 1.

                # PNG stuff
                fNL_eq = (data.mcmc_parameters['f^{eq}_{NL}']['current'] * data.mcmc_parameters['f^{eq}_{NL}']['scale'])
                fNL_orth = (data.mcmc_parameters['f^{orth}_{NL}']['current'] * data.mcmc_parameters['f^{orth}_{NL}']['scale'])

                # Check we're running in the correct mode for PNG
                if fNL_eq!=0. or fNL_orth!=0.:
                        try:
                                data.cosmo_arguments['PNG']
                        except:
                                raise Exception('PNG mode must be turned on to use fNL!')
                        assert data.cosmo_arguments['PNG']=='Yes', 'PNG mode must be turned on to use fNL!'

                beta_dotpi2 = (data.mcmc_parameters['beta_dotpi2']['current'] * data.mcmc_parameters['beta_dotpi2']['scale'])
                beta_nablapi2 = (data.mcmc_parameters['beta_nablapi2']['current'] * data.mcmc_parameters['beta_nablapi2']['scale'])

                # Check we're running in the correct mode for collider
                if beta_dotpi2!=0. or beta_nablapi2!=0.:
                        try:
                                self.use_collider
                        except:
                                raise Exception('collider mode must be turned on to use beta collider!')
                        assert self.use_collider, 'collider mode must be turned on to use beta collider!'

                mu = (data.mcmc_parameters['mu']['current'] * data.mcmc_parameters['mu']['scale'])
                cs = np.power(10.,(data.mcmc_parameters['log10cs']['current'] * data.mcmc_parameters['log10cs']['scale']))

                alpha_rs = (data.mcmc_parameters['alpha_{r_s}']['current'] * data.mcmc_parameters['alpha_{r_s}']['scale'])

                z = self.z[:self.nz]
                fz = np.asarray([cosmo.scale_independent_growth_factor_f(zz) for zz in z])
                # Define local variables
                dataset = self.dataset

                for i_s in range(1,1+self.nz):
                    if (data.mcmc_parameters['b^{('+str(i_s)+')}_2']['status']=='varying' or data.mcmc_parameters['b^{('+str(i_s)+')}_{G_2}']['status']=='varying') and self.bias_relations_fixed:
                        raise Exception("~> don't need to vary quadratic and tidal parameters if fixing bias relations!")

                # Load non-linear nuisance parameters
                b1 = np.asarray([(data.mcmc_parameters['b^{('+str(i_s)+')}_1']['current']*data.mcmc_parameters['b^{('+str(i_s)+')}_1']['scale']) for i_s in range(1,1+self.nz)])
                b2 = np.asarray([(data.mcmc_parameters['b^{('+str(i_s)+')}_2']['current']*data.mcmc_parameters['b^{('+str(i_s)+')}_2']['scale']) for i_s in range(1,1+self.nz)])
                bG2 = np.asarray([(data.mcmc_parameters['b^{('+str(i_s)+')}_{G_2}']['current']*data.mcmc_parameters['b^{('+str(i_s)+')}_{G_2}']['scale']) for i_s in range(1,1+self.nz)])

                if self.bias_relations_fixed:
                        # Replace quadratic biases with *mean* bias relation predictions from HOD fitting of Kaz
                        assert not self.bias_relations_varying, '~> you are asking to fix biases and vary them at the same time!'
                        b2  = -0.38 - 0.15*b1 - 0.021*b1**2. - 0.047*b1**3.
                        bG2 = 0.18 - 0.11*b1 - 0.015*b1**2.

                # Define implicitly marginalized parameters and non-Gaussianity parameters
                # The former are the same as the list of biases for which we do not have a fit to the variance from HOD...
                linear_params = ['bGamma3','Pshot','Bshot','c1','a0','a2','cs0','cs2','cs4','b4']
                ng_pars = {}
                if self.use_eq_orth:
                    linear_params += ['bphi']
                    ng_pars |= {'fNL_eq': fNL_eq, 'fNL_orth': fNL_orth}
                    if cosmo.pars['PNG'] not in ['Yes','yes','y',True]: raise Exception("Need to run CLASS-PT with PNG mode turned on!")
                if self.use_collider:
                    linear_params += ['bphi_coll_1','bphi_coll_2']
                    ng_pars |= {'beta_dotpi2': beta_dotpi2, 'beta_nablapi2': beta_nablapi2, 'mu': mu, 'cs': cs}

                # Compute useful quantities for AP parameters
                if self.use_AP or self.use_B:
                    DA_th = np.asarray([cosmo.angular_distance(zz) for zz in z])
                    Hz_th = np.asarray([cosmo.Hubble(zz) for zz in z])
                    rs_th = cosmo.rs_drag()

                # Initialize output chi2
                chi2 = 0.

                # Iterate over redshift bins
                for zi in range(self.nz):

                        # Copy local variables
                        nP, nQ, nB, nAP, nlP, nlB = dataset.nP[zi], dataset.nQ[zi], dataset.nB[zi], dataset.nAP, dataset.nlP, dataset.nlB

                        # Create output arrays (will be overwritten for each redshift)
                        theory_minus_data = np.zeros(nlP*nP+nlB*nB+nQ+nAP)
                        deriv = {p: np.zeros(nlP*nP+nlB*nB+nQ+nAP) for p in linear_params}

                        ## Create list of bias parameters and variances for this redshift slice
                        biases = {'b1': b1[zi], 'b2': b2[zi], 'bG2': bG2[zi],
                                  'inv_nbar': self.inv_nbar[zi], 'bGamma3':self.prior_means['bGamma3'](b1)[zi]}
                        biases |= {key: self.prior_means[key][zi] for key in linear_params[1:]}
                        stds    = {key: self.prior_stds[key][zi] for key in linear_params}
                        if self.bias_relations_varying:
                                # Replace quadratic biases mean and variance with bias relation predictions from HOD fitting of Kaz
                                assert not self.bias_relations_fixed, '~> you are asking to fix biases and vary them at the same time!'
                                stds   |= {'b2': self.prior_stds['b2'](b1)[zi], 'bG2': self.prior_stds['bG2'](b1)[zi]}
                        else:
                                stds   |= {'b2': self.prior_stds['b2'][zi], 'bG2': self.prior_stds['bG2'][zi]}

                        ## Run CLASS and set-up necessary functions
                        if self.use_P or self.use_Q or self.use_B:

                                # Define k-array
                                if self.bin_integration_P:
                                    k_grid = np.exp(np.linspace(np.log(1.e-4),np.log(max(dataset.kPQ[zi])+0.01),100))
                                else:
                                    k_grid = dataset.kPQ[zi]

                                # Run CLASS-PT
                                all_theory = cosmo.get_pk_mult(k_grid*h, z[zi], len(k_grid), no_wiggle = self.no_wiggle, alpha_rs = alpha_rs)

                                # Load IR resummation utilities
                                Pk_lin_table1 = -1.*norm**2.*(all_theory[10]/h**2./k_grid**2)*h**3
                                Pk_lin_table2 = norm**2.*(all_theory[14])*h**3.
                                P0int = interpolate.InterpolatedUnivariateSpline(k_grid,Pk_lin_table1,ext=3)
                                Tfunc = lambda k: (P0int(k)/(As*2.*np.pi**2.*((k*h/0.05)**(cosmo.n_s()-1.))/k**3.))**0.5

                        if self.use_P or self.use_Q:
                                # Define PkTheory class, used to compute power spectra and derivatives
                                pk_theory = PkTheory(self, all_theory, h, As, zi, z[zi], ng_pars,
                                                     norm, fz[zi], k_grid, Tfunc(k_grid))

                        if self.use_B:
                                # Define coordinate rescaling parameters and BAO parameters
                                apar=self.Hz_fid[zi]/(Hz_th[zi]*(self.h_fid/h)/3.33564095198145e-6) # including kmsMpc conversion factor
                                aperp=DA_th[zi]/self.DA_fid[zi]/(self.h_fid/h)
                                r_bao = rs_th*h

                                # Define BkTheory class, used to compute bispectra and derivatives
                                bk_theory = BkTheory(self, As, zi, ng_pars, apar, aperp, fz[zi], cosmo.sigma8(), r_bao, k_grid, Tfunc, Pk_lin_table1, Pk_lin_table2, self.inv_nbar[zi], self.gauss_w, self.gauss_w2, self.mesh_mu)

                        ## Compute P_ell
                        if self.use_P:

                                # Compute theory model for Pl and add to (theory - data)
                                Pl = pk_theory.compute_Pl_oneloop(biases)
                                for i in range(nlP):
                                    theory_minus_data[i*nP:(i+1)*nP] = Pl[:,i] - dataset.Pl[zi][:,i]

                                # Compute derivatives of Pl with respect to parameters
                                derivP = pk_theory.compute_Pl_derivatives(biases)

                                # Add to joint derivative vector
                                for key in derivP: deriv[key][:nlP*nP] = derivP[key]

                        ## Compute Q0
                        if self.use_Q:

                                # Compute theory model for Q0 and add to (theory - data)
                                Q0 = pk_theory.compute_Q0_oneloop(biases)
                                theory_minus_data[nlP*nP:nlP*nP+nQ] = Q0 - dataset.Q0[zi]

                                # Compute derivatives of Q0 with respect to parameters
                                derivQ = pk_theory.compute_Q0_derivatives(biases)

                                # Add to joint derivative vector
                                for key in derivQ: deriv[key][nlP*nP:nlP*nP+nQ] = derivQ[key]

                        #### Compute AP parameters
                        if self.use_AP:

                                # Compute theory model for AP and add to (theory - data)
                                A_par = self.rdHfid[zi]/(rs_th*Hz_th[zi])
                                A_perp = self.rdDAfid[zi]/(rs_th/DA_th[zi])
                                theory_minus_data[-2] = A_par - dataset.alphas[zi][0]
                                theory_minus_data[-1] = A_perp - dataset.alphas[zi][1]

                        #### Compute bispectrum
                        if self.use_B:

                                # Compute the theory model for Bl and parameter derivatives
                                Bl, derivB = bk_theory.compute_Bl_theory_derivs(biases)

                                # Add Bl to (theory - data), including discreteness weights
                                for li in range(nlB):
                                    theory_minus_data[nlP*nP+nQ+li*nB:nlP*nP+nQ+(li+1)*nB] = Bl[:,li]*dataset.discreteness_weights[zi][:,li] - dataset.Bl[zi][:,li]

                                # Add derivatives to joint derivative vector
                                for key in derivB.keys(): deriv[key][nlP*nP+nQ:nlP*nP+nQ+nlB*nB] = derivB[key]

                        # Update inverse covariance
                        #marg_icov, delta_logdet = self._sm_inv_det(dataset.icov[zi],dataset.logdetcov[zi],[stds[key]*deriv[key] for key in linear_params])
                        marg_icov, delta_logdet = self._sm_inv_det(dataset.icov[zi],0.,[stds[key]*deriv[key] for key in linear_params])

                        # Compute chi2 from data and theory
                        chi2 += np.inner(theory_minus_data,np.inner(marg_icov,theory_minus_data))

                        # Correct covariance matrix normalization
                        chi2 += delta_logdet

                        # Add bias parameter priors for explicitly marginalized parameters
                        if not self.bias_relations_fixed:
                            chi2 += (b2[zi]-self.prior_means['b2'](b1)[zi])**2./stds['b2']**2. + (bG2[zi]-self.prior_means['bG2'](b1)[zi])**2./stds['bG2']**2.

                # Return full loglikelihood
                loglkl = -0.5*chi2

                return loglkl
