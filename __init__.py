import numpy as np
from montepython.likelihood_class import Likelihood_prior
from scipy import interpolate
from scipy.special.orthogonal import p_roots
from fs_utils import Datasets, PkTheory, BkTheory

class full_shape_spectra(Likelihood_prior):

        # initialisation of the class is done within the parent Likelihood_prior. For
        # this case, it does not differ, actually, from the __init__ method in
        # Likelihood class.

        def __init__(self,path,data,command_line):
                """Initialize the full shape likelihood. This loads the data-set and pre-computes a number of useful quantities."""

                # Initialize the likelihood
                Likelihood_prior.__init__(self,path,data,command_line)

                # Load the data
                self.dataset = Datasets(self)

                # Define nuisance parameter mean and standard deviations
                self.prior_b2 = 0., 1.
                self.prior_bG2 = 0., 1.
                self.prior_bGamma3 = lambda b1: (23.*(b1-1.)/42., 0.)  
                self.prior_cs0 = 0, 30.
                self.prior_cs2 = 30., 30.
                self.prior_cs4 = 0., 30.
                self.prior_b4 = 500., 500.
                self.prior_c1 = 0., 5.
                self.prior_Pshot = 0., self.inv_nbar
                self.prior_Bshot = 1., self.inv_nbar # NB: different convention for mean and variance
                self.prior_a0 = 0., self.inv_nbar
                self.prior_a2 = 0., self.inv_nbar
                self.prior_bphi = 1., 5.

                # Pre-load useful quantities for bispectra
                if self.use_B:
                        n_gauss = 3
                        n_gauss2 = 8
                        [gauss_mu,self.gauss_w], [gauss_mu2,self.gauss_w2] = p_roots(n_gauss), p_roots(n_gauss2)
                        self.mesh_mu = np.meshgrid(gauss_mu,gauss_mu,gauss_mu,gauss_mu2,gauss_mu2, sparse=True, indexing='ij')

                
        def loglkl(self, cosmo, data):
                """Compute the log-likelihood for a given set of cosmological and nuisance parameters. Note that this marginalizes over nuisance parameters that enter the model linearly."""

                # Load cosmological parameters
                h = cosmo.h()
                As = cosmo.A_s()
                norm = 1. # (A_s/A_s_fid)^{1/2}
                fNL_eq = (data.mcmc_parameters['f^{eq}_{NL}']['current'] * data.mcmc_parameters['f^{eq}_{NL}']['scale'])
                fNL_orth = (data.mcmc_parameters['f^{orth}_{NL}']['current'] * data.mcmc_parameters['f^{orth}_{NL}']['scale'])
                z = self.z
                fz = cosmo.scale_independent_growth_factor_f(z)
                
                # Load non-linear nuisance parameters
                i_s=repr(3)
                b1 = (data.mcmc_parameters['b^{('+i_s+')}_1']['current'] *
                data.mcmc_parameters['b^{('+i_s+')}_1']['scale'])
                b2 = (data.mcmc_parameters['b^{('+i_s+')}_2']['current'] *
                data.mcmc_parameters['b^{('+i_s+')}_2']['scale'])
                bG2 = (data.mcmc_parameters['b^{('+i_s+')}_{G_2}']['current'] *
                data.mcmc_parameters['b^{('+i_s+')}_{G_2}']['scale'])
                
                print("FIXING B1 FOR TESTING")
                print(b1)
                b1 = 1.8682
                print(b1)

                ## Load parameter means and variances   
                mean_b2, std_b2 = self.prior_b2
                mean_bG2, std_bG2 = self.prior_bG2
                mean_bGamma3, std_bGamma3 = self.prior_bGamma3(b1)
                mean_cs0, std_cs0 = self.prior_cs0
                mean_cs2, std_cs2 = self.prior_cs2
                mean_cs4, std_cs4 = self.prior_cs4
                mean_b4, std_b4 = self.prior_b4
                mean_c1, std_c1 = self.prior_c1
                mean_a0, std_a0 = self.prior_a0
                mean_a2, std_a2 = self.prior_a2
                mean_Pshot, std_Pshot = self.prior_Pshot
                mean_Bshot, std_Bshot = self.prior_Bshot
                mean_bphi, std_bphi = self.prior_bphi
                
                # Define local variables 
                dataset = self.dataset
                nP, nQ, nB, nAP = dataset.nP, dataset.nQ, dataset.nB, dataset.nAP
                
                # Compute useful quantities for AP parameters
                if self.use_AP or self.use_B:
                        DA_th = cosmo.angular_distance(z)
                        rs_th = cosmo.rs_drag()
                        Hz_th = cosmo.Hubble(z)

                # Create output arrays
                theory_minus_data = np.zeros(3*nP+nB+nQ+nAP)
                deriv_bGamma3, deriv_Pshot, deriv_Bshot, deriv_c1, deriv_a0, deriv_a2, deriv_cs0, deriv_cs2, deriv_cs4, deriv_b4, deriv_bphi, deriv_c1 = [np.zeros(3*nP+nB+nQ+nAP) for _ in range(12)]

                if self.bin_integration_P:
                        k_grid = np.linspace(np.log(1.e-4),np.log(max(dataset.kPQ)+0.01),100)
                        k_grid = np.exp(k_grid)
                else:
                        k_grid = dataset.kPQ

                # Run CLASS-PT
                all_theory = cosmo.get_pk_mult(k_grid*h,z,len(k_grid))

                # Load fNL utilities
                Pk_lin_table1 = -1.*norm**2.*(all_theory[10]/h**2./k_grid**2)*h**3
                Pk_lin_table2 = norm**2.*(all_theory[14])*h**3.
                P0int = interpolate.InterpolatedUnivariateSpline(k_grid,Pk_lin_table1,ext=3)
                Tfunc = lambda k: (P0int(k)/(As*2.*np.pi**2.*((k*h/0.05)**(cosmo.n_s()-1.))/k**3.))**0.5
                
                ## Compute Pk
                if self.use_P:

                        # Define PkTheory class, used to compute power spectra and derivatives
                        pk_theory = PkTheory(self, all_theory, h, As, fNL_eq, fNL_orth, norm, fz, k_grid, dataset.kPQ, nP, nQ, Tfunc(k_grid))
                        
                        # Compute theory model for Pl and add to (theory - data)
                        P0, P2, P4 = pk_theory.compute_Pl_oneloop(b1, b2, bG2, mean_bGamma3, mean_cs0, mean_cs2, mean_cs4, mean_b4, mean_a0, mean_a2, self.inv_nbar, mean_Pshot, mean_bphi)
                        theory_minus_data[0*nP:1*nP] = P0 - dataset.P0
                        theory_minus_data[1*nP:2*nP] = P2 - dataset.P2
                        theory_minus_data[2*nP:3*nP] = P4 - dataset.P4

                        # Compute derivatives of Pl with respect to parameters
                        deriv_bGamma3P, deriv_cs0P, deriv_cs2P, deriv_cs4P, deriv_b4P, deriv_PshotP, deriv_a0P, deriv_a2P, deriv_bphiP = pk_theory.compute_Pl_derivatives(b1)
                        
                        # Add to joint derivative vector
                        deriv_bGamma3[:3*nP] = deriv_bGamma3P
                        deriv_cs0[:3*nP] = deriv_cs0P
                        deriv_cs2[:3*nP] = deriv_cs2P
                        deriv_cs4[:3*nP] = deriv_cs4P
                        deriv_b4[:3*nP] = deriv_b4P
                        deriv_Pshot[:3*nP] = deriv_PshotP
                        deriv_a0[:3*nP] = deriv_a0P
                        deriv_a2[:3*nP] = deriv_a2P
                        deriv_bphi[:3*nP] = deriv_bphiP
                        
                ## Compute Q0
                if self.use_Q:
                        
                        # Compute theoretical Q0 model and add to (theory - data)
                        Q0 = pk_theory.compute_Q0_oneloop(b1, b2, bG2, mean_bGamma3, mean_cs0, mean_cs2, mean_cs4, mean_b4, mean_a0, mean_a2, self.inv_nbar, mean_Pshot, mean_bphi)
                        theory_minus_data[3*nP:3*nP+nQ] = Q0 - dataset.Q0

                        # Compute derivatives of Q0 with respect to parameters
                        deriv_bGamma3Q, deriv_cs0Q, deriv_cs2Q, deriv_cs4Q, deriv_b4Q, deriv_PshotQ, deriv_a0Q, deriv_a2Q, deriv_bphiQ = pk_theory.compute_Q0_derivatives(b1)

                        # Add to joint derivative vector
                        deriv_bGamma3[3*nP:3*nP+nQ] = deriv_bGamma3Q
                        deriv_cs0[3*nP:3*nP+nQ] = deriv_cs0Q
                        deriv_cs2[3*nP:3*nP+nQ] = deriv_cs2Q
                        deriv_cs4[3*nP:3*nP+nQ] = deriv_cs4Q
                        deriv_b4[3*nP:3*nP+nQ] = deriv_b4Q
                        deriv_Pshot[3*nP:3*nP+nQ] = deriv_PshotQ
                        deriv_a0[3*nP:3*nP+nQ] = deriv_a0Q
                        deriv_a2[3*nP:3*nP+nQ] = deriv_a2Q
                        deriv_bphi[3*nP:3*nP+nQ] = deriv_bphiQ

                #### Compute AP parameters
                if self.use_AP:  

                        # Compute theoretical AP model and add to (theory - data)
                        A_par = self.rdHfid/(rs_th*Hz_th)
                        A_perp = self.rdDAfid/(rs_th/DA_th)
                        theory_minus_data[-2] = A_par - dataset.alphas[0]
                        theory_minus_data[-1] = A_perp - dataset.alphas[1]

                #### Compute bispectrum
                if self.use_B:

                        # Define coordinate rescaling parameters and BAO parameters
                        apar=self.Hz_fid/(Hz_th*(self.h_fid/h)/3.33564095198145e-6) # including kmsMpc conversion factor
                        aperp=DA_th/self.DA_fid/(self.h_fid/h)
                        r_bao = cosmo.rs_drag()*h

                        # Load the theory model class
                        bk_theory = BkTheory(self, As, fNL_eq, fNL_orth, apar, aperp, fz, r_bao, k_grid, Pk_lin_table1, Pk_lin_table2, self.gauss_w, self.gauss_w2, self.mesh_mu, nB)

                        # Compute the tree-level bispectrum and parameter derivatives
                        B0, deriv_PshotB, deriv_BshotB, deriv_c1B = bk_theory.compute_B0_tree_theory_derivs(b1, b2, bG2, mean_c1, mean_Pshot, mean_Bshot)

                        # Add B0 to (theory - data)
                        theory_minus_data[3*nP+nQ:3*nP+nQ+nB] = B0 - dataset.B0

                        # Add derivatives to joint derivative vector
                        deriv_Pshot[3*nP+nQ:3*nP+nQ+nB] = deriv_PshotB
                        deriv_Bshot[3*nP+nQ:3*nP+nQ+nB] = deriv_BshotB
                        deriv_c1[3*nP+nQ:3*nP+nQ+nB] = deriv_c1B

                
                # Assemble full covariance including nuisance parameter marginalizations
                marg_cov = dataset.cov + std_bGamma3*np.outer(deriv_bGamma3,deriv_bGamma3) + std_Pshot**2.*np.outer(deriv_Pshot,deriv_Pshot) + std_a0**2.*np.outer(deriv_a0,deriv_a0) + std_a2**2.*np.outer(deriv_a2,deriv_a2) + std_cs4**2.*np.outer(deriv_cs4,deriv_cs4)+std_cs2**2.*np.outer(deriv_cs2,deriv_cs2)+std_cs0**2.*np.outer(deriv_cs0,deriv_cs0) + std_b4**2.*np.outer(deriv_b4,deriv_b4) + std_Bshot**2.*np.outer(deriv_Bshot,deriv_Bshot) + std_c1**2.*np.outer(deriv_c1,deriv_c1)  + std_bphi**2.*np.outer(deriv_bphi,deriv_bphi)

                # Compute chi2
                chi2 = np.inner(theory_minus_data,np.inner(np.linalg.inv(marg_cov),theory_minus_data))
                
                # Correct covariance matrix normalization
                chi2 += np.linalg.slogdet(marg_cov)[1] - dataset.logdetcov
                
                # Add bias parameter priors
                chi2 += (b2-mean_b2)**2./std_b2**2. + (bG2-mean_bG2)**2./std_bG2**2.
                
                return -0.5*chi2
