import numpy as np
from montepython.likelihood_class import Likelihood_prior
from scipy import interpolate
from scipy.special.orthogonal import p_roots
import os,sys
path=os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)
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
                shape = np.ones(self.dataset.nz)
                self.prior_b2 = 0.*shape, 1.*shape
                self.prior_bG2 = 0.*shape, 1.*shape
                self.prior_bGamma3 = lambda b1: (23.*(b1-1.)/42., 1.*shape) 
                self.prior_cs0 = 0.*shape, 30.*shape
                self.prior_cs2 = 30.*shape, 30.*shape
                self.prior_cs4 = 0.*shape, 30.*shape
                self.prior_b4 = 500.*shape, 500.*shape
                self.prior_c1 = 0.*shape, 5.*shape
                self.prior_Pshot = 0.*shape, self.inv_nbar
                self.prior_Bshot = 1.*shape, self.inv_nbar # NB: different convention for mean and variance
                self.prior_a0 = 0.*shape, self.inv_nbar
                self.prior_a2 = 0.*shape, self.inv_nbar
                self.prior_bphi = 1.*shape, 5.*shape

                # Pre-load useful quantities for bispectra
                if self.use_B:
                        n_gauss = 3
                        n_gauss2 = 8
                        [gauss_mu,self.gauss_w], [gauss_mu2,self.gauss_w2] = p_roots(n_gauss), p_roots(n_gauss2)
                        self.mesh_mu = np.meshgrid(gauss_mu,gauss_mu,gauss_mu,gauss_mu2,gauss_mu2, sparse=True, indexing='ij')

                # Summarize input parameters
                print("#################################################\n")
                print("### Full shape likelihood\n")
                print("Redshifts: %s"%(['%.2f'%zz for zz in self.z[:self.nz]]))
                print("Power Spectrum: %s"%self.use_P)
                print("Q0: %s"%self.use_Q)
                print("Bispectra: %s"%self.use_B)
                print("AP Parameters: %s"%self.use_AP)
                print("")
                if self.use_P:
                        print("k-min (Pk): %.3f"%self.kminP)
                        print("k-max (Pk): %.3f"%self.kmaxP)
                        print("N-bins (Pk): %d"%(len(self.dataset.P0[0])))
                        print("")
                if self.use_Q:
                        print("k-min (Q0): %.3f"%self.kmaxP)
                        print("k-max (Q0): %.3f"%self.kmaxQ)
                        print("N-bins (Q0): %d"%len(self.dataset.Q0[0]))
                        print("")
                if self.use_B:
                        print("k-min (Bk): %.3f"%self.kminB)
                        print("k-max (Bk): %.3f"%self.kmaxB)
                        print("N-bins (Bk): %d"%len(self.dataset.B0[0]))
                        print("")
                print("#################################################\n")
                
                
        def loglkl(self, cosmo, data):
                """Compute the log-likelihood for a given set of cosmological and nuisance parameters. Note that this marginalizes over nuisance parameters that enter the model linearly."""

                # Load cosmological parameters
                h = cosmo.h()
                As = cosmo.A_s()
                norm = 1. # (A_s/A_s_fid)^{1/2}
                fNL_eq = (data.mcmc_parameters['f^{eq}_{NL}']['current'] * data.mcmc_parameters['f^{eq}_{NL}']['scale'])
                fNL_orth = (data.mcmc_parameters['f^{orth}_{NL}']['current'] * data.mcmc_parameters['f^{orth}_{NL}']['scale'])
                alpha_rs = (data.mcmc_parameters['alpha_{r_s}']['current'] * data.mcmc_parameters['alpha_{r_s}']['scale'])

                z = self.z[:self.nz]
                fz = np.asarray([cosmo.scale_independent_growth_factor_f(zz) for zz in z])
                
                # Load non-linear nuisance parameters
                b1 = np.asarray([(data.mcmc_parameters['b^{('+str(i_s)+')}_1']['current']*data.mcmc_parameters['b^{('+str(i_s)+')}_1']['scale']) for i_s in range(1,1+self.nz)])
                b2 = np.asarray([(data.mcmc_parameters['b^{('+str(i_s)+')}_2']['current']*data.mcmc_parameters['b^{('+str(i_s)+')}_2']['scale']) for i_s in range(1,1+self.nz)])
                bG2 = np.asarray([(data.mcmc_parameters['b^{('+str(i_s)+')}_{G_2}']['current']*data.mcmc_parameters['b^{('+str(i_s)+')}_{G_2}']['scale']) for i_s in range(1,1+self.nz)])
                
                ## Load parameter means and variances fromm self.prior_b2, ensuring each is a vector of length nz
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
                        DA_th = np.asarray([cosmo.angular_distance(zz) for zz in z])
                        Hz_th = np.asarray([cosmo.Hubble(zz) for zz in z])
                        rs_th = cosmo.rs_drag()
                        
                # Create output arrays (will be overwritten for each redshift)
                theory_minus_data = np.zeros(3*nP+nB+nQ+nAP)
                deriv_bGamma3, deriv_Pshot, deriv_Bshot, deriv_c1, deriv_a0, deriv_a2, deriv_cs0, deriv_cs2, deriv_cs4, deriv_b4, deriv_bphi, deriv_c1 = [np.zeros(3*nP+nB+nQ+nAP) for _ in range(12)]
                
                if self.use_P or self.use_B:
                        if self.bin_integration_P:
                                k_grid = np.linspace(np.log(1.e-4),np.log(max(dataset.kPQ)+0.01),100)
                                k_grid = np.exp(k_grid)
                        else:
                                k_grid = dataset.kPQ

                # Initialize output chi2
                chi2 = 0.

                # Iterate over redshift bins
                for zi in range(self.nz):

                        if self.use_P or self.use_B:
                                # Run CLASS-PT
                                all_theory = cosmo.get_pk_mult(k_grid*h, z[zi], len(k_grid), no_wiggle = self.no_wiggle, alpha_rs = alpha_rs)
                                # Load fNL utilities
                                Pk_lin_table1 = -1.*norm**2.*(all_theory[10]/h**2./k_grid**2)*h**3
                                Pk_lin_table2 = norm**2.*(all_theory[14])*h**3.
                                P0int = interpolate.InterpolatedUnivariateSpline(k_grid,Pk_lin_table1,ext=3)
                                Tfunc = lambda k: (P0int(k)/(As*2.*np.pi**2.*((k*h/0.05)**(cosmo.n_s()-1.))/k**3.))**0.5
                        
                        
                        ## Compute Pk
                        if self.use_P:

                                # Define PkTheory class, used to compute power spectra and derivatives
                                pk_theory = PkTheory(self, all_theory, h, As, fNL_eq, fNL_orth, norm, fz[zi], k_grid, dataset.kPQ, nP, nQ, Tfunc(k_grid))
                                
                                # Compute theory model for Pl and add to (theory - data)
                                P0, P2, P4 = pk_theory.compute_Pl_oneloop(b1[zi], b2[zi], bG2[zi], mean_bGamma3[zi], mean_cs0[zi], mean_cs2[zi], mean_cs4[zi], mean_b4[zi], mean_a0[zi], mean_a2[zi], self.inv_nbar[zi], mean_Pshot[zi], mean_bphi[zi])
                                theory_minus_data[0*nP:1*nP] = P0 - dataset.P0[zi]
                                theory_minus_data[1*nP:2*nP] = P2 - dataset.P2[zi]
                                theory_minus_data[2*nP:3*nP] = P4 - dataset.P4[zi]

                                # Compute derivatives of Pl with respect to parameters
                                deriv_bGamma3P, deriv_cs0P, deriv_cs2P, deriv_cs4P, deriv_b4P, deriv_PshotP, deriv_a0P, deriv_a2P, deriv_bphiP = pk_theory.compute_Pl_derivatives(b1[zi])
                                
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
                                Q0 = pk_theory.compute_Q0_oneloop(b1[zi], b2[zi], bG2[zi], mean_bGamma3[zi], mean_cs0[zi], mean_cs2[zi], mean_cs4[zi], mean_b4[zi], mean_a0[zi], mean_a2[zi], self.inv_nbar[zi], mean_Pshot[zi], mean_bphi[zi])
                                theory_minus_data[3*nP:3*nP+nQ] = Q0 - dataset.Q0[zi]

                                # Compute derivatives of Q0 with respect to parameters
                                deriv_bGamma3Q, deriv_cs0Q, deriv_cs2Q, deriv_cs4Q, deriv_b4Q, deriv_PshotQ, deriv_a0Q, deriv_a2Q, deriv_bphiQ = pk_theory.compute_Q0_derivatives(b1[zi])

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
                                A_par = self.rdHfid[zi]/(rs_th*Hz_th[zi])
                                A_perp = self.rdDAfid[zi]/(rs_th/DA_th[zi])
                                theory_minus_data[-2] = A_par - dataset.alphas[zi][0]
                                theory_minus_data[-1] = A_perp - dataset.alphas[zi][1]

                        #### Compute bispectrum
                        if self.use_B:

                                # Define coordinate rescaling parameters and BAO parameters
                                apar=self.Hz_fid[zi]/(Hz_th[zi]*(self.h_fid/h)/3.33564095198145e-6) # including kmsMpc conversion factor
                                aperp=DA_th[zi]/self.DA_fid[zi]/(self.h_fid/h)
                                r_bao = rs_th*h

                                # Load the theory model class
                                bk_theory = BkTheory(self, As, fNL_eq, fNL_orth, apar, aperp, fz[zi], r_bao, k_grid, T_func, Pk_lin_table1, Pk_lin_table2, self.inv_nbar[zi], self.gauss_w, self.gauss_w2, self.mesh_mu, nB)

                                # Compute the tree-level bispectrum and parameter derivatives
                                B0, deriv_PshotB, deriv_BshotB, deriv_c1B = bk_theory.compute_B0_tree_theory_derivs(b1[zi], b2[zi], bG2[zi], mean_c1[zi], mean_Pshot[zi], mean_Bshot[zi])

                                # Add B0 to (theory - data), including discreteness weights
                                theory_minus_data[3*nP+nQ:3*nP+nQ+nB] = B0*dataset.discreteness_weights[zi] - dataset.B0[zi]

                                # Add derivatives to joint derivative vector
                                deriv_Pshot[3*nP+nQ:3*nP+nQ+nB] = deriv_PshotB
                                deriv_Bshot[3*nP+nQ:3*nP+nQ+nB] = deriv_BshotB
                                deriv_c1[3*nP+nQ:3*nP+nQ+nB] = deriv_c1B

                        # Assemble full covariance including nuisance parameter marginalizations
                        marg_cov = dataset.cov[zi]+std_bGamma3[zi]*np.outer(deriv_bGamma3,deriv_bGamma3)+std_cs0[zi]**2.*np.outer(deriv_cs0,deriv_cs0)+std_cs2[zi]**2.*np.outer(deriv_cs2,deriv_cs2)+std_cs4[zi]**2.*np.outer(deriv_cs4,deriv_cs4)+std_b4[zi]**2.*np.outer(deriv_b4,deriv_b4)+std_Pshot[zi]**2.*np.outer(deriv_Pshot,deriv_Pshot)+std_Bshot[zi]**2.*np.outer(deriv_Bshot,deriv_Bshot)+std_a0[zi]**2.*np.outer(deriv_a0,deriv_a0)+std_a2[zi]**2.*np.outer(deriv_a2,deriv_a2)+std_c1[zi]**2.*np.outer(deriv_c1,deriv_c1)+std_bphi[zi]**2.*np.outer(deriv_bphi,deriv_bphi)
                        
                        # Compute chi2 from data and theory
                        chi2 += np.inner(theory_minus_data,np.inner(np.linalg.inv(marg_cov),theory_minus_data))
                        
                        # Correct covariance matrix normalization
                        chi2 += np.linalg.slogdet(marg_cov)[1] - dataset.logdetcov[zi]
                        
                        # Add bias parameter priors
                        chi2 += (b2[zi]-mean_b2[zi])**2./std_b2[zi]**2. + (bG2[zi]-mean_bG2[zi])**2./std_bG2[zi]**2.

                # Return full loglikelihood
                loglkl = -0.5*chi2

                return loglkl
