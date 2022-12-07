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
                
                # Copy in attributes (necessary in py3 for some reason...)
                for attr in extra_inputs.__dict__.keys():
                        setattr(self,attr,getattr(extra_inputs,attr))

                # Initialize the  likelihood
                Likelihood_prior.__init__(self,path,data,command_line)

                # Check whether to use one-loop Bk
                if self.use_B and not hasattr(self,'oneloop_B'):
                        self.oneloop_B = False
                if hasattr(self,'oneloop_B'):
                        raise Exception("Not yet fully implemented!")                        
                # Check ell-max
                if not hasattr(self, 'lmax'):
                        self.lmax = 4
                if self.lmax>4:
                        raise Exception("Multipoles beyond the hexadecapole not yet implemented!")
                self.nl = self.lmax//2+1

                # Load the data
                self.dataset = Datasets(self)

                # Define nuisance parameter mean and standard deviations
                shape = np.ones(self.dataset.nz)
                # prior based on b2(b1) relation
                self.prior_b2 = lambda b1: (0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**2 + 8./21.*(b1 - 1), 1.*shape)
                # self.prior_b2 = 0.*shape, 1.*shape # original prior
                # prior based on bG2(b1) relation
                self.prior_bG2 = lambda b1: (-2./7.*(b1-1), 1.*shape)
                # self.prior_bG2 = 0.*shape, 1.*shape # original prior
                self.prior_bGamma3 = lambda b1: (23.*(b1-1.)/42., 1.*shape) 
                self.prior_cs0 = 0.*shape, 30.*shape
                self.prior_cs2 = 30.*shape, 30.*shape
                self.prior_cs4 = 0.*shape, 30.*shape
                self.prior_b4 = 500.*shape, 500.*shape
                self.prior_c1 = 0.*shape, 5.*shape
                self.prior_Pshot = 0.*shape, self.inv_nbar # NB: different convention for mean and variance
                self.prior_Bshot = 1.*shape, self.inv_nbar 
                self.prior_a0 = 0.*shape, self.inv_nbar
                self.prior_a2 = 0.*shape, self.inv_nbar
                self.prior_bphi = 1.*shape, 5.*shape # non-Gaussian bias, only relevant to fNL

                # Define priors for one-loop parameters
                if self.oneloop_B:
                        self.prior_b3 = 0.*shape, 5.*shape
                        self.prior_g3 = 0.*shape, 5.*shape
                        self.prior_g2x = 0.*shape, 5.*shape
                        self.prior_g22 = 0.*shape, 5.*shape
                        self.prior_g21x = 0.*shape, 5.*shape
                        self.prior_g31 = 0.*shape, 5.*shape
                        self.prior_g211 = 0.*shape, 5.*shape
                        self.prior_eps2 = 0.*shape, self.inv_nbar
                        self.prior_eta21 = 0.*shape, self.inv_nbar
                        self.prior_eta22 = 0.*shape, self.inv_nbar
                        self.prior_betaBa = 0.*shape, 5.*shape
                        self.prior_betaBb = 0.*shape, 5.*shape
                        self.prior_betaBc = 0.*shape, 5.*shape
                        self.prior_betaBd = 0.*shape, 5.*shape
                        self.prior_betaBe = 0.*shape, 5.*shape
                
                # Pre-load useful quantities for bispectra
                if self.use_B:
                        [gauss_mu,self.gauss_w], [gauss_mu2,self.gauss_w2] = p_roots(self.n_gauss), p_roots(self.n_gauss2)
                        self.mesh_mu = np.meshgrid(gauss_mu,gauss_mu,gauss_mu,gauss_mu2,gauss_mu2, sparse=True, indexing='ij')

                # Summarize input parameters
                print("#################################################\n")
                print("### Full shape likelihood\n")
                print("Redshifts: %s"%(['%.2f'%zz for zz in self.z[:self.nz]]))
                print("Power Spectrum: %s"%self.use_P)
                print("Q0: %s"%self.use_Q)
                if self.use_B and not self.oneloop_B:
                        print("Bispectra: Tree-Level")
                elif self.use_B and self.oneloop_B:
                        print("Bispectra: One-Loop")
                else:
                        print("Bispectra: False")
                print("AP Parameters: %s"%self.use_AP)
                print("")
                for zi in range(self.nz):
                        print("# z-bin %d #"%zi)
                        if self.use_P:
                                print("k-min (Pk): %.3f"%self.kminP[zi])
                                print("k-max (Pk): %.3f"%self.kmaxP[zi])
                                print("N-bins (Pk): %d"%(len(self.dataset.P0[zi])))
                                print("l-max (Pk): %d"%self.lmax)
                                print("")
                        if self.use_Q:
                                print("k-min (Q0): %.3f"%self.kmaxP[zi])
                                print("k-max (Q0): %.3f"%self.kmaxQ[zi])
                                print("N-bins (Q0): %d"%len(self.dataset.Q0[zi]))
                                print("")
                        if self.use_B:
                                print("k-min (Bk): %.3f"%self.kminB[zi])
                                print("k-max (Bk): %.3f"%self.kmaxB[zi])
                                print("N-bins (Bk): %d"%len(self.dataset.B0[zi]))
                                print("")
                print("#################################################\n")
                
         def loglkl(self, cosmo, data):
                """Compute the log-likelihood for a given set of cosmological and nuisance parameters. Note that this marginalizes over nuisance parameters that enter the model linearly."""

                # Load cosmological parameters
                h = cosmo.h()
                As = cosmo.A_s()
                norm = 1. # (A_s/A_s_fid)^{1/2}
                if norm != 1. and self.oneloop_B:
                        raise Exception("One-loop Bk is not implemented for norm != 1.")
                fNL_eq = (data.mcmc_parameters['f^{eq}_{NL}']['current'] * data.mcmc_parameters['f^{eq}_{NL}']['scale'])
                fNL_orth = (data.mcmc_parameters['f^{orth}_{NL}']['current'] * data.mcmc_parameters['f^{orth}_{NL}']['scale'])
                alpha_rs = (data.mcmc_parameters['alpha_{r_s}']['current'] * data.mcmc_parameters['alpha_{r_s}']['scale'])

                # Check we're running in the correct mode
                if fNL_eq!=0 or fNL_orth!=0:
                        try:
                                 data.cosmo_arguments['PNG']
                        except:
                                raise Exception('PNG mode must be turned on to use fNL!')
                        assert data.cosmo_arguments['PNG']=='Yes', 'PNG mode must be turned on to use fNL!'

                z = self.z[:self.nz]
                fz = np.asarray([cosmo.scale_independent_growth_factor_f(zz) for zz in z])
                
                # Load non-linear nuisance parameters
                b1 = np.asarray([(data.mcmc_parameters['b^{('+str(i_s)+')}_1']['current']*data.mcmc_parameters['b^{('+str(i_s)+')}_1']['scale']) for i_s in range(1,1+self.nz)])
                b2 = np.asarray([(data.mcmc_parameters['b^{('+str(i_s)+')}_2']['current']*data.mcmc_parameters['b^{('+str(i_s)+')}_2']['scale']) for i_s in range(1,1+self.nz)])
                bG2 = np.asarray([(data.mcmc_parameters['b^{('+str(i_s)+')}_{G_2}']['current']*data.mcmc_parameters['b^{('+str(i_s)+')}_{G_2}']['scale']) for i_s in range(1,1+self.nz)])
                
                if hasattr(self,'bias_relations'):
                    if self.bias_relations:
                        # Replace quadratic biases with bias relation predictions
                        b2 = 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**2 + 8./21.*(b1 - 1)
                        bG2 = -2./7.*(b1-1)
                
                ## Load parameter means and variances from self.prior_b2, ensuring each is a vector of length nz
                mean_b2, std_b2 = self.prior_b2(b1)
                mean_bG2, std_bG2 = self.prior_bG2(b1)
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
                if self.oneloop_B:
                        mean_b3, std_b3 = self.prior_b3
                        mean_g3, std_g3 = self.prior_g3
                        mean_g2x, std_g2x = self.prior_g2x
                        mean_g22, std_g22 = self.prior_g22
                        mean_g21x, std_g21x = self.prior_g21x
                        mean_g31, std_g31 = self.prior_g31
                        mean_g211, std_g211 = self.prior_g211
                        mean_eps2, std_eps2 = self.prior_eps2
                        mean_eta21, std_eta21 = self.prior_eta21
                        mean_eta22, std_eta22 = self.prior_eta22
                        mean_betaBa, std_betaBa = self.prior_betaBa
                        mean_betaBb, std_betaBb = self.prior_betaBb
                        mean_betaBc, std_betaBc = self.prior_betaBc
                        mean_betaBd, std_betaBd = self.prior_betaBd
                        mean_betaBe, std_betaBe = self.prior_betaBe

                # Define local variables 
                dataset = self.dataset
                
                # Compute useful quantities for AP parameters
                if self.use_AP or self.use_B:
                        DA_th = np.asarray([cosmo.angular_distance(zz) for zz in z])
                        Hz_th = np.asarray([cosmo.Hubble(zz) for zz in z])
                        rs_th = cosmo.rs_drag()
                
                # Initialize output chi2
                chi2 = 0.

                # Iterate over redshift bins
                for zi in range(self.nz):
                    
                        nP, nQ, nB, nAP = dataset.nP[zi], dataset.nQ[zi], dataset.nB[zi], dataset.nAP
                        
                        # Create output arrays
                        theory_minus_data = np.zeros(self.nl*nP+nB+nQ+nAP)
                        deriv_bGamma3, deriv_Pshot, deriv_Bshot, deriv_c1, deriv_a0, deriv_a2, deriv_cs0, deriv_cs2, deriv_cs4, deriv_b4, deriv_bphi, deriv_c1 = [np.zeros(self.nl*nP+nB+nQ+nAP) for _ in range(12)]
                        if self.oneloop_B:
                                deriv_b3, deriv_g3, deriv_g2x, deriv_g22, deriv_g21x, deriv_g31, deriv_g211, deriv_eps2, deriv_eta21, deriv_eta22, deriv_betaBa, deriv_betaBb, deriv_betaBc, deriv_betaBd, deriv_betaBe = [np.zeros(self.nl*nP+nB+nQ+nAP) for _ in range(15)]
                     
                        # Define k grid
                        if self.use_P or self.use_B:
                            if self.bin_integration_P:
                                    k_grid = np.exp(np.linspace(np.log(1.e-4),np.log(max(dataset.kPQ[zi])+0.01),100))
                            else:
                                    k_grid = dataset.kPQ[zi]


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
                                pk_theory = PkTheory(self, zi, all_theory, h, As, fNL_eq, fNL_orth, norm, fz[zi], k_grid, dataset.kPQ[zi], nP, nQ, Tfunc(k_grid))
                                
                                # Compute theory model for Pl and add to (theory - data)
                                P0, P2, P4 = pk_theory.compute_Pl_oneloop(b1[zi], b2[zi], bG2[zi], mean_bGamma3[zi], mean_cs0[zi], mean_cs2[zi], mean_cs4[zi], mean_b4[zi], mean_a0[zi], mean_a2[zi], self.inv_nbar[zi], mean_Pshot[zi], mean_bphi[zi])
                                theory_minus_data[0*nP:1*nP] = P0 - dataset.P0[zi]
                                if self.lmax>0:
                                        theory_minus_data[1*nP:2*nP] = P2 - dataset.P2[zi]
                                if self.lmax>2:
                                        theory_minus_data[2*nP:3*nP] = P4 - dataset.P4[zi]
                                
                                # Compute derivatives of Pl with respect to parameters
                                deriv_bGamma3P, deriv_cs0P, deriv_cs2P, deriv_cs4P, deriv_b4P, deriv_PshotP, deriv_a0P, deriv_a2P, deriv_bphiP = pk_theory.compute_Pl_derivatives(b1[zi])
                                
                                # Add to joint derivative vector
                                deriv_bGamma3[:self.nl*nP] = deriv_bGamma3P[:self.nl*nP]
                                deriv_cs0[:self.nl*nP] = deriv_cs0P[:self.nl*nP]
                                deriv_cs2[:self.nl*nP] = deriv_cs2P[:self.nl*nP]
                                deriv_cs4[:self.nl*nP] = deriv_cs4P[:self.nl*nP]
                                deriv_b4[:self.nl*nP] = deriv_b4P[:self.nl*nP]
                                deriv_Pshot[:self.nl*nP] = deriv_PshotP[:self.nl*nP]
                                deriv_a0[:self.nl*nP] = deriv_a0P[:self.nl*nP]
                                deriv_a2[:self.nl*nP] = deriv_a2P[:self.nl*nP]
                                deriv_bphi[:self.nl*nP] = deriv_bphiP[:self.nl*nP]
                                
                        ## Compute Q0
                        if self.use_Q:
                                
                                # Compute theoretical Q0 model and add to (theory - data)
                                Q0 = pk_theory.compute_Q0_oneloop(b1[zi], b2[zi], bG2[zi], mean_bGamma3[zi], mean_cs0[zi], mean_cs2[zi], mean_cs4[zi], mean_b4[zi], mean_a0[zi], mean_a2[zi], self.inv_nbar[zi], mean_Pshot[zi], mean_bphi[zi])
                                theory_minus_data[self.nl*nP:self.nl*nP+nQ] = Q0 - dataset.Q0[zi]
                                
                                # Compute derivatives of Q0 with respect to parameters
                                deriv_bGamma3Q, deriv_cs0Q, deriv_cs2Q, deriv_cs4Q, deriv_b4Q, deriv_PshotQ, deriv_a0Q, deriv_a2Q, deriv_bphiQ = pk_theory.compute_Q0_derivatives(b1[zi])

                                # Add to joint derivative vector
                                deriv_bGamma3[self.nl*nP:self.nl*nP+nQ] = deriv_bGamma3Q
                                deriv_cs0[self.nl*nP:self.nl*nP+nQ] = deriv_cs0Q
                                deriv_cs2[self.nl*nP:self.nl*nP+nQ] = deriv_cs2Q
                                deriv_cs4[self.nl*nP:self.nl*nP+nQ] = deriv_cs4Q
                                deriv_b4[self.nl*nP:self.nl*nP+nQ] = deriv_b4Q
                                deriv_Pshot[self.nl*nP:self.nl*nP+nQ] = deriv_PshotQ
                                deriv_a0[self.nl*nP:self.nl*nP+nQ] = deriv_a0Q
                                deriv_a2[self.nl*nP:self.nl*nP+nQ] = deriv_a2Q
                                deriv_bphi[self.nl*nP:self.nl*nP+nQ] = deriv_bphiQ

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
                                bk_theory = BkTheory(self, zi, As, fNL_eq, fNL_orth, apar, aperp, fz[zi], cosmo.sigma8(), r_bao, k_grid, Tfunc, Pk_lin_table1, Pk_lin_table2, self.inv_nbar[zi], self.gauss_w, self.gauss_w2, self.mesh_mu, nB)

                                # Compute the tree-level bispectrum and parameter derivatives
                                if not self.oneloop_B:
                                        bias_list = [b1[zi],b2[zi],bG2[zi],mean_c1[zi],mean_Pshot[zi],mean_Bshot[zi]]
                                else:
                                        bias_list = [b1[zi], b2[zi], bG2[zi], mean_bGamma3[zi], mean_b3[zi], mean_g3[zi], mean_g2x[zi], mean_g22[zi], mean_g21x[zi], mean_g31[zi], mean_g211[zi], mean_c1[zi], mean_Pshot[zi], mean_Bshot[zi], mean_eps2[zi], mean_eta21[zi], mean_eta22[zi], mean_betaBa[zi], mean_betaBb[zi], mean_betaBc[zi], mean_betaBd[zi], mean_betaBe[zi]]
                                B0, derivs = bk_theory.compute_B0_theory_derivs(bias_list)

                                # Add B0 to (theory - data), including discreteness weights
                                theory_minus_data[self.nl*nP+nQ:self.nl*nP+nQ+nB] = B0*dataset.discreteness_weights[zi] - dataset.B0[zi]
                                
                                # Add derivatives to joint derivative vector
                                deriv_Pshot[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[0]
                                deriv_Bshot[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[1]
                                deriv_c1[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[2]

                                if self.oneloop_B:
                                        deriv_b3[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[3]
                                        deriv_g3[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[4]
                                        deriv_bGamma3[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[5]*(-4./7.) # switch to bGamma3 convention here!
                                        deriv_g2x[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[6]
                                        deriv_g22[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[7]
                                        deriv_g21x[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[8]
                                        deriv_g31[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[9]
                                        deriv_g211[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[10]
                                        deriv_eps2[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[11]
                                        deriv_eta21[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[12]
                                        deriv_eta22[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[13]
                                        deriv_betaBa[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[14]
                                        deriv_betaBb[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[15]
                                        deriv_betaBc[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[16]
                                        deriv_betaBd[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[17]
                                        deriv_betaBe[self.nl*nP+nQ:self.nl*nP+nQ+nB] = derivs[18]

                        # Assemble full inverse covariance including nuisance parameter marginalizations
                        def update_icov(icov_in, logdet_in, v):
                            """Update the inverse covariance and log(determinant) via the Sherman-Morrison identity"""
                            icov_v = np.inner(icov_in,v)
                            v_icov_v = np.inner(v,icov_v)
                            icov_out = icov_in - np.outer(icov_v,icov_v)/(1.+v_icov_v)
                            logdet_out = logdet_in + np.log(1.+v_icov_v)
                            return icov_out, logdet_out
                        
                        # Compute marginalized inverse covariance
                        icov, logdet = dataset.icov[zi], dataset.logdetcov[zi]
                        icov, logdet = update_icov(icov, logdet, std_bGamma3[zi]*deriv_bGamma3)
                        icov, logdet = update_icov(icov, logdet, std_cs0[zi]*deriv_cs0)
                        icov, logdet = update_icov(icov, logdet, std_cs2[zi]*deriv_cs2)
                        icov, logdet = update_icov(icov, logdet, std_cs4[zi]*deriv_cs4)
                        icov, logdet = update_icov(icov, logdet, std_b4[zi]*deriv_b4)
                        icov, logdet = update_icov(icov, logdet, std_Pshot[zi]*deriv_Pshot)
                        icov, logdet = update_icov(icov, logdet, std_Bshot[zi]*deriv_Bshot)
                        icov, logdet = update_icov(icov, logdet, std_a0[zi]*deriv_a0)
                        icov, logdet = update_icov(icov, logdet, std_a2[zi]*deriv_a2)
                        icov, logdet = update_icov(icov, logdet, std_c1[zi]*deriv_c1)
                        icov, logdet = update_icov(icov, logdet, std_bphi[zi]*deriv_bphi)
                        
                        if self.oneloop_B:
                            icov, logdet = update_icov(icov, logdet, std_b3[zi]*deriv_b3)
                            icov, logdet = update_icov(icov, logdet, std_g3[zi]*deriv_g3)
                            icov, logdet = update_icov(icov, logdet, std_g2x[zi]*deriv_g2x)
                            icov, logdet = update_icov(icov, logdet, std_g22[zi]*deriv_g22)
                            icov, logdet = update_icov(icov, logdet, std_g21x[zi]*deriv_g21x)
                            icov, logdet = update_icov(icov, logdet, std_g31[zi]*deriv_g31)
                            icov, logdet = update_icov(icov, logdet, std_g211[zi]*deriv_g211)
                            icov, logdet = update_icov(icov, logdet, std_eps2[zi]*deriv_eps2)
                            icov, logdet = update_icov(icov, logdet, std_eta21[zi]*deriv_eta21)
                            icov, logdet = update_icov(icov, logdet, std_eta22[zi]*deriv_eta22)
                            icov, logdet = update_icov(icov, logdet, std_betaBa[zi]*deriv_betaBa)
                            icov, logdet = update_icov(icov, logdet, std_betaBb[zi]*deriv_betaBb)
                            icov, logdet = update_icov(icov, logdet, std_betaBc[zi]*deriv_betaBc)
                            icov, logdet = update_icov(icov, logdet, std_betaBd[zi]*deriv_betaBd)
                            icov, logdet = update_icov(icov, logdet, std_betaBe[zi]*deriv_betaBe)
                            
                        # Compute chi2 from data and theory
                        chi2 += np.inner(theory_minus_data,np.inner(icov,theory_minus_data))
                        
                        # Correct covariance matrix normalization
                        chi2 += logdet - dataset.logdetcov[zi]
                        
                        # Add bias parameter priors for explicitly marginalized parameters
                        chi2 += (b2[zi]-mean_b2[zi])**2./std_b2[zi]**2. + (bG2[zi]-mean_bG2[zi])**2./std_bG2[zi]**2.
                        
                # Return full loglikelihood
                loglkl = -0.5*chi2

                return loglkl

