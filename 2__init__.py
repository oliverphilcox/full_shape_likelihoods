import os
import numpy as np
from montepython.likelihood_class import Likelihood_prior
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.integrate import quad
import scipy.integrate as integrate
from numpy import log, exp, sin, cos
from scipy.special.orthogonal import p_roots

class ngcz3_bxp(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.

    def __init__(self,path,data,command_line):
        Likelihood_prior.__init__(self,path,data,command_line)

        self.n_gauss = 3
        [self.gauss_mu,self.gauss_w]=p_roots(self.n_gauss)

        self.n_gauss2 = 8
        [self.gauss_mu2,self.gauss_w2]=p_roots(self.n_gauss2)

        triag = [ [0 for x in range(self.ksize*self.ksize*self.ksize)] for y in range(3)]
        ntriag = self.ntriag

        self.weights = np.zeros(ntriag,'float64')
        datafile = open(os.path.join(self.data_directory, self.weights_file), 'r')
        for i in range(ntriag):
            line = datafile.readline()
            self.weights[i] = float(line.split()[0])
        datafile.close()
	self.Bk = np.zeros(ntriag,'float64')

        khere = np.zeros(ntriag,'float64')
        khere2 = np.zeros(ntriag,'float64')
        khere3 = np.zeros(ntriag,'float64')
        datafile = open(os.path.join(self.data_directory, self.measurements_b_file), 'r')
        for i in range(ntriag):
            line = datafile.readline()
            khere[i] = float(line.split()[0])
            khere2[i] = float(line.split()[1])
            khere3[i] = float(line.split()[2])
            self.Bk[i] = float(line.split()[3])
        datafile.close()
        self.kf = 0.00163625

        self.k = np.zeros(self.ksize, 'float64')
        self.k = np.linspace(self.kmin,self.kmax,self.ksize)

### this is just taken from the file
        self.new_triag = [ [0 for x in range(ntriag)] for y in range(3)]
        self.dk = 0.01
        for j in range(ntriag):
                self.new_triag[0][j] = int(khere[j]/self.dk) - int(self.kmin/self.dk)
                self.new_triag[1][j] = int(khere2[j]/self.dk) - int(self.kmin/self.dk)
                self.new_triag[2][j] = int(khere3[j]/self.dk) - int(self.kmin/self.dk)

        self.count = 0
        self.omit = 0
        dx=np.loadtxt(os.path.join(self.data_directory, self.measurements_p_file), skiprows = 0)
        self.k1 = dx[:,0]
        self.Pk01 = dx[:,1]
        self.Pk21 = dx[:,2]
        self.Pk41 = dx[:,3]
        del dx
        for i in range(self.ksize1):
                if (self.k1[i] <= self.kmax2 and self.k1[i] >= self.kmin2):
                        self.count = self.count + 1
                if self.k1[i] < self.kmin2:
                        self.omit = self.omit + 1
        self.ksize2 = self.count

        self.k2 = np.zeros(self.ksize2,'float64')
        self.Pk0 = np.zeros(self.ksize2,'float64')
        self.Pk2 = np.zeros(self.ksize2,'float64')
        self.Pk4 = np.zeros(self.ksize2,'float64')

        for i in range(self.ksize2):
                self.k2[i] = self.k1[i + self.omit]
                self.Pk0[i] = self.Pk01[i + self.omit]
                self.Pk2[i] = self.Pk21[i + self.omit]
		self.Pk4[i] = self.Pk41[i + self.omit]

	

#        self.invcov = np.loadtxt(os.path.join(self.data_directory, self.covmat_file))
#        self.invcov = np.zeros(
#            (3*3*self.ksize2+self.ntriag, 3*self.ksizep0p2 +self.ksizemu0+self.ntriag), 'float64')
        self.cov1 = np.zeros((3*self.ksize1+self.ntriag, 3*self.ksize1 +self.ntriag), 'float64')
        self.cov = np.zeros((3*self.ksize2+self.ntriag, 3*self.ksize2 +self.ntriag), 'float64')

        datafile = open(os.path.join(self.data_directory, self.covmat_file), 'r')
        for i in range(3*self.ksize1+self.ntriag):
            line = datafile.readline()
            while line.find('#') != -1:
                line = datafile.readline()
            for j in range(3*self.ksize1+self.ntriag):
                self.cov1[i,j] = float(line.split()[j])

        for i in range(self.ksize2):
                for j in range(self.ksize2):
                        self.cov[i][j] = self.cov1[self.omit + i, self.omit + j]
                        self.cov[i][j+self.ksize2] = self.cov1[self.omit + i,self.omit + j + self.ksize1]
                        self.cov[i + self.ksize2][j] = self.cov1[self.omit + i + self.ksize1,self.omit + j]
                        self.cov[i + self.ksize2][j + self.ksize2] = self.cov1[self.omit + i + self.ksize1, self.omit + j + self.ksize1]
                        self.cov[i][j+2*self.ksize2] = self.cov1[self.omit + i,self.omit + j + 2*self.ksize1]
                        self.cov[i+2*self.ksize2][j] = self.cov1[self.omit + i + 2*self.ksize1,self.omit + j]
                        self.cov[i+self.ksize2][j+2*self.ksize2] = self.cov1[self.omit + i+ self.ksize1,self.omit + j + 2*self.ksize1]
                        self.cov[i+2*self.ksize2][j+self.ksize2] = self.cov1[self.omit + i + 2*self.ksize1,self.omit + j+ self.ksize1]
                        self.cov[i+2*self.ksize2][j+2*self.ksize2] = self.cov1[self.omit + i + 2*self.ksize1,self.omit + j+ 2*self.ksize1]
	for i in range(self.ntriag):
		for j in range(self.ntriag):
			self.cov[3*self.ksize2+i][3*self.ksize2+j] = self.cov1[3*self.ksize1 + i, 3*self.ksize1 + j]	
	if self.cross:
		for i in range(self.ksize2):
			for j in range(self.ntriag):
				self.cov[i][3*self.ksize2+j] = self.cov1[i+self.omit][3*self.ksize1+j]
                        	self.cov[i+self.ksize2][3*self.ksize2+j] = self.cov1[i+self.ksize1+self.omit][3*self.ksize1+j]
                        	self.cov[i+2*self.ksize2][3*self.ksize2+j] = self.cov1[i+2*self.ksize1+self.omit][3*self.ksize1+j]
                        	self.cov[3*self.ksize2+j][i] = self.cov1[3*self.ksize1+j][i+self.omit]
                        	self.cov[3*self.ksize2+j][i+self.ksize2] = self.cov1[3*self.ksize1+j][i+self.ksize1+self.omit]
                        	self.cov[3*self.ksize2+j][i+2*self.ksize2] = self.cov1[3*self.ksize1+j][i+2*self.ksize1+self.omit]

	#self.cov = self.cov/100.
        self.invcov = np.linalg.inv(self.cov)
        datafile.close()
        self.logdetcov = np.linalg.slogdet(self.cov)[1]


        self.D1 = lambda k1,k2,k3,beta: (15. + 10.*beta+beta**2. + 2.*beta**2.*((k3**2.-k1**2.-k2**2.)/(2.*k1*k2))**2.)/15.
        self.D2 = lambda k1,k2,k3,beta: beta/3+(4 *beta**2.)/15-(k1**2. *beta**2.)/(15 *k2**2.)-(k2**2. *beta**2.)/(15 *k1**2.)-(k1**2. *beta**2.)/(30 *k3**2.)+(k1**4 *beta**2.)/(30 *k2**2. *k3**2.)-(k2**2. *beta**2.)/(30 *k3**2.)+(k2**4. *beta**2.)/(30 *k1**2. *k3**2.)+(k3**2. *beta**2.)/(30 *k1**2.)+(k3**2. *beta**2.)/(30 *k2**2.)+(2 *beta**3)/35-(k1**2. *beta**3.)/(70 *k2**2.)-(k2**2. *beta**3)/(70 *k1**2.)-(k1**2. *beta**3)/(70 *k3**2.)+(k1**4 *beta**3)/(70 *k2**2.*k3**2.)-(k2**2. *beta**3)/(70 *k3**2.)+(k2**4 *beta**3)/(70 *k1**2. *k3**2.)-(k3**2. *beta**3)/(70 *k1**2.)-(k3**2. *beta**3)/(70 *k2**2.)+(k3**4 *beta**3)/(70 *k1**2. *k2**2.)
        self.D3 = lambda k1,k2,k3,beta: beta/6-(k1**2. *beta)/(12 *k2**2.)-(k2**2. *beta)/(12 *k1**2.)+(k3**2. *beta)/(12 *k1**2.)+(k3**2. *beta)/(12 *k2**2.)+ beta**2./6-(k1**2. *beta**2.)/(12 *k2**2.)-(k2**2. *beta**2.)/(12 *k1**2.)+(k3**2. *beta**2.)/(60 *k1**2.)+(k3**2. *beta**2.)/(60 *k2**2.)+(k3**4. *beta**2.)/(15 *k1**2. *k2**2.)+(2 *beta**3.)/35-(k1**4. *beta**3.)/(140 *k2**4.)-(3 *k1**2. *beta**3.)/(140 *k2**2.)-(3 *k2**2. *beta**3.)/(140 *k1**2.)-(k2**4. *beta**3.)/(140 *k1**4.)-(k3**2. *beta**3.)/(35 *k1**2.)+(3 *k1**2. *k3**2. *beta**3.)/(140 *k2**4.)-(k3**2. *beta**3.)/(35 *k2**2.)+(3 *k2**2. *k3**2. *beta**3.)/(140 *k1**4.)-(3 *k3**4. *beta**3.)/(140 *k1**4.)-(3 *k3**4. *beta**3.)/(140 *k2**4.)+(3 *k3**4. *beta**3.)/(70 *k1**2. *k2**2.)+(k3**6 *beta**3.)/(140 *k1**2. *k2**4.)+(k3**6 *beta**3.)/(140 *k1**4. *k2**2.)+ beta**4./105-(k1**4. *beta**4.)/(420 *k2**4.)-(k1**2. *beta**4.)/(420 *k2**2.)-(k2**2. *beta**4.)/(420 *k1**2.)-(k2**4. *beta**4.)/(420 *k1**4.)-(k3**2. *beta**4.)/(105 *k1**2.)+(k1**2. *k3**2. *beta**4.)/(180 *k2**4.)-(k3**2. *beta**4.)/(105 *k2**2.)+(k2**2. *k3**2. *beta**4.)/(180 *k1**4.)-(k3**4. *beta**4.)/(420 *k1**4.)-(k3**4. *beta**4.)/(420 *k2**4.)+(k3**4. *beta**4.)/(70 *k1**2. *k2**2.)-(k3**6 *beta**4.)/(420 *k1**2. *k2**4.)-(k3**6 *beta**4.)/(420 *k1**4. *k2**2.)+(k3**8 *beta**4.)/(630 *k1**4. *k2**4.)
        self.F2 = lambda k1,k2,k3,beta,b1,b2,bG2: (b1*(-5.*(k1**2.-k2**2.)**2.+3.*(k1**2.+k2**2.)*k3**2.+2.*k3**4.)*self.D1(k1,k2,k3,beta) + b1*(-3.*(k1**2.-k2**2.)**2.-1.*(k1**2.+k2**2.)*k3**2.+4.*k3**4.)*self.D2(k1,k2,k3,beta) + 7.*self.D1(k1,k2,k3,beta)*(2.*b2*k1**2.*k2**2. + bG2*(k1-k2-k3)*(k1+k2-k3)*(k1-k2+k3)*(k1+k2+k3)))*b1**2./28./k1**2./k2**2. + b1**4.*self.D3(k1,k2,k3,beta)

        self.F2real = lambda k1,k2,k3,b1,b2,bG2: (b1*(-5.*(k1**2.-k2**2.)**2.+3.*(k1**2.+k2**2.)*k3**2.+2.*k3**4.) + 7.*(2.*b2*k1**2.*k2**2. + bG2*(k1-k2-k3)*(k1+k2-k3)*(k1-k2+k3)*(k1+k2+k3)))*b1**2./28./k1**2./k2**2.

        self.G2 = lambda k1,k2,k3: -((3*(k1**2-k2**2)**2+(k1**2+k2**2)*k3**2-4*k3**4)/(28 *k1**2 *k2**2))


        self.Bbinl0 = lambda I01,I02,I21,I22,I41,I42,I61,I62,Im21,Im22,Im41,Im42,k32,k34,k36,k38,k3m2,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19,e20,e21: 2.*(e1*I22*k3m2*I01 + e2*I61*Im42*k3m2 + e3*k32*Im22*I01 + e4*k34*Im42*I01 + e5*I01*I02 + e6*I41*Im22*k3m2 + e7*I41*Im42+ e8*I21*I02*k3m2 + e9*I21*Im22 + e10*I21*k32*Im42 + Im41*(e11*I62*k3m2 + e12*k34*I02 + e13*k36*Im22 + e14*I42 + e15*k38*Im42 + e16*I22*k32) + Im21*(e17*I42*k3m2 +e18*k32*I02 + e19*I22 + e20*k36*Im42 +e21*k34*Im22) )


        self.Bbin = lambda I01,I02,I21,I22,Im21,Im22,k32,k34,c1,c2,c3,c4,b1: 2.*(b1**2.)*(c1*I01*I02 + c2*(I21*Im22+I22*Im21)+c3*(I02*Im21+I01*Im22)*k32 +c4*(Im21*Im22)*k34)
#        self.F2 = lambda k1,k2,k3,beta,b1,b2,bG2: (b1*(-5.*(k1**2.-k2**2.)**2.+3.*(k1**2.+k2**2.)*k3**2.+2.*k3**4.)*self.D1(k1,k2,k3,beta) + b1*(-3.*(k1**2.-k2**2.)**2.-1.*(k1**2.+k2**2.)*k3**2.+4.*k3**4.)*0. + 7.*self.D1(k1,k2,k3,beta)*(2.*b2*k1**2.*k2**2. + bG2*(k1-k2-k3)*(k1+k2-k3)*(k1-k2+k3)*(k1+k2+k3)))*b1**2./28./k1**2./k2**2.

        self.j2 = lambda x: (3./x**2.-1.)*np.sin(x)/x - 3.*np.cos(x)/x**2.
 

    def loglkl(self, cosmo, data):

        h = cosmo.h()


        #norm = (data.mcmc_parameters['norm']['current'] *
        #         data.mcmc_parameters['norm']['scale'])
	norm = 1.


        i_s=repr(1)
        b1 = (data.mcmc_parameters['b^{('+i_s+')}_1']['current'] *
             data.mcmc_parameters['b^{('+i_s+')}_1']['scale'])
        b2 = (data.mcmc_parameters['b^{('+i_s+')}_2']['current'] *
             data.mcmc_parameters['b^{('+i_s+')}_2']['scale'])
        bG2 = (data.mcmc_parameters['b^{('+i_s+')}_{G_2}']['current'] *
             data.mcmc_parameters['b^{('+i_s+')}_{G_2}']['scale'])

#        c1 = (data.mcmc_parameters['c1']['current'] *
#                 data.mcmc_parameters['c1']['scale'])

	c1 = 0.
	fNL = 0.
        Pshot = (data.mcmc_parameters['P^{('+i_s+')}_{shot}']['current'] *
                 data.mcmc_parameters['P^{('+i_s+')}_{shot}']['scale'])
        Bshot = (data.mcmc_parameters['B^{('+i_s+')}_{shot}']['current'] *
                 data.mcmc_parameters['B^{('+i_s+')}_{shot}']['scale'])

        #fNL = (data.mcmc_parameters['f_{NL}']['current'] *
        #         data.mcmc_parameters['f_{NL}']['scale'])

##        bGamma3 = (data.mcmc_parameters['b_{Gamma_3}']['current'] *
##                 data.mcmc_parameters['b_{Gamma_3}']['scale'])
#        css0 = (data.mcmc_parameters['c^2_{0}']['current'] *
#                 data.mcmc_parameters['c^2_{0}']['scale'])
#        css2 = (data.mcmc_parameters['c^2_{2}']['current'] *
#                 data.mcmc_parameters['c^2_{2}']['scale'])
#        b4 = (data.mcmc_parameters['b_4']['current'] *
#                 data.mcmc_parameters['b_4']['scale'])
#        css4 = (data.mcmc_parameters['c^2_{4}']['current'] *
#                 data.mcmc_parameters['c^2_{4}']['scale'])

#        a0 = (data.mcmc_parameters['a_0']['current'] *
#                 data.mcmc_parameters['a_0']['scale'])

        dk2 = 0.005;
        kmin = 0.00163625
        kint = np.linspace(log(1.e-4),log(0.41),100)
        kint = np.exp(kint)
        krange_2 = len(kint)
        P0inttab = np.zeros(krange_2)
        P2inttab = np.zeros(krange_2)
        P4inttab = np.zeros(krange_2)

###### mean values ######
        bGamma3 = 0.57
        #Pshot = 0.
        a0 = 0.
        a2 = 0.
        css4 = 0.
        css2 = 30.
        css0 = 0.
        b4 = 500.*0.
#### standard deviations ######
        psh = 3e3;
        sigbGamma3 = 1.
        sigPshot = 0.*0.3*psh
        siga0 = psh*1.
        sigcs0 = 30.
        sigcs2 = 30.
        sigcs4 = 30.
        sigb4 = 500.
        siga2 = psh*2.


        z = self.z;
        fz = cosmo.scale_independent_growth_factor_f(z)

        # Run CLASS-PT
        all_theory = cosmo.get_pk_mult(kint*h,self.z,krange_2)

        P0inttab = (norm**2.*all_theory[15] +norm**4.*(all_theory[21])+ norm**1.*b1*all_theory[16] +norm**3.*b1*(all_theory[22]) + norm**0.*b1**2.*all_theory[17] +norm**2.*b1**2.*all_theory[23] + 0.25*norm**2.*b2**2.*all_theory[1] +b1*b2*norm**2.*all_theory[30]+ b2*norm**3.*all_theory[31] + b1*bG2*norm**2.*all_theory[32]+ bG2*norm**3.*all_theory[3] + b2*bG2*norm**2.*all_theory[4]+ bG2**2.*norm**2.*all_theory[5] + 2.*css0*norm**2.*all_theory[11]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**2.*(b1*all_theory[7]+norm*all_theory[8]))*h**3. + (psh)*Pshot + a0*(10**4)*(kint/0.5)**2.  + fz**2.*b4*kint**2.*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)*all_theory[13]*h + a2*(1./3.)*(10.**4.)*(kint/0.45)**2.
        P2inttab = (norm**2.*all_theory[18] +  norm**4.*(all_theory[24])+ norm**1.*b1*all_theory[19] +norm**3.*b1*(all_theory[25]) + b1**2.*norm**2.*all_theory[26] +b1*b2*norm**2.*all_theory[34]+ b2*norm**3.*all_theory[35] + b1*bG2*norm**2.*all_theory[36]+ bG2*norm**3.*all_theory[37]  + 2.*css2*norm**2.*all_theory[12]/h**2. + (2.*bG2+0.8*bGamma3*norm)*norm**3.*all_theory[9])*h**3. + fz**2.*b4*kint**2.*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)*all_theory[13]*h + a2*(10.**4.)*(2./3.)*(kint/0.45)**2.
        P4inttab = (norm**2.*all_theory[20] + norm**4.*all_theory[27]+ b1*norm**3.*all_theory[28] + b1**2.*norm**2.*all_theory[29] + b2*norm**3.*all_theory[38] + bG2*norm**3.*all_theory[39]  +2.*css4*norm**2.*all_theory[13]/h**2.)*h**3. + fz**2.*b4*kint**2.*(norm**2.*fz**2.*210./143. + 30.*fz*b1*norm/11.+b1**2.)*all_theory[13]*h

        E0bG3 = (0.8*sigbGamma3*norm)*norm**2.*(b1*all_theory[7]+norm*all_theory[8])*h**3.
        E2bG3 = (0.8*sigbGamma3*norm)*norm**3.*all_theory[9]*h**3.
        Ecs4 = 2.*norm**2.*all_theory[13]*h**1.
        Ecs2 = 2.*norm**2.*all_theory[12]*h**1.
        Ecs0 = 2.*norm**2.*all_theory[11]*h**1.
        Eb4 = fz**2.*kint**2.*all_theory[13]*h


        P0int = interpolate.InterpolatedUnivariateSpline(kint,P0inttab,ext=3)
        P2int = interpolate.InterpolatedUnivariateSpline(kint,P2inttab,ext=3)
        P4int = interpolate.InterpolatedUnivariateSpline(kint,P4inttab,ext=3)
        E0bG3int = interpolate.InterpolatedUnivariateSpline(kint,E0bG3,ext=3)
        E2bG3int = interpolate.InterpolatedUnivariateSpline(kint,E2bG3,ext=3)
        Ecs4int = interpolate.InterpolatedUnivariateSpline(kint,Ecs4,ext=3)
        Ecs2int = interpolate.InterpolatedUnivariateSpline(kint,Ecs2,ext=3)
        Ecs0int = interpolate.InterpolatedUnivariateSpline(kint,Ecs0,ext=3)
        Eb4int = interpolate.InterpolatedUnivariateSpline(kint,Eb4,ext=3)

        integr0bG3 = lambda k: exp(3.*k)*E0bG3int(exp(k))
        integr2bG3 = lambda k: exp(3.*k)*E2bG3int(exp(k))
        integrand0 = lambda k: exp(3.*k)*P0int(exp(k))
        integrand2 = lambda k: exp(3.*k)*P2int(exp(k))
        integrand4 = lambda k: exp(3.*k)*P4int(exp(k))
        integrandEcs4 = lambda k: exp(3.*k)*Ecs4int(exp(k))
        integrandEcs2 = lambda k: exp(3.*k)*Ecs2int(exp(k))
        integrandEcs0 = lambda k: exp(3.*k)*Ecs0int(exp(k))
        integrandEb4 = lambda k: exp(3.*k)*Eb4int(exp(k))

        x1 = np.zeros(3*self.ksize2+self.ntriag)

        P0th = np.zeros(self.ksize2)
        P2th = np.zeros(self.ksize2)
        P4th = np.zeros(self.ksize2)

        E0bG3th = np.zeros(self.ksize2)
        E2bG3th = np.zeros(self.ksize2)
        Ecs4th = np.zeros(self.ksize2)
        Ecs2th = np.zeros(self.ksize2)
        Ecs0th = np.zeros(self.ksize2)

        EbG3cov = np.zeros(3*self.ksize2+self.ntriag)
        Pshotcov = np.zeros(3*self.ksize2+self.ntriag)
        a0cov = np.zeros(3*self.ksize2+self.ntriag)
        a2cov = np.zeros(3*self.ksize2+self.ntriag)
        Ecs4cov = np.zeros(3*self.ksize2+self.ntriag)
        Ecs2cov = np.zeros(3*self.ksize2+self.ntriag)
        Ecs0cov = np.zeros(3*self.ksize2+self.ntriag)

        Eb4cov = np.zeros(3*self.ksize2+self.ntriag)
        Eb4th = np.zeros(self.ksize2)



        for i in range(self.ksize2):
            P0th[i] = integrate.quad(integrand0, log(dk2*i+self.kmin2), log(dk2*(i+1)+self.kmin2))[0]*3./((dk2*(i+1)+self.kmin2)**3.-(dk2*i+self.kmin2)**3.)
            P2th[i] = integrate.quad(integrand2, log(dk2*i+self.kmin2), log(dk2*(i+1)+self.kmin2))[0]*3./((dk2*(i+1)+self.kmin2)**3.-(dk2*i+self.kmin2)**3.)
            P4th[i] = integrate.quad(integrand4, log(dk2*i+self.kmin2), log(dk2*(i+1)+self.kmin2))[0]*3./((dk2*(i+1)+self.kmin2)**3.-(dk2*i+self.kmin2)**3.)

            Ecs0th[i] = integrate.quad(integrandEcs0, log(dk2*i+self.kmin2), log(dk2*(i+1)+self.kmin2))[0]*3./((dk2*(i+1)+self.kmin2)**3.-(dk2*i+self.kmin2)**3.)
            Ecs2th[i] = integrate.quad(integrandEcs2, log(dk2*i+self.kmin2), log(dk2*(i+1)+self.kmin2))[0]*3./((dk2*(i+1)+self.kmin2)**3.-(dk2*i+self.kmin2)**3.)
            Ecs4th[i] = integrate.quad(integrandEcs4, log(dk2*i+self.kmin2), log(dk2*(i+1)+self.kmin2))[0]*3./((dk2*(i+1)+self.kmin2)**3.-(dk2*i+self.kmin2)**3.)
            E0bG3th[i] = integrate.quad(integr0bG3, log(dk2*i+self.kmin2), log(dk2*(i+1)+self.kmin2))[0]*3./((dk2*(i+1)+self.kmin2)**3.-(dk2*i+self.kmin2)**3.)
            E2bG3th[i] = integrate.quad(integr2bG3, log(dk2*i+self.kmin2), log(dk2*(i+1)+self.kmin2))[0]*3./((dk2*(i+1)+self.kmin2)**3.-(dk2*i+self.kmin2)**3.)
            Eb4th[i] = integrate.quad(integrandEb4, log(dk2*i+self.kmin2), log(dk2*(i+1)+self.kmin2))[0]*3./((dk2*(i+1)+self.kmin2)**3.-(dk2*i+self.kmin2)**3.)

        for i in range(self.ksize2):
            x1[i] = P0th[i] - self.Pk0[i]
            x1[i + self.ksize2] = P2th[i] - self.Pk2[i]
            x1[i + 2*self.ksize2] = P4th[i] - self.Pk4[i]
            Pshotcov[i] = 1.
            a0cov[i] = (self.k2[i]/0.45)**2.
            a2cov[i] = (1./3.)*(self.k2[i]/0.45)**2.
            a2cov[i+self.ksize2] = (2./3.)*(self.k2[i]/0.45)**2.;
            Ecs4cov[i+2*self.ksize2] = Ecs4th[i]
            Ecs2cov[i+self.ksize2] = Ecs2th[i]
            Ecs0cov[i] = Ecs0th[i]
            EbG3cov[i] = E0bG3th[i]
            EbG3cov[i+self.ksize2] = E2bG3th[i]

            Eb4cov[i] = Eb4th[i]*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)
            Eb4cov[i + self.ksize2] = Eb4th[i]*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)   
            Eb4cov[i + 2*self.ksize2] = Eb4th[i]*(norm**2.*fz**2.*210./143. + 30.*fz*b1*norm/11.+b1**2.)


### bispectrum part starts

        Ashot = 0.
        c0 = 0.
        c2 = 0.
        SigmaB = 0.
        beta = fz/b1
        #all_theory_2 = cosmo.get_pk_mult(self.k*h,self.z,self.ksize)
        a0 = 1. + 2.*beta/3. + beta**2./5.
        Plintab = np.zeros(krange_2)
        Plintab = -1.*norm**2.*(all_theory[10]/h**2./kint**2)*h**3
        P2 = norm**2.*(all_theory[14])*h**3.

        delta = np.zeros(self.ntriag)
        Pbar = psh
        ng = (1.+Ashot)/Pbar

        rbao = cosmo.rs_drag()*h
        P0int = interpolate.InterpolatedUnivariateSpline(kint,Plintab,ext=3)
        Sigma = integrate.quad(lambda k: (4*np.pi)*exp(1.*k)*P0int(exp(k))*(1.-3*(2*rbao*exp(k)*cos(exp(k)*rbao)+(-2+rbao**2*exp(k)**2)*sin(rbao*exp(k)))/(exp(k)*rbao)**3)/(3*(2*np.pi)**3.), log(2.e-4), log(0.2))[0]
        deltaSigma = integrate.quad(lambda k: (4*np.pi)*exp(1.*k)*P0int(exp(k))*(self.j2(exp(k)*rbao))/((2*np.pi)**3.), log(2.e-4), log(0.2))[0]
        Pw = (Plintab-P2)/(np.exp(-kint**2.*Sigma)-np.exp(-kint**2.*Sigma)*(1+kint**2.*Sigma));
        Pnw = Plintab - Pw*np.exp(-kint**2.*Sigma)

        Pwfunc = interpolate.InterpolatedUnivariateSpline(kint,Pw,ext=3)
        Pnwfunc = interpolate.InterpolatedUnivariateSpline(kint,Pnw,ext=3)

        ks = 0.05
        Sigma = integrate.quad(lambda k: (4*np.pi)*exp(1.*k)*P0int(exp(k))*(1.-3*(2*rbao*exp(k)*cos(exp(k)*rbao)+(-2+rbao**2*exp(k)**2)*sin(rbao*exp(k)))/(exp(k)*rbao)**3)/(3*(2*np.pi)**3.), log(2.e-4), log(ks))[0]
        deltaSigma = integrate.quad(lambda k: (4*np.pi)*exp(1.*k)*P0int(exp(k))*(self.j2(exp(k)*rbao))/((2*np.pi)**3.), log(2.e-4), log(ks))[0]

        #Pres = lambda k, mu: P0int(k)
        Pres = lambda k, mu: Pnwfunc(k) +  np.exp(-k**2.*(Sigma*(1.+2.*fz*mu**2.*(2.+fz)) + deltaSigma*mu**2.*fz**2.*(mu**2.-1.)))*Pwfunc(k) -(c0+c1*mu**2.+c2*mu**4.)*(k/0.3)**2.*P0int(k)/(b1+fz*mu**2.)

        da=cosmo.angular_distance(self.z)/(self.hfid/h)
        hz=cosmo.Hubble(self.z)*(self.hfid/h)/self.kmsMpc


        apar=self.hzfid/hz
        aperp=da/self.dafid
        B0th = np.zeros(self.ntriag)
        new_triag = self.new_triag;


        Azeta = cosmo.A_s()*2.*np.pi**2.

        Tfunc = lambda k: (P0int(k)/(Azeta*((k/0.05)**cosmo.n_s())/k**3.))**0.5
        BNG = lambda k1, k2, k3: Azeta**2.*(Tfunc(k1)*Tfunc(k2)*Tfunc(k3)*(18./5.)*(-1./k1**3./k2**3.
                        -1./k3**3./k2**3.-1./k1**3./k3**3.-2./k1**2./k2**2./k3**2.
                        +1/k1/k2**2./k3**3.+1/k1/k3**2./k2**3.+1/k2/k3**2./k1**3.
                        +1/k2/k1**2./k3**3.+1/k3/k1**2./k2**3.+1/k3/k2**2./k1**3.))


        for j in range(int(self.ntriag)):
                kc1 = self.k[new_triag[0][j]]
                kc2 = self.k[new_triag[1][j]]
                kc3 = self.k[new_triag[2][j]]
                dk1 = self.dk
                dk2 = self.dk
                dk3 = self.dk
                if (self.k[new_triag[0][j]]<self.dk):
                        kc1 = 0.0058
                        dk1  = 0.0084
                if (self.k[new_triag[1][j]]<self.dk):
                        kc2 = 0.0058
                        dk2  = 0.0084
                if (self.k[new_triag[2][j]]<self.dk):
                        kc3 = 0.0058
                        dk3  = 0.0084

                xfunc = lambda k1, k2, k3: ((kc3+k3*dk3/2.)**2.-(kc1+k1*dk1/2.)**2. - (kc2+k2*dk2/2.)**2.)/(2.*(kc1+k1*dk1/2.)*(kc2+k2*dk2/2.));
                yfunc = lambda k1, k2, k3: np.sqrt(np.abs(1.-xfunc(k1,k2,k3)**2.))
                nu1 = lambda mu: mu/apar/(np.sqrt(np.abs(mu**2./apar**2. + (1-mu**2.)/aperp**2.)));

                mu2 = lambda k1, k2, k3, mu, phi: xfunc(k1,k2,k3)*mu - np.sqrt(1.-mu**2.)*yfunc(k1,k2,k3)*np.cos(phi*2.*np.pi)
                mu3 = lambda k1, k2, k3, mu, phi: -((kc2+k2*dk2/2.)/(kc3+k3*dk3/2.))*mu2(k1,k2,k3,mu,phi)-((kc1+k1*dk1/2.)/(kc3+k3*dk3/2.))*mu;

                nu2 = lambda k1, k2, k3, mu, phi: mu2(k1, k2, k3, mu, phi)/apar/(np.sqrt(np.abs(mu2(k1, k2, k3, mu, phi)**2./apar**2. + (1-mu2(k1, k2, k3, mu, phi)**2.)/aperp**2.)));
                nu3 = lambda k1, k2, k3, mu, phi: mu3(k1, k2, k3, mu, phi)/apar/(np.sqrt(np.abs(mu3(k1, k2, k3, mu, phi)**2./apar**2. + (1-mu3(k1, k2, k3, mu, phi)**2.)/aperp**2.)));

                q1 = lambda mu: np.sqrt(np.abs(mu**2/apar**2 + (1-mu**2)/aperp**2));
                q2 = lambda k1, k2, k3, mu, phi: np.sqrt(np.abs(mu2(k1, k2, k3, mu, phi)**2/apar**2 + (1-mu2(k1, k2, k3, mu, phi)**2)/aperp**2))
                q3 = lambda k1, k2, k3, mu, phi: np.sqrt(np.abs(mu3(k1, k2, k3, mu, phi)**2/apar**2 + (1-mu3(k1, k2, k3, mu, phi)**2)/aperp**2))

                z21 = lambda k1,k2,k3,mu, phi: self.F2real((kc1+k1*dk1/2.)*q1(mu),(kc2+k2*dk2/2.)*q2(k1, k2, k3, mu, phi),(kc3+k3*dk3/2.)*q3(k1, k2, k3, mu, phi),b1,b2,bG2)+b1**3.*beta*((nu2(k1,k2,k3,mu,phi)*(kc2+k2*dk2/2.)*q2(k1, k2, k3, mu, phi)+nu1(mu)*(kc1+k1*dk1/2.)*q1(mu))/(kc3+k3*dk3/2.)/q3(k1, k2, k3, mu, phi))**2.*self.G2((kc1+k1*dk1/2.)*q1(mu),(kc2+k2*dk2/2.)*q2(k1, k2, k3, mu, phi),(kc3+k3*dk3/2.)*q3(k1, k2, k3, mu, phi))+(b1**4.*beta/2.)*(nu2(k1,k2,k3,mu,phi)*(kc2+k2*dk2/2.)*q2(k1, k2, k3, mu, phi)+nu1(mu)*(kc1+k1*dk1/2.)*q1(mu))*(nu1(mu)*(1.+beta*nu2(k1, k2, k3,mu,phi)**2.)/(kc1+k1*dk1/2.)/q1(mu) + nu2(k1,k2,k3,mu,phi)*(1.+beta*nu1(mu)**2.)/(kc2+k2*dk2/2.)/q2(k1, k2, k3, mu, phi))
                z22 = lambda k1,k2,k3,mu, phi: self.F2real((kc1+k1*dk1/2.)*q1(mu),(kc3+k3*dk3/2.)*q3(k1, k2, k3, mu, phi),(kc2+k2*dk2/2.)*q2(k1, k2, k3, mu, phi),b1,b2,bG2)+b1**3.*beta*((nu3(k1,k2,k3,mu,phi)*(kc3+k3*dk3/2.)*q3(k1, k2, k3, mu, phi)+nu1(mu)*(kc1+k1*dk1/2.)*q1(mu))/(kc2+k2*dk2/2.)/q2(k1, k2, k3, mu, phi))**2.*self.G2((kc1+k1*dk1/2.)*q1(mu),(kc3+k3*dk3/2.)*q3(k1, k2, k3, mu, phi),(kc2+k2*dk2/2.)*q2(k1, k2, k3, mu, phi))+(b1**4.*beta/2.)*(nu3(k1,k2,k3,mu,phi)*(kc3+k3*dk3/2.)*q3(k1, k2, k3, mu, phi)+nu1(mu)*(kc1+k1*dk1/2.)*q1(mu))*(nu1(mu)*(1.+beta*nu3(k1, k2, k3,mu,phi)**2.)/(kc1+k1*dk1/2.)/q1(mu) + nu3(k1,k2,k3,mu,phi)*(1.+beta*nu1(mu)**2.)/(kc3+k3*dk3/2.)/q3(k1, k2, k3, mu, phi))
                z23 = lambda k1,k2,k3,mu, phi: self.F2real((kc2+k2*dk2/2.)*q2(k1, k2, k3, mu, phi),(kc3+k3*dk3/2.)*q3(k1, k2, k3, mu, phi),(kc1+k1*dk1/2.)*q1(mu),b1,b2,bG2)+b1**3.*beta*((nu2(k1,k2,k3,mu,phi)*(kc2+k2*dk2/2.)*q2(k1, k2, k3, mu, phi)+nu3(k1,k2,k3,mu,phi)*(kc3+k3*dk3/2.)*q3(k1, k2, k3, mu, phi))/(kc1+k1*dk1/2.)/q1(mu))**2.*self.G2((kc2+k2*dk2/2.)*q2(k1, k2, k3, mu, phi),(kc3+k3*dk3/2.)*q3(k1, k2, k3, mu, phi),(kc1+k1*dk1/2.)*q1(mu))+(b1**4.*beta/2.)*(nu2(k1,k2,k3,mu,phi)*(kc2+k2*dk2/2.)*q2(k1, k2, k3, mu, phi)+nu3(k1,k2,k3,mu,phi)*(kc3+k3*dk3/2.)*q3(k1, k2, k3, mu, phi))*(nu2(k1,k2,k3,mu,phi)*(1.+beta*nu3(k1, k2, k3,mu,phi)**2.)/(kc2+k2*dk2/2.)/q2(k1, k2, k3, mu, phi) + nu3(k1,k2,k3,mu,phi)*(1.+beta*nu2(k1,k2,k3,mu,phi)**2.)/(kc3+k3*dk3/2.)/q3(k1, k2, k3, mu, phi))


                Fnlfunc = lambda k1, k2, k3, mu, phi: fNL*BNG((kc1+k1*dk1/2.)*q1(mu),(kc2+k2*dk2/2.)*q2(k1, k2, k3, mu, phi),(kc3+k3*dk3/2.)*q3(k1, k2, k3, mu, phi))*b1**3.*(1+beta*nu1(mu)**2)*(1. + beta*(nu3(k1, k2, k3, mu, phi))**2.)*(1+beta*(nu2(k1, k2, k3, mu, phi))**2)*(kc1+k1*dk1/2.)*(kc2+k2*dk2/2.)*(kc3+k3*dk3/2.)*(dk1/2.)*(dk2/2.)*(dk3/2.)


                F2func1 = lambda k1,k2,k3,mu, phi:  z21(k1,k2,k3,mu,phi)*(1+beta*nu1(mu)**2)*(1. + beta*(nu2(k1, k2, k3, mu, phi))**2.)*Pres((kc1+k1*dk1/2.)*q1(mu),nu1(mu))*(kc1+k1*dk1/2.)*(dk1/2.)*Pres((kc2+k2*dk2/2.)*q2(k1, k2, k3, mu, phi),nu2(k1,k2,k3,mu,phi))*(kc2+k2*dk2/2.)*(dk2/2.)*(kc3+k3*dk3/2.)*(dk3/2.)+1.*0.5*(Bshot/ng)*b1**2.*Pres((kc1+k1*dk1/2.)*q1(mu),nu1(mu))*(kc1+k1*dk1/2.)*(1.+beta*nu1(mu)**2.*(Bshot+2.*(1.+Pshot))/Bshot + beta**2.*nu1(mu)**4.*2.*(1.+Pshot)/Bshot)*(kc2+k2*dk2/2.)*(kc3+k3*dk3/2.)*(dk1/2.)*(dk2/2.)*(dk3/2.) + ((1.+Pshot)/ng)**2.*(kc1+k1*dk1/2.)*(kc2+k2*dk2/2.)*(kc3+k3*dk3/2.)*(dk1/2.)*(dk2/2.)*(dk3/2.)/2.
                F2func2 = lambda k1,k2,k3,mu, phi:  z22(k1,k2,k3,mu,phi)*(1+beta*nu1(mu)**2)*(1. + beta*(nu3(k1, k2, k3, mu, phi))**2.)*Pres((kc1+k1*dk1/2.)*q1(mu),nu1(mu))*(kc1+k1*dk1/2.)*(dk1/2.)*Pres((kc3+k3*dk3/2.)*q3(k1, k2, k3, mu, phi),nu3(k1,k2,k3,mu,phi))*(kc3+k3*dk3/2.)*(dk3/2.)*(kc2+k2*dk2/2.)*(dk2/2.)+1.*0.5*(Bshot/ng)*b1**2.*Pres((kc2+k2*dk2/2.)*q2(k1, k2, k3, mu, phi),nu2(k1,k2,k3,mu,phi))*(kc2+k2*dk2/2.)*((1.+beta*nu2(k1,k2,k3,mu,phi)**2.*(Bshot+2.*(1.+Pshot))/Bshot + beta**2.*nu2(k1,k2,k3,mu,phi)**4.*2.*(1.+Pshot)/Bshot))*(kc1+k1*dk1/2.)*(kc3+k3*dk3/2.)*(dk1/2.)*(dk2/2.)*(dk3/2.) + 0.*(1/ng)**2.*(kc1+k1*dk1/2.)*(kc2+k2*dk2/2.)*(kc3+k3*dk3/2.)*(dk1/2.)*(dk2/2.)*(dk3/2.)/6.
                F2func3 = lambda k1,k2,k3,mu, phi:  z23(k1,k2,k3,mu,phi)*(1+beta*(nu2(k1, k2, k3, mu, phi))**2)*(1. + beta*(nu3(k1, k2, k3, mu, phi))**2.)*Pres((kc2+k2*dk2/2.)*q2(k1, k2, k3, mu, phi),nu2(k1,k2,k3,mu,phi))*(kc2+k2*dk2/2.)*(dk2/2.)*Pres((kc3+k3*dk3/2.)*q3(k1, k2, k3, mu, phi),nu3(k1,k2,k3,mu,phi))*(kc3+k3*dk3/2.)*(dk3/2.)*(kc1+k1*dk1/2.)*(dk1/2.) + 1.*0.5*(Bshot/ng)*b1**2.*Pres((kc3+k3*dk3/2.)*q3(k1, k2, k3, mu, phi),nu3(k1,k2,k3,mu,phi))*(kc3+k3*dk3/2.)*(1.+beta*nu3(k1,k2,k3,mu,phi)**2.*(Bshot + 2.*(1.+Pshot))/Bshot + beta**2.*nu3(k1,k2,k3,mu,phi)**4.*2.*(1.+Pshot)/Bshot)*(kc2+k2*dk2/2.)*(kc1+k1*dk1/2.)*(dk1/2.)*(dk2/2.)*(dk3/2.) + 0.*(1/ng)**2.*(kc1+k1*dk1/2.)*(kc2+k2*dk2/2.)*(kc3+k3*dk3/2.)*(dk1/2.)*(dk2/2.)*(dk3/2.)/6.

                Bfunc3 = lambda k1,k2,k3,mu, phi: ((2.*F2func1(k1,k2,k3,mu, phi) + 2.*F2func2(k1,k2,k3,mu, phi) + 2.*F2func3(k1,k2,k3,mu, phi)+Fnlfunc(k1,k2,k3,mu,phi))/apar**2./aperp**4.)

                Nk1 = ((kc1+dk1/2.)**2. - (kc1-dk1/2.)**2.)/2.
                Nk2 = ((kc2+dk2/2.)**2. - (kc2-dk2/2.)**2.)/2.
                Nk3 = ((kc3+dk3/2.)**2. - (kc3-dk3/2.)**2.)/2.

                mat4 = Bfunc3(*np.meshgrid(self.gauss_mu,self.gauss_mu,self.gauss_mu,self.gauss_mu2,self.gauss_mu2, sparse=False, indexing='ij'))
                B0th[j] = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(mat4,self.gauss_w2)/2.,self.gauss_w2)/2.,self.gauss_w),self.gauss_w),self.gauss_w)/Nk1/Nk2/Nk3

                delta[j] = B0th[j]*self.weights[j] - self.Bk[j]
		x1[3*self.ksize2 + j] = delta[j]

	chi2 =0.
        marg_cov = self.cov + np.outer(EbG3cov,EbG3cov) + sigPshot**2.*np.outer(Pshotcov,Pshotcov) + siga0**2.*np.outer(a0cov,a0cov) + siga2**2.*np.outer(a2cov,a2cov) + sigcs4**2.*np.outer(Ecs4cov,Ecs4cov)+sigcs2**2.*np.outer(Ecs2cov,Ecs2cov)+sigcs0**2.*np.outer(Ecs0cov,Ecs0cov) + sigb4**2.*np.outer(Eb4cov,Eb4cov)
        chi2 = np.inner(x1,np.inner(np.linalg.inv(marg_cov),x1));
        chi2 +=np.linalg.slogdet(marg_cov)[1] - self.logdetcov
	chi2 +=(Pshot)**2. + 1.*(Bshot-1.)**2.+1.*(c1)**2./5.**2.+ (b2 - 0.)**2./1**2. + (bG2 - 0.)**2/1**2.
        #chi2 += (b2/norm - 0.)**2./1.**2. + (bG2/norm - 0.)**2./1.**2.
        loglkl = -0.5 * chi2
	#print('chi2 PxB=',chi2)
#	chi2=np.dot(x1,np.dot(self.invcov,x1))+(Pshot)**2/0.3**2

        return loglkl
