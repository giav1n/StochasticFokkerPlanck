"""
Gianni Valerio Vinci

Stochastic Fokker-Planck integration of LIF neural population
The Following code is a riadaptation of the one produced by Augstin et all
"Cite..."
For theory and exaples:
"Cite... "
"""
import numpy as np
from numba import prange
import scipy.sparse
import scipy.sparse.linalg
from math import log, sqrt, exp
from scipy.linalg import solve_banded
import deepdish.io as dd
from scipy.interpolate import RectBivariateSpline as RB
Para=False

# Usefull functions

def moving_average(a,t, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    MA=ret[n - 1:] / n
    tMA=np.linspace(t[0],t[-1],len(MA))
    return MA,tMA



# Get self-consistent external current:
def ExternalCurrent(P):
    MuExt=float((P['Mu0']/P['TauV'])- P['K']*P['J']*P['R0']-P['IExt'] +P['gC']*P['AlphaC']*P['R0']*P['TauC'])
    SigESquare=(P['Sig0']**2/P['TauV'])-P['K']*(P['J']**2)*P['R0']
    SigExt=float(np.sqrt(SigESquare))
    return MuExt,SigExt



# Finite size noise generator

def LoadFuncOU(N):
    """
    Interpolate the function of the O.U. process that generates finite-size
    noise Eta:

    Input: N Size of the pool (Only option are 10^3,10^4,10^5)
    Output: functions a(Mu,Sig),a1(Mu,Sig),a2(Mu,Sig),b(Mu,Sig)
    """

    M=dd.load('./Coefficients/OU'+str(int(N))+'.h5')
    MuArr=M['Mu']
    SigArr=M['Sig']
    aM=M['a']
    a1M=M['a1']
    a2M=M['a2']
    bM=M['b']
    aT=RB(MuArr,SigArr,aM,kx=1,ky=1)
    fha=lambda Mu,Sig: aT(Mu,Sig)[0][0]
    a1T=RB(MuArr,SigArr,a1M,kx=1,ky=1)
    fha1=lambda Mu,Sig: a1T(Mu,Sig)[0][0]
    a2T=RB(MuArr,SigArr,a2M,kx=1,ky=1)
    fha2=lambda Mu,Sig: a2T(Mu,Sig)[0][0]
    bT=RB(MuArr,SigArr,bM,kx=1,ky=1)
    fhb=lambda Mu,Sig: bT(Mu,Sig)[0][0]
    return fha,fha1,fha2,fhb

def InitializeEta(N):
    """
    Handle function, returns the integration routine for Eta
    Input: N size of the pool
    """
    fha,fha1,fha2,fhb=LoadFuncOU(N)
    def IntegrateEta(dt,Z,Eta,u1,u2,Nu0,N,Mu,Sig):
        a,b,a1,a2=fha(Mu,Sig),fhb(Mu,Sig),fha1(Mu,Sig),fha2(Mu,Sig)
        Beta=np.sqrt(Nu0/N)
        u1n=u1 +dt*(-a*u1-a1*u2) + b*np.sqrt(dt)*Z
        u2n=u2 +dt*(-a2*u1-a*u2)
        Eta=Beta*Z/np.sqrt(dt) +u1n+u2n
        return Eta,u1n,u2n
    return IntegrateEta

### Functions for FP integration:
class Grid(object):
    '''
    This class implements the V discretization, which is used for both the
    Scharfetter-Gummel method and the upwind implementation
    '''

    def __init__(self, V_0=-200., V_1=-40., V_r=-70., N_V=100):
        self.V_0 = V_0
        self.V_1 = V_1
        self.V_r = V_r
        self.N_V = int(N_V)
        # construct the grid object
        self.construct()

    def construct(self):
        self.V_centers = np.linspace(self.V_0, self.V_1, self.N_V)
        # shift V_centers by half of the grid spacing to the left
        # such that the last interface lies exactly on V_l
        self.V_centers -= (self.V_centers[-1] - self.V_centers[-2]) / 2.
        self.dV_centers = np.diff(self.V_centers)
        self.V_interfaces = np.zeros(self.N_V + 1)
        self.V_interfaces[1:-1] = self.V_centers[:-1] + 0.5 * self.dV_centers
        self.V_interfaces[0] = self.V_centers[0] - 0.5 * self.dV_centers[0]
        self.V_interfaces[-1] = self.V_centers[-1] + 0.5 * self.dV_centers[-1]
        self.dV_interfaces = np.diff(self.V_interfaces)
        self.dV = self.V_interfaces[2] - self.V_interfaces[1]
        self.ib = np.argmin(np.abs(self.V_centers - self.V_r))

# Drift term
try:
    from numba import njit
except:
    def njit(func):
        return func

####

@njit(parallel=False)
def exp_vdV_D(v,dV,D):
    return exp(-v*dV/D)


@njit(parallel=Para)
def get_v_numba(L, Vi, EL, taum, mu):
    # LIF model
    drift = np.empty(L)
    for i in prange(L):
        drift[i] = (EL - Vi[i]) / taum + mu
    return drift        

def get_v(grid, mu, params):
    '''returns the coefficients for the drift part of the flux for LIF neuron '''
    Vi = grid.V_interfaces
    EL = params['Vr']
    taum = params['TauV']
    drift = (EL - Vi) / taum + mu
    return drift


@njit(parallel=Para)
def matAdt_opt(N,v,D,dV,dt):
    mat = np.zeros((3,N))
    dt_dV = dt/dV

    for i in prange(1,N-1):
        if v[i] != 0.0:
            exp_vdV_D1 = exp_vdV_D(v[i],dV,D)
            mat[1,i] = -dt_dV*v[i]*exp_vdV_D1/(1.-exp_vdV_D1) # diagonal
            mat[2,i-1] = dt_dV*v[i]/(1.-exp_vdV_D1) # lower diagonal
        else:
            mat[1,i] = -dt_dV*D/dV # diagonal
            mat[2,i-1] = dt_dV*D/dV # lower diagonal
        if v[i+1] != 0.0:
            exp_vdV_D2 = exp_vdV_D(v[i+1],dV,D)
            mat[1,i] -= dt_dV*v[i+1]/(1.-exp_vdV_D2) # diagonal
            mat[0,i+1] = dt_dV*v[i+1]*exp_vdV_D2/(1.-exp_vdV_D2) # upper diagonal
        else:
            mat[1,i] -= dt_dV*D/dV # diagonal
            mat[0,i+1] = dt_dV*D/dV # upper diagonal

    # boundary conditions
    if v[1] != 0.0:
        tmp1 = v[1]/(1.-exp_vdV_D(v[1],dV,D))
    else:
        tmp1 = D/dV
    if v[-1] != 0.0:
        tmp2 = v[-1]/(1.-exp_vdV_D(v[-1],dV,D))
    else:
        tmp2 = D/dV
    if v[-2] != 0.0:
        tmp3 = v[-2]/(1.-exp_vdV_D(v[-2],dV,D))
    else:
        tmp3 = D/dV

    mat[1,0] = -dt_dV*tmp1  # first diagonal
    mat[0,1] = dt_dV*tmp1*exp_vdV_D(v[1],dV,D)  # first upper
    mat[2,-2] = dt_dV*tmp3  # last lower
    mat[1,-1] = -dt_dV * ( tmp3*exp_vdV_D(v[-2],dV,D)
                          +tmp2*(1.+exp_vdV_D(v[-1],dV,D)) )  # last diagonal
    return mat

####
def initial_p_distribution(grid,params):
    """
    Given grid and parmas to set the wantend initial condition
    """
    if params['fp_v_init'] == 'delta':
        delta_peak_index = np.argmin(np.abs(grid.V_centers - params['Vr']))
        p_init = np.zeros_like(grid.V_centers)
        p_init[delta_peak_index] = 1.
    elif params['fp_v_init'] == 'uniform':
        # uniform dist on [Vr, Vcut]
        p_init = np.zeros_like(grid.V_centers)
        p_init[grid.ib:] = 1.
    elif params['fp_v_init'] == 'normal':
        mean_gauss = params['fp_normal_mean']
        sigma_gauss = params['fp_normal_sigma']
        p_init = np.exp(-np.power((grid.V_centers - mean_gauss), 2) / (2 * sigma_gauss ** 2))
    elif params['fp_v_init'] == 'P0':
        # uniform dist on [Vr, Vcut]
        p_init = params['P0']
        #p_init[grid.ib:] = 1.


    else:
        err_mes = ('Initial condition "{}" is not implemented! See params dict for options.').format(params['fp_v_init'])
        raise NotImplementedError(err_mes)
    # normalization with respect to the cell widths
    p_init =p_init/np.sum(p_init*grid.dV_interfaces)
    return p_init


# FokkerPlanck Integration

def IntegrateFP(inPar,grid):

    '''
    Solve the fp equation using the scharfetter gummel method 
       for a time step dt

    '''
    dt=inPar['dt']
    EL=grid.V_r
    taum=inPar['TauV']
    taud=inPar['TauD']
    Eta=inPar['Eta']
    # external input
    #mu_ext = inPar['MuExt']
    #sigma_ext = inPar['SigExt']
    
    # mu_syn (synaptic) combines the external input with recurrent input
    #mu_syn = K * J * r_rec + mu_ext
    #sigma_tot = sqrt(K * J ** 2 * r_rec + sigma_ext ** 2)

    mu_tot=inPar['mu_tot']
    sigma_tot=inPar['sigma_tot']


    # compute mu_tot from mu_syn and mean adaptation

    dV = grid.dV
    N_V = grid.N_V

    p,r_d,rND=inPar['p'],inPar['r_d'],inPar['rND']
    # data matrix Adt which is used for integration of p:
    # (1I-Adt)p(t+dt)=p(t) [implicit]

    reinject_ib = np.zeros(grid.N_V); reinject_ib[grid.ib] = 1.



    # calculate mass inside and outside the comp. domain
    int_P = np.sum(p*dV)
    # We assume no refracctory time for simplicity
    int_ref =inPar['int_ref']
    # normalize the density in the inner domain
    #todo change this
    p_marg = p/int_P
    # calculate the mean membrane voltage
    Vmean = np.sum(p_marg*grid.V_centers*dV)
    mass_timecourse = int_P + int_ref


    # normalize the probability distribution

    p*=(1.-int_ref)/int_P



    # drift coefficients
    v = get_v_numba(grid.N_V+1, grid.V_interfaces, EL,
                    taum, mu_tot)

    # Diffusion coefficient
    D = (sigma_tot ** 2) * 0.5

    # create banded matrix A in each time step
    Adt=matAdt_opt(grid.N_V,v,D,dV,dt)

    rhs = p.copy()
    # reinject either the noisy rate or the
    #rN_Ref=rN[n-n_ref]
    reinjection =  inPar['rNref']*(dt/dV)
    rhs[grid.ib] += reinjection
    Adt *= -1.
    Adt[1,:] += np.ones(grid.N_V)
    # solve the linear system
    p_new = solve_banded((1, 1), Adt, rhs)


    # compute rate
    if v[-1] != 0.0:
        r = v[-1]*((1.+exp((-v[-1]*dV)/D))/(1.-exp((-v[-1]*dV)/D)))*p_new[-1]
    else:
        r = 2*D/dV * p_new[-1]

    # add extra finite size noise 
    rN = r+Eta
    
    # solve delayed ODE for r_rec: d_rd/d_t = r(t-t_d)-rd/taud
    r_diff = (rND- r_d)
    r_dnew = r_d + dt * r_diff/taud
    r_rec = r_d

  
    return r_dnew,rN,r,p_new


