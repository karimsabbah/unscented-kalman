import numpy as np
from scipy.linalg import cholesky
import math
from numpy.random import randn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.patches as mpatches


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False




class SigmaWeights:

    def __init__(self, x, P, alpha, beta, kappa):
        self.x = x
        self.P = P
        self.n = len(x)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.sigmaPoints = np.zeros((2*self.n+1,self.n))
        self.weights = np.zeros(2*self.n+1)

        self._lambda = (alpha**2)*(self.n+kappa) - self.n


    def computeSigmaPoints(self):
        self.A = (self.n+self._lambda)*self.P
        if isPD(self.A):
            self.U = cholesky(self.A)
        else:
            self.A_pd = nearestPD(self.A)
            self.U = cholesky(self.A_pd)
        self.sigmaPoints[0] = self.x
        for i in range(self.n):
            self.sigmaPoints[i+1] = self.x + self.U[i]
            self.sigmaPoints[self.n+i+1] = self.x - self.U[i]


    def computeWeights(self):
        self.Wc = np.full(2*self.n+1, 1./(2*(self.n + self._lambda)))
        self.Wm = np.full(2*self.n+1, 1./(2*(self.n + self._lambda)))
        self.Wc[0] = self._lambda/(self.n+self._lambda) + (1. - alpha**2+beta)
        self.Wm[0] = self._lambda / (self.n +self._lambda)



def stateTransition(sigmas, f, state_time, dt):

    sigmasState = np.zeros_like(sigmas.sigmaPoints)
    for i, s in enumerate(sigmas.sigmaPoints):
        Rt=np.log2((16000*c)/(Rcref+Rdref))
        p_old=s[0]
        Rc=s[1]
        C=s[2]
        Rd=s[3]
        Zero=s[4]

        sol= solve_ivp(lambda t,y:f(t, y, C ,Rd ,Rc),
                       (state_time, state_time+dt), np.array([p_old]),
                       method='LSODA',
                       atol=1e-8,
                       rtol=1e-6)
        p=sol.y[0]
        p_new = p[-1]
        sigmasState[i,0] = p_new
        sigmasState[i,1] = Rc
        sigmasState[i,2] = C
        sigmasState[i,3] = Rd
        sigmasState[i,4] = Rc+Rd-Rt


    return sigmasState

def unscentedTransform(sigmas, sigmasState, Q):

    mean = np.dot(sigmas.Wm, sigmasState)
    covariance = np.zeros((sigmas.n,sigmas.n))
    for i in range(2*sigmas.n+1):
        temp = (sigmasState[i].reshape((sigmas.n,1))-mean.reshape((sigmas.n,1)))
        temp2 = np.matmul( temp, np.transpose(temp) )
        covariance += sigmas.Wc[i]*temp2

    covariance += Q



    return mean, covariance


def h_func(x):

    return np.array([x[0],x[4]])


def computeMeasurementSigmas(h, sigmasState):

    sigmasMeasurement = []
    for i in range(sigmasState.shape[0]):
        sigmasMeasurement.append(h(sigmasState[i,:]))

    return np.array(sigmasMeasurement)


def computeMeasurementMeanAndCovariance(sigmas, sigmasZ, R):

    lenColZ = sigmasZ.shape[1]
    mean = np.dot(sigmas.Wm, sigmasZ)
    covariance = np.zeros((lenColZ,lenColZ))
    for i in range(2*sigmas.n+1):
        temp = (sigmasZ[i].reshape((lenColZ,1))-mean.reshape((lenColZ,1)))
        temp2 = np.matmul( temp, np.transpose(temp) )
        covariance += sigmas.Wc[i]*temp2

    covariance += R

    return mean, covariance


def computeCrossCovariance_Pxz(sigmas, meanPredict, meanZ, sigmasState, sigmasZ):

    lenColZ = sigmasZ.shape[1]
    covariance = np.zeros((sigmas.n,lenColZ))
    for i in range(2*sigmas.n+1):
        temp1 = (sigmasState[i].reshape((sigmas.n,1))-meanPredict.reshape((sigmas.n,1)))
        temp2 = (sigmasZ[i].reshape((lenColZ,1))-meanZ.reshape((lenColZ,1)))
        temp3 = np.matmul( temp1, np.transpose(temp2) )
        covariance += sigmas.Wc[i]*temp3

    return covariance


def computeKalmanGain(crossCov, covZ):

    return np.dot(crossCov,np.linalg.inv(covZ))


def mkNoisyMeasurements(f, y0, time_noisy, std_p, std_constrain):

    C = np.log2(((2.5e-5)/c)/Cref)
    Rd = np.log2((13000*c)/Rdref)
    Rc = np.log2((1600*c)/Rcref)

    sol = solve_ivp(lambda t,y:f(t,y,C,Rd,Rc),
                    (time_noisy[0],time_noisy[-1]), np.array([y0]), t_eval=time_noisy,
                    method='LSODA')
    p = sol.y[0]
    zP = p+np.random.normal(0,std_p,len(p))
    zR = np.random.normal(0,std_constrain,len(p))
    z = np.array([zP,zR])


    return p, z


def computeFinalSol(f, y0, time_sol, *args):

    C = args[1]
    Rd = args[2]
    Rc = args[0]

    sol = solve_ivp(lambda t,y:f(t,y,C,Rd,Rc),
                    (time_sol[0],time_sol[-1]), np.array([y0]), t_eval=time_sol,
                   method='LSODA')
    p = sol.y[0]

    return p


def f(t, y, C ,Rd ,Rc):

    t = t%(60/HR)

    return 1/C*(Q(t)+Rc*C*dQ(t)-y/Rd+(Q(t))*Rc/Rd)

def Q(t):
    t=t%(60/HR)
    S = 0.065
    return (1.4/(S*np.sqrt(np.pi)))*np.exp(-((t-0.36)**2)/(2*S**2))+5

def dQ(t):
    t=t%(60/HR)
    dt = 1e-8
    return (Q(t+dt)-Q(t))/dt



if __name__ == "__main__":

    global c,Cref,Rcref,Rdref,Pref

    c = 0.000750061683  #unit conversion 1 dyn/cm^2=0.0007750061 mmHg
    Pref=50
    Cref=(2e-5)/c
    Rcref=1500*c
    Rdref=12000*c
    Rt=np.log2((500*c)/Rcref)+np.log2((25000*c)/Rdref)

    x = np.array([np.log2(80/Pref),
                  np.log2((500*c)/Rcref),
                  np.log2((0.00065/c)/Cref),
                  np.log2((25000*c)/Rdref),
                  np.log2((500*c)/Rcref)+np.log2((25000*c)/Rdref)])


    P = np.diag([8,8,8,8,8])


    cycles = 2

    alpha = 0.2
    beta = 2.
    kappa = 2
    R_adjust = 0.005
    z_std = 0.1
    std_constrain = 0.001
    np.random.seed(3)
    dt = 1/500
    HR = 80
    cycle_time = 60/HR
    time = np.linspace(0, (60/HR)*cycles+dt, num=((60/HR)*cycles+dt)/dt)
    time_noise = np.linspace(0, (60/HR)*5+dt, num=((60/HR)*5+dt)/dt)
    y0=1
    solNoise, zNoise  = mkNoisyMeasurements(f, y0, time_noise, z_std, std_constrain)
    sol = solNoise[:int(cycles*(60/HR+dt)/dt)]
    z = zNoise[:int(cycles*(60/HR+dt)/dt)]
    std = np.std(z)

    R = np.diag([R_adjust, 0.0000850])


    Qprocess =np.array([[(dt**2),       (dt**2),      (dt**2),       (dt**2),      (dt**2)],
                       [ (dt**2),       (dt**2),      (dt**2),       (dt**2),      (dt**2)],
                       [ (dt**2),       (dt**2),      (dt**2),       (dt**2),      (dt**2)],
                       [ (dt**2),       (dt**2),      (dt**2),       (dt**2),      (dt**2)],
                       [ (dt**2),       (dt**2),      (dt**2) ,      (dt**2),       (dt**2)]])*650

    predictions = []
    state = []
    res = []
    Filter_Order=[]
#    state.append(x) Uncomment if you want to show initial guess. Size of arrays will shift by 1 

    bitch_time = 0
    for i in range(len(time)):
        bitch_time += dt
        Filter_Order.append(P[0,0])
#        Pplot=np.array([[P[0,0],P[0,1]],[P[1,0],P[1,1]]])
#        plot_covariance_ellipse((x[0],x[1]),Pplot, fc='g', alpha=0.2,)
#        plt.gca().grid(b=False);
        print('loop {}'.format(i))
        sigmas = SigmaWeights(x, P, alpha, beta, kappa)
        sigmas.computeSigmaPoints()
        sigmas.computeWeights()
        sigma_point_locations=np.array([[sigmas.sigmaPoints[:,0]],[sigmas.sigmaPoints[:,1]]])
#        plt.plot(sigma_point_locations[0,:],sigma_point_locations[1,:],'o')
        sigmasState = stateTransition(sigmas,f, bitch_time, dt)
        meanPredict, covPredict = unscentedTransform(sigmas, sigmasState, Qprocess)
        predictions.append(meanPredict)
        sigmasZ = computeMeasurementSigmas(h_func, sigmasState)
        meanZ, covZ = computeMeasurementMeanAndCovariance(sigmas, sigmasZ, R)
        crossCov = computeCrossCovariance_Pxz(sigmas, meanPredict, meanZ, sigmasState, sigmasZ)
        K = computeKalmanGain(crossCov, covZ)
        residual = np.subtract(z[:,i], meanZ)
        res.append(residual)
        x = np.add( meanPredict, np.dot(K,residual) )
        P = np.subtract( covPredict, np.dot(K, np.dot(covZ,np.transpose(K))) )
        state.append(x)


    plt.show()

    Fil_Ord=np.sqrt(np.array(Filter_Order))
    predictions = np.array(predictions)
    res = np.array(res)
    state = np.array(state)
    print(state[-1,:])
    indexStartLastCycle = int((cycles-1)*(60/HR)/dt)
    #Rc = np.mean(Rcref/c*(2**(state[indexStartLastCycle:-1,1])))
    #C = np.mean(Cref*c*(2**(state[indexStartLastCycle:-1,2])))
    #Rd = np.mean(Rdref/c*(2**(state[indexStartLastCycle:-1,3])))
    Rc = Rcref/c*(2**(state[-1,1]))
    C = Cref*c*(2**(state[-1,2]))
    Rd = Rdref/c*(2**(state[-1,3]))
    print('Rc = {}\nC = {}\nRd = {}'.format(Rc,C,Rd))
    state_normal=Pref*2**(state)
    finalSol = computeFinalSol(f, 80, time, Rc*c, C/c, Rd*c)
    realSol = computeFinalSol(f, 80, time, 1600*c, 2.5e-5/c, 13000*c)


#------------------Plot of pressure Vs time in Transformed Space--------------


    plt.figure(1, figsize=(10,10))

    plt.subplot(2,1,1)

    plt.plot(time, sol, 'r')
    green_patch = mpatches.Patch(color='r', label='Process Model')

    plt.plot(time, z[0], 'b')
    blue_patch = mpatches.Patch(color='b', label='Pressure Measurements')

    plt.plot(time, state[:,0], 'g')
    red_patch = mpatches.Patch(color='g', label='Filter Estimate')

    plt.legend(handles=[green_patch,blue_patch,red_patch], loc= 'lower right')

    plt.title('Pressure vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Transformed Pressure')

#---------------------Plot of Pressure vs time---------------------------

    plt.subplot(2,1,2)

    plt.plot(time, realSol, 'black')
    black_patch = mpatches.Patch(color='black', label='ODE solved with True Parameters')
    plt.plot(time, finalSol, 'm', label='ODE sol with UKF parameters')
    magenta_patch = mpatches.Patch(color='m', label='ODE sol with UKF parameters')

    plt.legend(handles=[black_patch,magenta_patch], loc= 'lower right')

    plt.title('Pressure vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')

    plt.show()


#-------------------------State plots------------------------

    plt.figure(2, figsize=(16,12))

    plt.subplot(2,2,1)
    plt.plot(time, state[:,0],'g')
    plt.plot(time, realSol,'--')
    plt.title('Pressure vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.legend(['Calculated Pressure','Target Pressure'])

    plt.subplot(2,2,2)
    plt.plot(time, Rcref/c*(2**(state[:,1])),'g')
    plt.plot(time,1600*np.ones_like(time),'--')
    plt.title('Resistance Rc vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Resistance (dyn-s/cm^3)')
    plt.legend(['Calculated Resistance','Target Resistance'])

    plt.subplot(2,2,3)
    plt.plot(time, Cref*c*(2**(state[:,2])),'r')
    plt.plot(time, 2.5e-5*np.ones_like(time),'--')
    plt.title('C')
    plt.legend(['Calculated Capacitance','Target Capacitance'])
    plt.title('Capacitance Rd vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Capacitance (cm^3/dyn)')

    plt.subplot(2,2,4)
    plt.plot(time, Rdref/c*(2**(state[:,3])),'b')
    plt.plot(time, 13000*np.ones_like(time),'--')
    plt.legend(['Calculated Resistance','Target Resistance'])
    plt.title('Resistance Rd vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Resistance (dyn-s/cm^3)')

    plt.show()

#---------------------------Residuals---------------------------------------


    fig, ax1 = plt.subplots(1, 1, sharex=True)
    ax1.plot(time, 3*Fil_Ord, '--', color='black')
    ax1.plot(time, -3*Fil_Ord, '--', color='black')
    ax1.plot(time, res[:len(res),0], '-',color='red')
    ax1.fill_between(time, 3*Fil_Ord, -3* Fil_Ord, alpha=0.3)
    ax1.legend(['Residual Bounds (3*Sigma)'])

    plt.title('Residual vs Time ')
    plt.xlabel('Time (s)')
    plt.ylabel('Residual')

    plt.show()









