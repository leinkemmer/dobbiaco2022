from  numpy import *

class UniformMesh:
    def __init__(self, xmin, xmax, ncells):
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.deltax = (self.xmax-self.xmin)/ncells
        self.ncells = ncells
    def getx(self,i):
        return self.xmin + i*self.deltax
    def getpoints(self):
        return self.xmin + self.deltax*arange(self.ncells)

def landauex(t,alpha,kx):
    """Exact potential energy (|| E^2(t) ||_{L_2(x)}^2) for Landau problem for wavenumber kx (linearized, principal mode)
    """

    if (abs(kx-0.2) < 1e-5):
        # exact solution for kx=0.2
        om_i = -5.510e-5
        om_r = 1.064
        r = 1.129664
        phi = 0.00127377
    elif (abs(kx-0.3) < 1e-5):
        # exact solution for kx=0.3
        om_i = -0.0126
        om_r = 1.1598
        r = 0.63678
        phi = 0.114267
    elif (abs(kx-0.4) < 1e-5):
        # exact solution for kx=0.4
        om_i = -0.0661
        om_r = 1.2850
        r = 0.424666
        phi = 0.3357725
    elif (abs(kx-0.5) < 1e-5):
        # exact solution for kx=0.5
        om_i = -0.153359440777
        om_r = 1.41566194584
        r = 0.3677
        phi = 0.536245
    else:
        print("Exact solution for k = " +str(kx)+ " not implemented")

    return (pi/kx)*(4*alpha*r*exp(om_i*t)*cos(om_r*t-phi))**2

def plot_fH(x,v,f,phi):
    """creates a plot of f versus hamiltonian h=0.5*v**2 - phi(x)"""
    X,V=meshgrid(x,v)
    PHI,V1=meshgrid(phi,v)
    h= 0.5*V**2 - PHI
    pl.plot(ravel(h),ravel(f),',')

def bspline(p,j,x):
    """Return the value at x in [0,1[ of the B-spline with integer nodes of degree p with support starting at j.
    Implemented recursively using the de Boor's recursion formula"""
    x=float(x)
    assert ((x >= 0.0) & (x<=1.0))
    assert ((type(p)==int) & (type(j)==int))
    if p==0:
        if j==0:
            return 1.0
        else:
            return 0.0
    else:
        w = (x-j)/p
        w1 = (x-j-1)/p
        return w*bspline(p-1,j,x) + (1-w1)*bspline(p-1,j+1,x)

def interpSpline(p,f,mesh,alpha):
    """compute the interpolating spline of degree p of odd degree of a function f on a periodic uniform mesh, at
    all points xi-alpha"""
    assert ((p-1)/2. == int((p-1)/2.))  # check that p is odd
    n = size(f)
    assert (n == mesh.ncells)
    # compute eigenvalues of degree p b-spline matrix
    modes = 2*pi*arange(n)/n
    eig_bspl = bspline(p,-(p+1)/2,0.0)
    for j in range(1,(p+1)/2):
        eig_bspl += bspline(p,j-(p+1)/2,0.0)*2*cos(j*modes)
    # compute eigenvalues of cubic splines evaluated at displaced points
    ishift = floor (-alpha/mesh.deltax)
    beta = -ishift - alpha/mesh.deltax
    eigalpha = zeros(n,dtype=complex)
    for j in range(-(p-1)/2,(p+1)/2+1):
        eigalpha += bspline(p,j-(p+1)/2,beta)*exp((ishift+j)*1j*modes)
    # compute interpolating spline using fft and properties of circulant matrices
    interpSpline = real(fft.ifft( fft.fft(f) * eigalpha / eig_bspl))
    return interpSpline

def interpCubicSpline(f,mesh,alpha):
    """compute the interpolating cubic spline of a function f on a periodic uniform mesh, at
    all points xi-alpha"""
    n = size(f)
    assert (n == mesh.ncells)
    # compute eigenvalues of cubic spline matrix
    modes = 2*pi*arange(n)/n
    eigs3 = 2.0/3.0 + 2.0/6.0*cos(modes)
    # compute eigenvalues of cubic splines evaluated at displaced points
    ishift = floor (-alpha/mesh.deltax)
    beta = -ishift - alpha/mesh.deltax
    c2 = beta**3 / 6.0
    c1 = (4.0 - 6.0 * (1.0 - beta)**2 + 3.0 * (1.0 - beta)**3) / 6.0
    c0 = (4.0 - 6.0 * beta**2 + 3.0 * beta**3) / 6.0
    cm1 = (1.0 - beta)**3 / 6.0
    eigalpha = (cm1*exp((ishift-1)*1j*modes) + c0*exp((ishift)*1j*modes)
                + c1*exp((ishift+1)*1j*modes) + c2*exp((ishift+2)*1j*modes))
    # compute interpolating spline using fft and properties of circulant matrices
    interpSpline = real(fft.ifft( fft.fft(f) * eigalpha / eigs3))
    return interpSpline

def poisson(f,meshx,meshv):
    "compute the electric field from a 2D distribution function"
    nx = size(f,1)
    assert (nx == meshx.ncells)
    # compute rho adding neutralizing background
    rho = 1.0 - meshv.deltax * sum(f,axis=0)

    # compute Ex using that ik*Ex = rho
    modes = 2*pi*arange(nx/2+1)/(meshx.xmax-meshx.xmin)
    rhok = fft.rfft(rho)
    modes[0] = 1.  # avoid division by 0
    ex = real(fft.irfft(-1j*rhok/modes))
    phi = real(fft.irfft(rhok/(modes**2)))
    return ex, phi

def advect_x(f,meshx,meshv,deltat):
    """Advection in x"""
    for j in range(meshv.ncells):
        alpha = meshv.getx(j) * deltat
        #f[j,:] = interpSpline(3,f[j,:],meshx,alpha)
        f[j,:] = interpCubicSpline(f[j,:],meshx,alpha)
    return f

def advect_v(f,ex,meshx,meshv,deltat):
    """adecvection in the v direction"""
    for i in range(meshx.ncells):
        alpha = - ex[i] * deltat
        #f[:,i] = interpSpline(3,f[:,i],meshv,alpha)
        f[:,i] = interpCubicSpline(f[:,i],meshv,alpha)
    return f


# setup the mesh/initial value
meshv= UniformMesh(-8.,8.,128)
meshx= UniformMesh(0.,4*pi,128)
x=meshx.getpoints()
v=meshv.getpoints()
X,V = meshgrid(x,v)
eps = 0.01
v0 = 2.4
k = 2*pi/(meshx.xmax-meshx.xmin)
#f0 = (1.+eps*cos(k*X))*(exp(-0.5*((V-v0)**2))+exp(-0.5*(((V+v0)**2))))/sqrt(2.*pi)
f0 = (1.+eps*cos(k*X))*exp(-0.5*V**2)/sqrt(2.*pi)
f=f0.copy()
ex,phi=poisson(f,meshx,meshv)


import time
import matplotlib.pyplot as pl
pl.subplot(121)
imf = pl.imshow(f0,extent=(meshx.xmin,meshx.xmax,meshv.xmin,meshv.xmax),origin='lower')
pl.subplot(122)
ime = pl.plot(x,ex)

# time loop
nbiter = 100
deltat = 0.2
l2_f = zeros(nbiter+1)
l2_f[0] = sum(f**2)
l2_ex = zeros(nbiter+1)
l2_ex[0] = sum(ex**2)
t = deltat*arange(nbiter+1)
for n in range(nbiter):
    f=advect_v(f,ex,meshx,meshv,0.5*deltat)
    f=advect_x(f,meshx,meshv,deltat)
    ex,phi= poisson(f,meshx,meshv)
    f=advect_v(f,ex,meshx,meshv,0.5*deltat)
    # replot every ten iterations
    if n%10==0:
        pl.subplot(121)
        imf.set_data(f)
        pl.title('distribution function, iter='+str(n+1))
        #time.sleep(.1)
        pl.subplot(122)
        pl.plot(x,ex)
        pl.title('Electric field, iter='+str(n+1))
    l2_ex[n+1] = sum(ex**2)
pl.show()   


pl.semilogy(t,0.1*l2_ex)
pl.semilogy(t,landauex(t,eps,k))
pl.show()

