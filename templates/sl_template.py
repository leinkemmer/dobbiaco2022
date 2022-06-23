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
    """compute the electric field from a 2D distribution function
    using FFT
    """
    #TODO
    return ex, phi

def advect_x(f,meshx,meshv,deltat):
    """Advection in x with cubic spline interpolation"""
    #TODO
    return f

def advect_v(f,ex,meshx,meshv,deltat):
    """advection in the v direction with cubic spline interpolation"""
    #TODO
    return f

