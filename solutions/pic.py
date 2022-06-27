from pylab import *
from scipy.interpolate import interp1d
import numpy.random as random
from scipy import stats

n_cells     = 100
n_particles = 100*n_cells

t_final = 30.0
deltat  = 0.1

# nonlinear Landau damping initial value
alpha=0.5
L = 4*pi
def distribution_x(x):
    return (1.0 + alpha*cos(0.5*x))/L

def distribution_v(v):
    return exp(-0.5*v**2)/sqrt(2*pi)

xs = linspace(0, L, n_cells, endpoint=False)
xswe = linspace(0, L, n_cells+1, endpoint=True)
hx = xs[1]-xs[0]

def compute_rho(part):
    rho = zeros(n_cells) + 1
    w = L/float(n_particles)
    for x in part[0,:]:
        i_nearest = int(round(x/hx))
        x_closest= xswe[i_nearest] 

        for offset in range(-9,10):
            dist = x-x_closest + hx*offset
            rho[(i_nearest + offset) % n_cells] -= w*exp(-0.5*(dist/hx)**2)/(hx*sqrt(2*pi))
    return rho

def compute_mass(rho):
    return sum(rho)*hx + 1.0

def compute_momentum(part):
    w = L/float(n_particles)
    return w*sum(part[1,:])

def compute_kinenergy(part):
    w = L/float(n_particles)
    return w*sum(0.5*part[1,:]**2)


def compute_E(rho):
    rhohat = fft(rho)
    Ehat = 1j*zeros(len(rhohat))
    Ehat[1:] = 1.0/(1j*2*pi/L*fftfreq(len(rhohat), 1)[1:]*len(rhohat))*rhohat[1:]
    Ehat[0] = 0.0
    return real(ifft(Ehat))

def electric_energy(E):
    return 0.5*sum(E**2)*hx

def euler(part, deltat):
    # compute and interpolate electric field
    rho = compute_rho(part)
    E_grid = compute_E(rho)
    E_gridwe = zeros(n_cells+1)
    E_gridwe[0:-1] = E_grid
    E_gridwe[-1]   = E_grid[0]
    E = interp1d(xswe, E_gridwe)

    partx = part[0,:].copy()

    # \dot{x} = v
    part[0,:] = (part[0,:] + deltat*part[1,:]) % L
    
    # \dot{v} = E
    part[1,:] -= deltat*E(partx)

    return part, electric_energy(E_grid), compute_mass(rho)


def plot_all(part):
    rho = compute_rho(part)
    E = compute_E(rho)

    scatter(part[0,:], part[1,:])
    plot(xs, rho, 'r-')
    plot(xs, E, 'g-')
    show()



# initialize the problem
class lldistribution(stats.rv_continuous):
    def _pdf(self, x):
        return distribution_x(x)
llspace = lldistribution(a=0.0, b=L)
class lldistributionvel(stats.rv_continuous):
    def _pdf(self, v):
        return distribution_v(v)
llveloc = lldistributionvel()

part = zeros((2,n_particles))
E    = zeros(n_cells)

for i in range(n_particles):
    part[0,i] = llspace.rvs()
    part[1,i] = llveloc.rvs()

plot_all(part)


#
# run the simulation
#
fs = open('evolution.data', 'w')
fs.write('# t electric_energy mass momentum\n')


t = 0.0
num_steps = int(ceil(t_final/deltat))
for i in range(num_steps):
    if t_final - t < deltat:
        deltat = t_final - t

    momentum = compute_momentum(part)
    kinenergy = compute_kinenergy(part)
    part, ee, mass = euler(part, deltat)

    fs.write('{0: <30} {1: <30} {2: <30} {3: <30} {4: <30}\n'.format(t, ee, mass, momentum, kinenergy + ee))

    fs.flush()
    t += deltat
    print('\r', end='')
    print('t={}'.format(t), end='')

fs.close()

plot_all(part)

