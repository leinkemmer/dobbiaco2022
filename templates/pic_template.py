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
    # TODO: This is easy to do in O(n_cells n_particles) operations, but
    # it can also be done in O(n_particles) operation. The latter will be
    # a lot faster.

def compute_E(rho):
    # TODO
    Ehat[1:] = 1.0/(1j*2*pi/L*fftfreq(len(rhohat), 1)[1:]*len(rhohat))*rhohat[1:]
    Ehat[0] = 0.0
    return # TODO

def electric_energy(E):
    return 0.5*sum(E**2)*hx

def euler(part, deltat):
    # \dot{x} = v (TODO)

    # compute and interpolate electric field
    rho = compute_rho(part)
    E_grid = compute_E(rho)
    E_gridwe = zeros(n_cells+1)
    E_gridwe[0:-1] = E_grid
    E_gridwe[-1]   = E_grid[0]
    E = interp1d(xswe, E_gridwe)

    # \dot{v} = E (TODO)

    return part, electric_energy(E_grid)

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
fs.write('# t electric_energy\n')

t = 0.0
num_steps = int(ceil(t_final/deltat))
for i in range(num_steps):
    if t_final - t < deltat:
        deltat = t_final - t

    part, ee = euler(part, deltat)

    fs.write('{0: <30} {1: <30}\n'.format(t, ee))
    fs.flush()
    t += deltat
    print('\r', end='')
    print('t={}'.format(t), end='')

fs.close()

plot_all(part)

