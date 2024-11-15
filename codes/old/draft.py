"""
A script to compute parameters for mean-field model
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.special import kn as bessel
from scipy.signal import convolve2d
from scipy.interpolate import RegularGridInterpolator
import pickle

# mozaik loader of parameters
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet, load_parameters


def hyperbolic_pdf2D(shape, step, alpha, theta, err = 0.0005):
    """Two dimensional hyperbolic distribution centered in the middle

    Normalization factor was taken from analytical result on wikipedia
    """
    # First compute normalization
    at = alpha * theta
    xs, ys = shape
    
    xmax = -np.log(err*bessel(1, at)*at)/(at)
    xmax = round(xmax, 1) * theta

    bins, res = divmod(2*xmax, step)
    if res:
        bins = bins+1
    if xs%2 == bins%2:   
        x = np.arange(0, 2*xmax , step)
    else:  # odd number
        x = np.arange(0, 2*xmax+step , step)
    x = x-x.mean()
    x = x[:, np.newaxis]
    pdf = np.exp(-alpha*np.sqrt(theta**2 + x**2 + x.T**2))
    # norm = np.trapz(np.trapz(pdf, dx=step, axis=0), dx=step, axis=0)
    norm = np.trapz(np.trapz(pdf, axis=0), axis=0)
    # To make it faster I can take the values from pdf already...
    xs = np.arange(xs)*step
    xs = xs - xs.mean()
    xs = xs[:,np.newaxis]

    ys = np.arange(ys)*step
    ys = ys -ys.mean()
    ys = ys[np.newaxis,:]

    pdf =  np.exp(-alpha*np.sqrt(theta**2 + xs**2 + ys**2))
    return pdf/norm

def interpolate2d(original_array, xs, ys, target_shape, method='linear'):
    # 'cubic' method is brutally slow
    interp_func = RegularGridInterpolator((ys, xs), original_array, method=method, bounds_error=False, fill_value=0.0)
    xs_new = np.linspace(xs[0], xs[-1], target_shape[0])
    ys_new = np.linspace(ys[0], ys[-1], target_shape[1])
    xs_new_grid, ys_new_grid = np.meshgrid(xs_new, ys_new)
    return interp_func((ys_new_grid, xs_new_grid))


pwd = os.getcwd()
path_to_model = '/home/pavel/work/mozaik/mozaik-models/LSV1M/'
os.chdir(path_to_model)  # Change the pwd
parameters = load_parameters('param/defaults')
os.chdir(pwd)  # change pwd to the original one



alpha = 0.013944  # 1/um ??
theta = 207.76  # um ??
x_size = 4000  # um
y_size = 4000  # um

column_rad = 150  # um
step = 1  # um

# Load orientation map
map_file = 'or_map_new_6x6'  # (62,62)
map_file = 'or_map_new_16x16'  # (358, 358)
with open(map_file, 'rb') as f:
    mmap = pickle.load(f, encoding="latin1")
x_bins, y_bins = mmap.shape
xs = np.linspace(-x_size/2, x_size/2, x_bins)
ys = np.linspace(-y_size/2, y_size/2, y_bins)
plt.contourf(mmap)
plt.savefig('or_map6x6.png')
plt.savefig('or_map16x16.png')


mmap.max()
mmap.min()

# Interpolate orientation map
# TODO: TBD
# u interpolace je potiz cyklicnost!!!!
# interpolovat radeji korelaci???

# interpolattion trustworthy between 0.25-0.75
# ormap = interpolate2d(mmap, xs, ys, (x_size//step+1, y_size//step +1))


full_field = (x_size//step+1, y_size//step +1)
column_field = (2*column_rad//step+1, 2*column_rad//step +1)
column_field_big = (4*column_rad//step+1, 4*column_rad//step +1)

pdf = hyperbolic_pdf2D(column_field, step, alpha, theta)
pdf_big = hyperbolic_pdf2D(column_field_big, step, alpha, theta)

xs_col = np.arange(-column_rad, column_rad+step, step)[:,np.newaxis]
ys_col = np.arange(-column_rad, column_rad+step, step)[np.newaxis]
circle = np.ones(column_field)
circle[xs_col**2+ys_col**2>column_rad**2] = 0

xs_big = np.arange(-2*column_rad, 2*column_rad+step, step)[:,np.newaxis]
ys_big = np.arange(-2*column_rad, 2*column_rad+step, step)[np.newaxis]
circle_big = np.ones(column_field_big)
circle_big[xs_big**2+ys_big**2>column_rad**2] = 0


# tohle je usekle!
conns = convolve2d(circle_big, pdf, mode='valid')
prob = (conns*circle).sum()/circle.sum()

conns_big = convolve2d(pdf_big, circle, mode='valid')
prob_big = (conns_big*circle).sum()/circle.sum()

pdf.shape
circle.shape
(pdf*circle).sum()

import time

start_time = time.time()
np.power(xs_big,2)
time.time()-start_time

start_time = time.time()
xs_big**2
time.time()-start_time

# each point gives the probability of connections inside the column for a given point
# circle is not normalized because it represents the summation/integration
# so the convolution is practically integration over the space for each point
# basically the circle works as step function, cutoff, whatever

# conns * circle  - this restricts to only values inside the column
# cannot use .mean(), because that would be counting points outside of the column
# I am interested only in points in the column
prob = (conns*circle).sum()/circle.sum()


# TODO: number of neurons in the column?
# TODO: number of connections in the column?


interp_func = RegularGridInterpolator((ys, xs), mmap, method="nearest", bounds_error=False, fill_value=0.0)


# interp_func(np.array([[,0]]))


# TODO: size of the orientation map
# TODO: correlation
# TODO: column choice





# TEST: numerical integration with the radius... that should be upper bound
(pdf*circle).sum()




parameters['sheets']['l4_cortex_exc']['params']['cell']['params'].keys()


# HOW TO TEST IT???
# normalized to sum!
hyperbolic_pdf2D((248, 248), 0.25, alpha, theta).sum()
hyperbolic_pdf2D((124, 124), 0.5, alpha, theta).sum()
hyperbolic_pdf2D((62, 62), 1, alpha, theta).sum()
hyperbolic_pdf2D((31, 31), 2, alpha, theta).sum()
hyperbolic_pdf2D((62, 62), 65.6, alpha, theta).sum()


plt.clf()
plt.contourf(circle)
plt.savefig('circle.png')


# ISSUEs:
#   1) delay functions (delay in synapses based on distance?)
#   2) short-term plasticity
#   3) variable strength of connections (multiplicativity)
#   4) Feedback loops (outside column, outside layer 4)

# IDEA: built in Mozaik - whenever running simulation generate parameters for mean-field






"""
The way to go with mozaik inbuilts 

# importing specific connectivity distributions

# Hyperbolic distribution (H)
from mozaik.connectors.modular_connector_functions import HyperbolicModularConnectorFunction

# Lateral connectivity (L)
from mozaik.connectors.modular_connector_functions import GaussianDecayModularConnectorFunction

# Correlation (of RF afferent) (C)
from mozaik.connectors.vision import V1CorrelationBasedConnectivity

# Orientation (Map Dependent) (O)
from mozaik.connectors.vision import MapDependentModularConnectorFunction


# Generate initial RFs - used later in (C)
from mozaik.connectors.meta_connectors import GaborConnector

# source = mozaik class? get_neuron_annotation method
# target =
# parameters =
# corr_map = V1CorrelationBasedConnectivity(source, target, parameters)

# index = index of a neuron
# corr_map = corr_map.evaluate(index)
# for each neuron I have single correlation map!


"""