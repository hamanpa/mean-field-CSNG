import numpy as np
from scipy.special import erfc, erfcinv
from scipy.optimize import minimize, curve_fit

import pyNN.nest as sim
import nest

sim.native_synapse_type('static_synapse')

# Helpers

def activity_from_spikes(spike_times: np.array, time_interval: float) -> float:
    """
    time_interval in ms
    """
    return spike_times[spike_times > (time_interval/2)].size/(time_interval/2*1e-3)

def convert_to_arrays(*args):
    return [convert_to_array(arg) for arg in args]

def convert_to_array(arg):
    if type(arg) in{int, float}:
        return np.array([arg])
    elif type(arg) == np.ndarray:
        return arg
    elif type(arg) == list:
        return np.array(arg)

def move_and_rescale(vals, expansion_point, expansion_norm, axis=-1):
    """ Moves the values by expansion_point and rescales them by expansion_norm along a given axis


    """

    assert type(expansion_point) == type(expansion_norm)

    if type(expansion_point) is np.ndarray:
        assert expansion_point.size == expansion_norm.size == vals.shape[axis]

    new_vals = np.moveaxis(vals, axis, -1)
    new_vals = (new_vals - expansion_point)/expansion_norm
    new_vals = np.moveaxis(new_vals, -1, axis)

    return new_vals

def flatten_and_remove_nans(*vals: np.ndarray):
    """Returns flattened version of vals with nan values removed
    """
    assert np.all([val.shape == vals[0].shape for val in vals])
    
    # 1. flatten the values
    new_vals = [val.flatten() for val in vals]
    
    # 2. find nans
    nans = np.zeros_like(new_vals[0],dtype=bool)
    for val in new_vals:
        nans = nans | np.isnan(val)

    # 3. remove nans
    new_vals = [val[~nans] for val in new_vals]

    return new_vals

# Plotting

# Simulations

def simulate_neuron_pynn(nu_e, nu_i, neuron_params, 
                         exc_syn_params, inh_syn_params, 
                         exc_syn_number, inh_syn_number, 
                         synapse_type, w=None,
                         simulation_time=1000.0, poisson_input=True):
 
    sim.setup(timestep=0.1)
    if w is not None:
        a,b = neuron_params['a'], neuron_params['b']
        neuron_params['a'], neuron_params['b'] = 0., 0.
        neuron_params['i_offset'] = -w
    neuron = sim.Population(1, sim.EIF_cond_exp_isfa_ista(**neuron_params))
    synapse_exc = sim.native_synapse_type(synapse_type)(**exc_syn_params)
    synapse_inh = sim.native_synapse_type(synapse_type)(**inh_syn_params)
    connector = sim.AllToAllConnector()  # Connect all-to-all
    if poisson_input:
        poisson_input_exc = sim.Population(exc_syn_number, sim.SpikeSourcePoisson(rate=nu_e))
        sim.Projection(poisson_input_exc, neuron, connector, synapse_type=synapse_exc, receptor_type='excitatory')

        poisson_input_inh = sim.Population(inh_syn_number, sim.SpikeSourcePoisson(rate=nu_i))
        sim.Projection(poisson_input_inh, neuron, connector, synapse_type=synapse_inh, receptor_type='inhibitory')

    else:
        raise NotImplementedError
        # TODO: for constant input there has to be randomization so that not all
        # inputs arrive at the same moment, but are uniformly distributed
        if nu_e > 0:
            spike_times_exc = np.arange(0, simulation_time, simulation_time/nu_e)
            const_input_exc = sim.Population(synapse_number, sim.SpikeSourceArray(spike_times=spike_times_exc))
            sim.Projection(const_input_exc, neuron, connector, synapse_type=synapse_exc)

        if nu_i > 0:
            spike_times_inh = np.arange(0, simulation_time, simulation_time/nu_i)
            const_input_inh = sim.Population(synapse_number, sim.SpikeSourceArray(spike_times=spike_times_inh))
            sim.Projection(const_input_inh, neuron, connector, synapse_type=synapse_inh)
    neuron.record(['v', 'spikes', 'w', 'gsyn_exc', 'gsyn_inh'])
    sim.run(simulation_time)

    membrane_potential = neuron.get_data().segments[0].filter(name="v")[0]
    spike_times = neuron.get_data().segments[0].spiketrains[0]
    adaptation = neuron.get_data().segments[0].filter(name="w")[0]
    sim.end()
    if w is not None:
        neuron_params['a'], neuron_params['b'] = a,b
        neuron_params['i_offset'] = 0.
    return membrane_potential, spike_times, adaptation

def simulate_neuron_nest(nu_e, nu_i, neuron_params, 
                         exc_syn_params, inh_syn_params, 
                         exc_syn_number, inh_syn_number, 
                         simulation_time=1000.0, poisson_input=True):
    raise NotImplementedError

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})

    neuron = nest.Create("aeif_cond_exp", 1, params=neuron_params)

    # TODO: add tsodyks synapse to aeif neuron
    # make it exc and inh
    poisson_input = nest.Create("poisson_generator", params={'rate': 10.0})  # spikes/s
    parrot_neuron = nest.Create("parrot_neuron")

    spike_recorder = nest.Create("spike_recorder")
    multimeter = nest.Create("multimeter")
    multimeter.set(record_from=["V_m", "w", "g_ex", "g_in"])

    nest.Connect(poisson_input, parrot_neuron)
    nest.Connect(parrot_neuron, neuron_exc, syn_spec={'weight': 0.22, 'synapse_model': 'static_synapse'})
    nest.Connect(parrot_neuron, spike_recorder)
    nest.Connect(multimeter, neuron_exc)

    nest.Simulate(1000.0)
    data = multimeter.get("events")
    input_spikes = spike_recorder[0].get("events")['times']

    return data, input_spikes

# Transfer function

def mean_potential_fluctuations(nu_e, nu_i, w, params):
    """
    
    Parameters
    ----------
    nu_e : float or 1D np.ndarray
        Excitatory input rate in Hz.

    nu_i : float or 1D np.ndarray
        Inhibitory input rate in Hz.

    w : float or 1D np.ndarray or 2D np.ndarray
        Adaptations

        w == 0, corresponds to adaptation not treated specifically (Zerlaut model)
        w == 1D, array corresponds to adaptation as input variable, 
                 hold constant in simulations
        w == 2D, corresponds to adaptation treated as output variable, 
                 for a given (nu_e, nu_i) there is a w
    """
    nu_e = convert_to_array(nu_e)
    nu_i = convert_to_array(nu_i)
    w = convert_to_array(w)
    if w.ndim == 1:
        # w is regarded as input variable, same as nu_e and nu_i
        nu_e = nu_e[:,np.newaxis, np.newaxis]
        nu_i = nu_i[np.newaxis, :, np.newaxis]
        w = w[np.newaxis, np.newaxis, :]
    elif w.ndim == 2 and w.shape==(nu_e.size, nu_i.size):
        # w is regarded as output variable, i.e. for a given (nu_e, nu_i) there is a w
        nu_e = nu_e[:,np.newaxis]
        nu_i = nu_i[np.newaxis, :]
    else:
        raise ValueError("Invalid shape of w")
    nu_e[nu_e<1e-9]=1e-9
    nu_i[nu_i<1e-9]=1e-9
    
    def stp_stationary_limit(rate, U, tau_rec, **kwarg):
        """Return stationary limit of short-term plasticity."""
        exp = np.exp(-1/(rate*1e-3*tau_rec))
        return U*(1-exp)/(1-(1-U)*exp)

    if params['synapse_type'] == 'tsodyks_synapse':
        stp_e = stp_stationary_limit(nu_e, **params['exc_syn_params'])
        stp_i = stp_stationary_limit(nu_i, **params['inh_syn_params'])
    else:
        stp_e = 1.
        stp_i = 1.


    tau_e = params['neuron_params']['tau_syn_E']
    num_e = params['exc_syn_number']
    weight_e = params['exc_syn_params']['weight']
    mu_Ge = nu_e * 1e-3 * tau_e * num_e * weight_e * stp_e  
    # 1e-3 to convert Hz to kHz --> result in nS

    tau_i = params['neuron_params']['tau_syn_I']
    num_i = params['inh_syn_number']
    weight_i = params['inh_syn_params']['weight']
    mu_Gi = nu_i * 1e-3 * tau_i * num_i * weight_i * stp_i  
    # 1e-3 to convert Hz to kHz --> result in nS

    g_L = params['neuron_params']['cm'] / params['neuron_params']['tau_m'] * 1e3
    # 1e3 to covert to nS

    mu_G = mu_Ge + mu_Gi + g_L
    tau_eff = params['neuron_params']['cm'] / mu_G * 1e3
    # 1e3 to convert to ms

    mu_Ve = mu_Ge * params['neuron_params']['e_rev_E'] 
    mu_Vi = mu_Gi * params['neuron_params']['e_rev_I']
    mu_Vl = g_L * params['neuron_params']['v_rest']

    mu_V = (mu_Ve + mu_Vi + mu_Vl - w) / mu_G

    u_e = weight_e * stp_e *(params['neuron_params']['e_rev_E'] - mu_V) / mu_G
    u_i = weight_i * stp_i *(params['neuron_params']['e_rev_I'] - mu_V) / mu_G

    s_e = num_e * nu_e * 1e-3 * (u_e*tau_e)**2 / (2*(tau_eff+tau_e))
    s_i = num_i * nu_i * 1e-3 * (u_i*tau_i)**2 / (2*(tau_eff+tau_i))
    sigma_V = np.sqrt(s_e + s_i)

    t_e = num_e * nu_e * 1e-3 * (u_e*tau_e)**2
    t_i = num_i * nu_i * 1e-3 * (u_i*tau_i)**2
    tau_V = (t_e + t_i) / (t_e/(tau_eff+tau_e) + t_i/(tau_eff+tau_i))

    tau_VN = tau_V / params['neuron_params']['tau_m']
    return mu_V.squeeze(), sigma_V.squeeze(), tau_V.squeeze(), tau_VN.squeeze()

def v_eff_from_data(nu_out, mu_V, sigma_V, tau_V, tau_VN):
    """Computes and return V_eff from the data (nu_out, mu_V, sigma_V, tau_V, tau_VN)"""
    mu_V, sigma_V, tau_V, tau_VN = convert_to_arrays(mu_V, sigma_V, tau_V, tau_VN)
    nu_out[nu_out<1e-9]=1e-9
    tau_V[tau_V<1e-9]=1e-9
    return np.sqrt(2)*sigma_V * erfcinv(2*tau_V * nu_out * 1e-3) + mu_V

def v_eff_function_bare(points, *coefs):
    assert len(coefs) in {4, 10}, f'{len(coefs)}'  # linear or quadratic expansion
    assert points.shape[0] == 3
    x_vals = np.concatenate([np.ones_like(points[0])[np.newaxis], points], axis=0)
    if len(coefs) == 10:  # quadratic expansion
        x1, x2, x3 = points
        point_quad = np.stack([x1*x1, x1*x2, x1*x3, x2*x2, x2*x3, x3*x3])
        x_vals = np.concatenate([x_vals, point_quad], axis=0)
    x_vals = np.moveaxis(x_vals, 0, -1)
    return x_vals @ np.array(coefs)

def v_eff_function(coefs: list, mu_V, sigma_V, tau_VN, 
          expansion_point = np.array([-60e-3, 4e-3, 0.5]),
          expansion_norm = np.array([10e-3, 6e-3, 1.])):
    """This function expands v_eff_function_bare by shift to expansion_point and
    by rescaling to expansion_norm"""
    mu_V, sigma_V, tau_VN = convert_to_arrays(mu_V, sigma_V, tau_VN)
    point = np.stack([mu_V, sigma_V, tau_VN], axis=0)
    point = move_and_rescale(point, expansion_point, expansion_norm, axis=0)
    return v_eff_function_bare(point, *coefs)

def transfer_function(nu_e, nu_i, w, v_eff_coefs, expansion_point, expansion_norm, params):
    mu_V, sigma_V, tau_V, tau_VN = mean_potential_fluctuations(nu_e, nu_i, w, params)
    v_eff = v_eff_function(v_eff_coefs, mu_V, sigma_V, tau_VN, expansion_point, expansion_norm)

    return 1/(2*tau_V*1e-3) * erfc((v_eff-mu_V)/(np.sqrt(2)*sigma_V))

def get_transfer_function(v_eff_coefs, expansion_point, expansion_norm, params):
    """returns transfer fucrtion as a function of nu_e, nu_i and w"""

    def transfer_function(nu_e, nu_i, w):
        mu_V, sigma_V, tau_V, tau_VN = mean_potential_fluctuations(nu_e, nu_i, w, params)
        v_eff = v_eff_function(v_eff_coefs, mu_V, sigma_V, tau_VN, expansion_point, expansion_norm)

        return 1/(2*tau_V*1e-3) * erfc((v_eff-mu_V)/(np.sqrt(2)*sigma_V))
    return transfer_function

# Fitting

def v_eff_curve_fit(coefs_init, y_data, mu_V, sigma_V, tau_VN, expansion_point, expansion_norm):
    """Returns coefficients of fitting v_eff_function with scipy.optimize.curve_fit
    """

    # 1. move the point
    point = np.stack([mu_V, sigma_V, tau_VN], axis=0)
    point = move_and_rescale(point, expansion_point, expansion_norm, axis=0)
    mu_V, sigma_V, tau_VN = point

    # 2. Prepare data, flatten and remove nans
    y_data, mu_V, sigma_V, tau_VN = flatten_and_remove_nans(y_data, mu_V, sigma_V, tau_VN)
    x_data = np.stack([mu_V, sigma_V, tau_VN], axis=0)
    coefs_init[:0] = [y_data.mean()] 
    coefs, pcov = curve_fit(v_eff_function_bare, x_data, y_data, p0=coefs_init)
    return coefs

def v_eff_minimize_fit(coefs_init, y_data, mu_V, sigma_V, tau_VN, expansion_point, expansion_norm):

    def v_eff_fit_error(coefs, v_eff, mu_V, sigma_V, tau_VN):
        mu_V, sigma_V, tau_VN = convert_to_arrays(mu_V, sigma_V, tau_VN)
        v_eff_fit = v_eff_function(coefs, mu_V, sigma_V, tau_VN, expansion_point, expansion_norm)
        return np.mean((v_eff - v_eff_fit)**2)
    
    y_data, mu_V, sigma_V, tau_VN = flatten_and_remove_nans(y_data, mu_V, sigma_V, tau_VN)
    coefs_init[:0] = [y_data.mean()] 

    fit = minimize(v_eff_fit_error, 
                   coefs_init,
                   args=(y_data, mu_V, sigma_V, tau_VN),
                   method='Nelder-Mead',
                   tol=1e-20,
                   options={'disp':True,'maxiter':20000})
    return fit.x


