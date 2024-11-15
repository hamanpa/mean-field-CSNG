"""
This script contains the parameters of various cell types.
"""

import json
import fnmatch
from pathlib import Path

PARAM_FILE = Path('/home/pavel/academia/mean-field-CSNG/data/parameters.json')

# Everything is 

# TODO: deal with delays!
# TODO: convert nest <--> pyNN
# TODO: deal with LGN cells

# pynn models here:
# https://pynn.readthedocs.io/en/latest/reference/neuronmodels.html
# EIF_cond_exp_isfa_ista, there are units and parameters



def get_items_recursive(d, key, items=None):
    if items is None:
        items = {}
    for k,v in d.items():
        if k.endswith(key):
            items[k] = v
        elif isinstance(v, dict):
            items = get_items_recursive(v, key, items=items)
    return items

def scrap_csng_model_parameters(param_file):
    params = {}
    with open(param_file, 'r') as f:
        full_params = json.load(f)

    for sheet in full_params['sheets'].values():
        if 'name' not in sheet['params']:
            continue
        sheet_name = sheet['params']['name']
        params[sheet_name] = {
            'model': sheet['params']['cell']['model'],
            'params': sheet['params']['cell']['params'],
        }
        if 'K' in sheet:
            params[sheet_name]['K']= sheet['K']
        conn_key = ''.join(sheet_name.replace('/','').split('_')[:0:-1])
        conn_key += 'Connection'

        params[sheet_name]['conns'] = get_items_recursive(full_params, conn_key)

    neurons = dict()
    for cell_name, neuron in params.items():
        layer = cell_name.replace('/', '').split("_")[2]
        conns = [k for k in neuron['conns'].keys() if k.startswith(layer)]
        neurons[cell_name] = {
            'neuron_params' : neuron['params'],
            'synapse_type': 'tsodyks_synapse',
            'simulation_time': 1000.0, 
            'poisson_input': True,
        }
        for conn in conns:
            source = conn.split(layer)[1].lower()
            neurons[cell_name][f'{source}_syn_params'] = {
                'weight': neuron['conns'][conn]['base_weight'] * 1000,  # nS to pS
                'delay': 1.0,
                **neuron['conns'][conn]['short_term_plasticity'],
            }
            neurons[cell_name][f'{source}_syn_number'] = int(neuron['conns'][conn]['num_samples'])
    return neurons

def load_zerlaut_model_parameters():
    # taken from Zerlaut2018_ModelingMesoscopic
    zerlaut_exc_neuron =  {
        'neuron_params': {
            'v_rest': -65.0,  # Resting membrane potential (mV)
            'v_reset': -65.0,  # Reset potential after spike (mV)
            'tau_refrac': 5.0,  # Refractory period (ms)
            'tau_m': 15.0,  # Membrane time constant (ms), takes as cm/gl
            'cm': 0.150,  # Membrane capacitance (nF)
            'e_rev_E': 0.0,  # Excitatory reversal potential (mV)
            'e_rev_I': -80.0,  # Inhibitory reversal potential (mV)
            'tau_syn_E': 5.0,  # Excitatory synaptic time constant (ms)
            'tau_syn_I': 5.0,  # Inhibitory synaptic time constant (ms)
            'a': 4.0,  # Subthreshold adaptation conductance (nS)
            'b': 0.02,  # Spike-triggered adaptation increment (nA)
            'delta_T': 2.0,  # Slope factor (mV)
            'tau_w': 500.0,  # Adaptation time constant (ms)
            'v_thresh': -50.0  # Spike threshold (mV)
        },
        'exc_syn_params': {
            'weight': 1.0,  # synaptic weight (nS)
            'delay': 1.0,
        },
        'inh_syn_params': {
            'weight': 5.0,  # synaptic weight (nS)
            'delay': 1.0,
        },
        'exc_syn_number': 400,
        'inh_syn_number': 100,
        'synapse_type': 'static_synapse',
        'simulation_time': 1000.0,
        'poisson_input': True
    }

    zerlaut_inh_neuron =  {
        'neuron_params': {
            'v_rest': -65.0,  # Resting membrane potential (mV)
            'v_reset': -65.0,  # Reset potential after spike (mV)
            'tau_refrac': 5.0,  # Refractory period (ms)
            'tau_m': 15.0,  # Membrane time constant (ms), takes as cm/gl
            'cm': 0.150,  # Membrane capacitance (nF)
            'e_rev_E': 0.0,  # Excitatory reversal potential (mV)
            'e_rev_I': -80.0,  # Inhibitory reversal potential (mV)
            'tau_syn_E': 5.0,  # Excitatory synaptic time constant (ms)
            'tau_syn_I': 5.0,  # Inhibitory synaptic time constant (ms)
            'a': 0.0,  # Subthreshold adaptation conductance (nS)
            'b': 0.0,  # Spike-triggered adaptation increment (nA)
            'delta_T': 0.5,  # Slope factor (mV)
            'tau_w': 500.0,  # Adaptation time constant (ms)
            'v_thresh': -50.0  # Spike threshold (mV)
        },
        'exc_syn_params': {
            'weight': 1.0,  # synaptic weight (nS)
            'delay': 1.0,
        },
        'inh_syn_params': {
            'weight': 5.0,  # synaptic weight (nS)
            'delay': 1.0,
        },
        'exc_syn_number': 400,
        'inh_syn_number': 100,
        'synapse_type': 'static_synapse',
        'simulation_time': 1000.0,
        'poisson_input': True
    }
    return {'zerlaut_exc': zerlaut_exc_neuron, 'zerlaut_inh': zerlaut_inh_neuron}

def load_divolo_model_parameters():
    # taken from diVolo2019_BiologicallyRealistic
    divolo_exc_neuron =  {
        'neuron_params': {
            'v_rest': -65.0,  # Resting membrane potential (mV)
            'v_reset': -65.0,  # Reset potential after spike (mV)
            'tau_refrac': 5.0,  # Refractory period (ms)
            'tau_m': 15.0,  # Membrane time constant (ms), takes as cm/gl
            'cm': 0.150,  # Membrane capacitance (nF)
            'e_rev_E': 0.0,  # Excitatory reversal potential (mV)
            'e_rev_I': -80.0,  # Inhibitory reversal potential (mV)
            'tau_syn_E': 5.0,  # Excitatory synaptic time constant (ms)
            'tau_syn_I': 5.0,  # Inhibitory synaptic time constant (ms)
            'a': 4.0,  # Subthreshold adaptation conductance (nS)
            'b': 0.02,  # Spike-triggered adaptation increment (nA)
            'delta_T': 2.0,  # Slope factor (mV)
            'tau_w': 500.0,  # Adaptation time constant (ms)
            'v_thresh': -50.0  # Spike threshold (mV)
        },
        'exc_syn_params': {
            'weight': 1.0,  # synaptic weight (nS)
            'delay': 1.0,
        },
        'inh_syn_params': {
            'weight': 5.0,  # synaptic weight (nS)
            'delay': 1.0,
        },
        'exc_syn_number': 400,
        'inh_syn_number': 100,
        'synapse_type': 'static_synapse',
        'simulation_time': 1000.0,
        'poisson_input': True
    }

    divolo_inh_neuron =  {
        'neuron_params': {
            'v_rest': -65.0,  # Resting membrane potential (mV)
            'v_reset': -65.0,  # Reset potential after spike (mV)
            'tau_refrac': 5.0,  # Refractory period (ms)
            'tau_m': 15.0,  # Membrane time constant (ms), takes as cm/gl
            'cm': 0.150,  # Membrane capacitance (nF)
            'e_rev_E': 0.0,  # Excitatory reversal potential (mV)
            'e_rev_I': -80.0,  # Inhibitory reversal potential (mV)
            'tau_syn_E': 5.0,  # Excitatory synaptic time constant (ms)
            'tau_syn_I': 5.0,  # Inhibitory synaptic time constant (ms)
            'a': 0.0,  # Subthreshold adaptation conductance (nS)
            'b': 0.0,  # Spike-triggered adaptation increment (nA)
            'delta_T': 0.5,  # Slope factor (mV)
            'tau_w': 500.0,  # Adaptation time constant (ms)
            'v_thresh': -50.0  # Spike threshold (mV)
        },
        'exc_syn_params': {
            'weight': 1.0,  # synaptic weight (nS)
            'delay': 1.0,
        },
        'inh_syn_params': {
            'weight': 5.0,  # synaptic weight (nS)
            'delay': 1.0,
        },
        'exc_syn_number': 400,
        'inh_syn_number': 100,
        'synapse_type': 'static_synapse',
        'simulation_time': 1000.0,
        'poisson_input': True
    }
    return {'divolo_exc': divolo_exc_neuron, 'divolo_inh': divolo_inh_neuron}

def load_all_neurons():
    neurons = dict()

    csng_neurons = scrap_csng_model_parameters(PARAM_FILE)
    neurons.update(**csng_neurons)

    zerlaut_neurons = load_zerlaut_model_parameters()
    neurons.update(**zerlaut_neurons)

    return neurons

# TODO: conversion PyNN <--> NEST
def pynn_to_nest(params):
    pass
def nest_to_pynn(params):
    pass

