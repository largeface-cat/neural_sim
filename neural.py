import numpy as np
from numpy.random import randint, rand, shuffle
from scipy.signal import convolve2d
from typing import List, Callable, Tuple, Dict
from collections import defaultdict
import time
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import re
from utils import *


def sig_prob(x):
    '''
    Return 1 with probability x, -1 with probability 1-x.
    '''
    return 2 * (rand() < x) - 1


class Environment:
    def __init__(self, space=(10, 10), physics=[0, 1], chemicals=[2, 3], ax=None):
        '''
        The environment class manages time and the physical / chemical concentrations in the environment.\\
        Parameters:
        space: 2D tuple (x, y)
        physics: list of physical property names
        chemicals: list of chemical names
        '''
        self.time = 0
        self.space = space
        self.chemicals = chemicals
        self.physics = physics
        self.space_dict = {i: np.zeros(space) for i in range(len(physics + chemicals))}
        self.space_dict_empty = deepcopy_dict(self.space_dict, np.zeros_like)
        self.ax = ax

    def legalize(self, exclude=[], eliminate_excess=False):
        '''
        Make sure all the physical and chemical concentrations are non-negative.
        '''
        for name, space in self.space_dict.items():
            if name not in exclude:
                self.space_dict[name] = np.maximum(space, 0)
                if eliminate_excess:
                    self.space_dict[name] -= (self.space_dict[name] - self.space_dict[name].mean()) * (self.space_dict[name] > self.space_dict[name].mean()) / 2
                
    def diffuse(self, name, kernel):
        '''
        Diffuse the chemical concentration in the environment.
        '''
        self.space_dict[name] = convolve2d(self.space_dict[name], kernel, mode='same', boundary='wrap')
            
    def visualize(self):
        '''
        Visualize all the physical and chemical concentrations in the environment.
        '''
        try:
            self.ax = self.ax.flatten()
        except:
            raise ValueError('Please provide a 2D ax.')
        for i, (name, space) in enumerate(self.space_dict.items()):
            self.ax[i].clear()
            # sns.heatmap(space, ax=self.ax[i], annot=SNS_ANNOT)
            self.ax[i].imshow(space)
            self.ax[i].set_title(str(name)+'-'+str(self.time))


class Neuron:
    '''
    Neuron takes upstream coordinates, judges whether it is activated, and updates the downstream coordinates.
    '''
    def __init__(self, upstream:List[np.ndarray], downstream:List[np.ndarray], threshold:int, 
                 intepretation:Dict[int, float], 
                 consumption:Dict[int, float], production:Dict[int, float], no_res_t:int, name='Neuron', env=Environment()):
        self.upstream = upstream
        self.downstream = downstream
        self.threshold = threshold
        self.active_level = 0
        self.intepretation = intepretation
        self.consumption = consumption
        self.production = production
        self.no_res_t = no_res_t
        self.timer = 1e9
        self.name = name
        self.center = np.average(upstream + downstream, axis=0)
        self.env = env
        
        
    def update(self, env_copy=None):
        self.active_level = 0
        for (x, y) in self.upstream:
            for chemical in self.env.chemicals:
                try:self.active_level += self.intepretation[chemical] * self.env.space_dict[chemical][x, y]
                except:pass
                try:env_copy[chemical][x, y] -= self.consumption[chemical] * self.env.space_dict[chemical][x, y]
                except:pass
        if self.active_level >= self.threshold and self.timer >= self.no_res_t:
            self.timer = 0
        for (x, y) in self.downstream:
            for chemical in self.env.chemicals:
                try:env_copy[chemical][x, y] += self.production[chemical] * np.exp(-self.timer)
                except:pass
        self.timer += 1
        return env_copy
        
    def __str__(self):
        return f"{self.name} {self.upstream} {self.downstream} {self.threshold} {self.intepretation} {self.consumption} {self.production} {self.no_res_t}"


class Receptor(Neuron):
    '''
    Receptor is a special type of neuron that does not have upstream neurons.\\
    Its input is special, else from Neurons' chemical concentration, named sensors.\\
    Its output is the same as Neurons'.
    '''
    def __init__(self, sensors:Tuple[int, List[np.ndarray]], downstream:List[np.ndarray], threshold:int, 
                 intepretation:Dict[int,  float], 
                 production:Dict[int, float], no_res_t:int, name='Receptor', env=Environment()):
        super().__init__([], downstream, threshold, intepretation, defaultdict(float), production, no_res_t, name, env)
        self.sensors = sensors
        self.sensor = sensors[0]
        
    def update(self, env_copy=None):
        self.active_level = 0
        for (x, y) in self.sensors[1]:
            try:self.active_level += self.intepretation[self.sensor] * self.env.space_dict[self.sensor][x, y]
            except:pass
        if self.active_level >= self.threshold and self.timer >= self.no_res_t:
            self.timer = 0
        for (x, y) in self.downstream:
            for chemical in self.env.chemicals:
                try:env_copy[chemical][x, y] += self.production[chemical] * np.exp(-self.timer)
                except:pass
        self.timer += 1
        return env_copy
        
    def __str__(self):
        return f"{self.name} {self.sensors} {self.downstream} {self.threshold} {self.intepretation} {self.production} {self.no_res_t}"


class Actuator(Neuron):
    '''
    Actuator is a special type of neuron that does not have downstream neurons.\\
    Its input is the same as Neurons'.\\
    Its output is special, else to Neurons' chemical concentration, named effector.
    '''
    def __init__(self, upstream:List[np.ndarray], effectors:Tuple[int, List[np.ndarray]], threshold:int, 
                 intepretation:Dict[int, float], 
                 consumption:Dict[int, float], no_res_t:int, name='Actuator', env=Environment()):
        super().__init__(upstream, [], threshold, intepretation, consumption, defaultdict(float), no_res_t, name, env)
        self.effectors = effectors
        self.effector = effectors[0]
        # self.effector_duration = effectors[1]
        
    def update(self, env_copy=None):
        self.active_level = 0
        for (x, y) in self.upstream:
            for chemical in self.env.chemicals:
                try:self.active_level += self.intepretation[chemical] * self.env.space_dict[chemical][x, y]
                except:pass
                try:env_copy[chemical][x, y] -= self.consumption[chemical] * self.env.space_dict[chemical][x, y]
                except:pass
        if self.active_level >= self.threshold and self.timer >= self.no_res_t:
            self.timer = 0
        for (x, y) in self.effectors[1]:
            try:self.env.space_dict[self.effector][x, y] += (self.timer<self.no_res_t) * np.exp(-self.timer)
            except:pass
        self.timer += 1
        return env_copy
        
    def __str__(self):
        return f"{self.name} {self.upstream} {self.effectors} {self.threshold} {self.intepretation} {self.consumption} {self.no_res_t}"


class NeuralNetwork:
    '''
    NeuralNetwork has a collection of Neurons.\\
    It has a gene sequence that determines all the neurons in the network, used to simulate evolution.\\
    It updates all the neurons in the network at each time step, according to their activities.
    '''
    def __init__(self, gene:str='', env=Environment(), ax=None):
        self.neurons = []
        self.gene = gene.upper()
        self.l = len(self.gene)
        self.env = env
        self.ax = ax
        self.history = defaultdict(list)
        
    def translate(self):
        '''
        Translate the gene sequence into a collection of Neurons.
        XN: Neuron, XR: Receptor, XA: Actuator, 
        XU: Upstream, XD: Downstream, 
        XT: Threshold, XI: Intepretation, 
        XC: Consumption, XP: Production, XO: No response time.\\
        Hexadecimal numbers following the letters are the parameters.
        '''
        self.l = len(self.gene)
        self.neurons = []
        self.pos_lim = min(self.env.space)
        self.elem_lim = len(self.env.physics + self.env.chemicals) - 1
        self.val_lim = 1e3
        gene_dicts = []
        state_pattern = re.compile(r"X([NRA])")
        hex_pattern = re.compile(r"^[0-9a-fA-F]{8}$")
        result = None
        current_type = None
        current_state = None
        i = 0
        while i < self.l:
            match = state_pattern.match(self.gene, i)
            if match:
                if result:
                    gene_dicts.append(result)
                current_state = match.group(1)
                current_type = None
                result = {**{"N": False, "R": False, "A": False}, **{key: [] for key in "UDTICPO"}}
                result[current_state] = True
                i += 2
                continue

            if current_state:
                valid_types = "UDTICPO"
                if current_state == "R":
                    valid_types = valid_types.replace("C", "")
                elif current_state == "A":
                    valid_types = valid_types.replace("P", "")
                valid_type_pattern = re.compile(f"X([{valid_types}])")
                match_type = valid_type_pattern.match(self.gene, i)
                if i < self.l and match_type:
                    current_type = match_type.group(1)
                    i += 2
                    continue
                if current_type:
                    if i + 8 <= self.l:
                        potential_hex = self.gene[i:i + 8]
                        if hex_pattern.match(potential_hex):
                            result[current_type].append(potential_hex)
                            i += 8
                            if i >= self.l and result:
                                gene_dicts.append(result)
                            continue
            i += 1
        for gene_dict in gene_dicts:
            gene_dict['T'] = hex2int(lim=(-self.val_lim, self.val_lim), s='00000000' if not gene_dict['T'] else gene_dict['T'][-1])
            gene_dict['I'] = {hex2int(lim=(0, self.elem_lim), s=gene_dict['I'][i*2]): hex2float(lim=(-self.val_lim, self.val_lim), s=gene_dict['I'][i*2+1]) for i in range(0, len(gene_dict['I'])//2)}
            gene_dict['C'] = {hex2int(lim=(0, self.elem_lim), s=gene_dict['C'][i*2]): hex2float(lim=(-self.val_lim, self.val_lim), s=gene_dict['C'][i*2+1]) for i in range(0, len(gene_dict['C'])//2)}
            gene_dict['P'] = {hex2int(lim=(0, self.elem_lim), s=gene_dict['P'][i*2]): hex2float(lim=(-self.val_lim, self.val_lim), s=gene_dict['P'][i*2+1]) for i in range(0, len(gene_dict['P'])//2)}
            gene_dict['O'] = hex2int(lim=(-self.val_lim, self.val_lim), s='00000000' if not gene_dict['O'] else gene_dict['O'][-1])
            if gene_dict['N']:
                gene_dict['U'] = [np.array([hex2int(lim=(0, self.pos_lim), s=gene_dict['U'][i*2]), hex2int(lim=(0, self.pos_lim), s=gene_dict['U'][i*2+1])]) for i in range(0, len(gene_dict['U'])//2)]
                gene_dict['D'] = [np.array([hex2int(lim=(0, self.pos_lim), s=gene_dict['D'][i*2]), hex2int(lim=(0, self.pos_lim), s=gene_dict['D'][i*2+1])]) for i in range(0, len(gene_dict['D'])//2)]
                self.neurons.append(Neuron(gene_dict['U'], gene_dict['D'], gene_dict['T'], gene_dict['I'], gene_dict['C'], gene_dict['P'], gene_dict['O'], 'Neuron'+str(len(self.neurons)), self.env))
            elif gene_dict['R']:
                gene_dict['U'] = (hex2int(lim=(0, self.elem_lim), s=gene_dict['U'][0]), [np.array([hex2int(lim=(0, self.pos_lim), s=gene_dict['U'][i*2+1]), hex2int(lim=(0, self.pos_lim), s=gene_dict['U'][i*2+2])]) for i in range(0, (len(gene_dict['U'])-1)//2)])
                gene_dict['D'] = [np.array([hex2int(lim=(0, self.pos_lim), s=gene_dict['D'][i*2]), hex2int(lim=(0, self.pos_lim), s=gene_dict['D'][i*2+1])]) for i in range(0, len(gene_dict['D'])//2)]
                self.neurons.append(Receptor(gene_dict['U'], gene_dict['D'], gene_dict['T'], gene_dict['I'], gene_dict['P'], gene_dict['O'], 'Receptor'+str(len(self.neurons)), self.env))
            elif gene_dict['A']:
                gene_dict['U'] = [np.array([hex2int(lim=(0, self.pos_lim), s=gene_dict['U'][i*2]), hex2int(lim=(0, self.pos_lim), s=gene_dict['U'][i*2+1])]) for i in range(0, len(gene_dict['U'])//2)]
                gene_dict['D'] = (hex2int(lim=(0, self.elem_lim), s=gene_dict['D'][0]), [np.array([hex2int(lim=(0, self.pos_lim), s=gene_dict['D'][i*2+1]), hex2int(lim=(0, self.pos_lim), s=gene_dict['D'][i*2+2])]) for i in range(0, (len(gene_dict['D'])-1)//2)])
                self.neurons.append(Actuator(gene_dict['U'], gene_dict['D'], gene_dict['T'], gene_dict['I'], gene_dict['C'], gene_dict['O'], 'Actuator'+str(len(self.neurons)), self.env))
            else:
                continue
    
    def reverse_translate(self, neurons:List[Neuron]):
        '''
        Reverse translate the collection of Neurons into a gene sequence.
        '''
        gene = ''
        for neuron in neurons:
            if isinstance(neuron, Neuron) and not isinstance(neuron, Receptor) and not isinstance(neuron, Actuator):
                gene += 'XNXU'
                for coord in neuron.upstream:
                    gene += int2hex(coord[0]) + int2hex(coord[1])
                gene += 'XD'
                for coord in neuron.downstream:
                    gene += int2hex(coord[0]) + int2hex(coord[1])
                gene += 'XT' + int2hex(neuron.threshold) + 'XI'
                for key, value in neuron.intepretation.items():
                    gene += int2hex(key) + float2hex(value)
                gene += 'XC'
                for key, value in neuron.consumption.items():
                    gene += int2hex(key) + float2hex(value)
                gene += 'XP'
                for key, value in neuron.production.items():
                    gene += int2hex(key) + float2hex(value)
                gene += 'XO' + int2hex(neuron.no_res_t)
            elif isinstance(neuron, Receptor):
                gene += 'XRXU'
                gene += int2hex(neuron.sensors[0])
                for coord in neuron.sensors[1]:
                    gene += int2hex(coord[0]) + int2hex(coord[1])
                gene += 'XD'
                for coord in neuron.downstream:
                    gene += int2hex(coord[0]) + int2hex(coord[1])
                gene += 'XT' + int2hex(neuron.threshold) + 'XI'
                for key, value in neuron.intepretation.items():
                    gene += int2hex(key) + float2hex(value)
                gene += 'XP'
                for key, value in neuron.production.items():
                    gene += int2hex(key) + float2hex(value)
                gene += 'XO' + int2hex(neuron.no_res_t)
            elif isinstance(neuron, Actuator):
                gene += 'XAXU'
                for coord in neuron.upstream:
                    gene += int2hex(coord[0]) + int2hex(coord[1])
                gene += 'XD'
                gene += int2hex(neuron.effectors[0])
                for coord in neuron.effectors[1]:
                    gene += int2hex(coord[0]) + int2hex(coord[1])
                gene += 'XT' + int2hex(neuron.threshold) + 'XI'
                for key, value in neuron.intepretation.items():
                    gene += int2hex(key) + float2hex(value)
                gene += 'XC'
                for key, value in neuron.consumption.items():
                    gene += int2hex(key) + float2hex(value)
                gene += 'XO' + int2hex(neuron.no_res_t)
        self.gene = gene
        self.l = len(self.gene)
        
    def update(self):
        env_copy = deepcopy_dict(self.env.space_dict_empty, np.zeros_like)
        neuron_active_levels = []
        actuator_active_levels = []
        receptor_active_levels = []
        for neuron in self.neurons:
            neuron.update(env_copy)
            if isinstance(neuron, Neuron) and not isinstance(neuron, Receptor) and not isinstance(neuron, Actuator):
                neuron_active_levels.append(neuron.active_level)
            elif isinstance(neuron, Actuator):
                actuator_active_levels.append(neuron.active_level)
            elif isinstance(neuron, Receptor):
                receptor_active_levels.append(neuron.active_level)
        self.history['neuron_active_levels'].append(avg(neuron_active_levels))
        self.history['actuator_active_levels'].append(avg(actuator_active_levels))
        self.history['receptor_active_levels'].append(avg(receptor_active_levels))
        return env_copy
    
    def visualize(self):
        try:
            self.ax = self.ax.flatten()
        except:
            raise ValueError('Please provide a 2D ax.')
        for i, key in enumerate(self.history.keys()):
            self.ax[i].clear()
            self.ax[i].plot(self.history[key][-20:])
            self.ax[i].set_title(key)


class Creature:
    def __init__(self, env:Environment, envind:list[int], sigind:list[int], c_pos:np.ndarray, nn:NeuralNetwork, muscles:Tuple[np.ndarray, np.ndarray, float]):
        self.env = env
        self.envind = envind
        self.sigind = sigind
        self.c_pos = c_pos
        self.nn = nn
        self.muscles = muscles

    def update(self):
        momentum = sum(self.muscles[1] * (self.env.space_dict[self.sigind][self.muscles[0]] > self.muscles[2]))
        self.c_pos