import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                           recall_score, roc_auc_score, confusion_matrix)
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, MessagePassing
from torch_geometric.utils import add_self_loops
from sklearn.metrics import pairwise_distances
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool
import os
from tabulate import tabulate
import time
from datetime import timedelta

class ACO_FeatureSelector:
    def __init__(self, n_features, n_ants=20, max_iter=100,
                 evaporation=0.5, alpha=1, beta=2, q=1):
        self.n_features = n_features
        self.n_ants = n_ants
        self.max_iter = max_iter
        self.evaporation = evaporation
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.pheromone = np.ones(n_features) * 0.1

    def run(self, fitness_func):
        best_solution = None
        best_fitness = -np.inf

        for _ in range(self.max_iter):
            solutions = []
            for _ in range(self.n_ants):
                prob = self.pheromone ** self.alpha
                prob /= prob.sum()
                solution = np.random.rand(self.n_features) < prob
                solutions.append(solution)

            fitness = np.array([fitness_func(sol) for sol in solutions])
            best_idx = np.argmax(fitness)

            if fitness[best_idx] > best_fitness:
                best_fitness = fitness[best_idx]
                best_solution = solutions[best_idx]

            self.pheromone *= (1 - self.evaporation)
            self.pheromone += self.q * fitness[best_idx] * best_solution

        return best_solution, best_fitness

class HarmonySearchOptimizer:
    def __init__(self, fitness_func, n_features,
                 hm_size=50, hmcr=0.95, par=0.3, bw=0.2, max_iter=100):
       
        self.fitness_func = fitness_func
        self.n_features = n_features
        self.hm_size = hm_size
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.max_iter = max_iter

        # Initialize Harmony Memory
        self.harmony_memory = np.random.rand(hm_size, n_features)
        self.fitness = np.array([fitness_func(sol > 0.5) for sol in self.harmony_memory])

    def run(self):
        for _ in range(self.max_iter):
            # Generate new harmony
            new_harmony = np.zeros(self.n_features)
            for i in range(self.n_features):
                if np.random.rand() < self.hmcr:
                    # Select from memory
                    new_harmony[i] = self.harmony_memory[np.random.randint(self.hm_size), i]
                    # Pitch adjustment
                    if np.random.rand() < self.par:
                        new_harmony[i] += self.bw * np.random.uniform(-1, 1)
                        new_harmony[i] = np.clip(new_harmony[i], 0, 1)
                else:
                    # Random selection
                    new_harmony[i] = np.random.rand()

            # Evaluate new harmony
            new_fitness = self.fitness_func(new_harmony > 0.5)

            # Update harmony memory
            worst_idx = np.argmin(self.fitness)
            if new_fitness > self.fitness[worst_idx]:
                self.harmony_memory[worst_idx] = new_harmony
                self.fitness[worst_idx] = new_fitness

        # Return best solution
        best_idx = np.argmax(self.fitness)
        return (self.harmony_memory[best_idx] > 0.5).astype(int), self.fitness[best_idx]

class GA_Optimizer:
    def __init__(self, fitness_func, n_features,
                 pop_size=50, crossover_rate=0.8,
                 mutation_rate=0.1, max_iter=100):
        
        self.fitness_func = fitness_func
        self.n_features = n_features
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_iter = max_iter

        # Initialize population
        self.population = np.random.rand(pop_size, n_features) > 0.5
        self.fitness = np.array([fitness_func(ind) for ind in self.population])

    def run(self):
        for _ in range(self.max_iter):
            # Selection (Tournament selection)
            parents = []
            for _ in range(self.pop_size):
                idx = np.random.choice(self.pop_size, size=2, replace=False)
                parents.append(self.population[idx[np.argmax(self.fitness[idx])]])

            # Crossover (Uniform crossover)
            offspring = []
            for i in range(0, self.pop_size, 2):
                parent1, parent2 = parents[i], parents[i+1]
                if np.random.rand() < self.crossover_rate:
                    mask = np.random.rand(self.n_features) > 0.5
                    child1 = np.where(mask, parent1, parent2)
                    child2 = np.where(mask, parent2, parent1)
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parent1, parent2])

            # Mutation (Bit flip)
            for i in range(len(offspring)):
                if np.random.rand() < self.mutation_rate:
                    mutation_points = np.random.rand(self.n_features) < (1/self.n_features)
                    offspring[i] = np.logical_xor(offspring[i], mutation_points)

            # Evaluate offspring
            offspring_fitness = np.array([self.fitness_func(ind) for ind in offspring])

            # Replacement (Elitism)
            combined_pop = np.vstack([self.population, offspring])
            combined_fit = np.concatenate([self.fitness, offspring_fitness])
            best_idx = np.argsort(combined_fit)[-self.pop_size:]
            self.population = combined_pop[best_idx]
            self.fitness = combined_fit[best_idx]

        # Return best solution
        best_idx = np.argmax(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

class PSO_Optimizer:
    def __init__(self, fitness_func, n_features,
                 swarm_size=50, w=0.8, c1=1.5, c2=1.5, max_iter=100):
        
        self.fitness_func = fitness_func
        self.n_features = n_features
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter

        # Initialize swarm
        self.positions = np.random.rand(swarm_size, n_features)
        self.velocities = np.random.uniform(-1, 1, (swarm_size, n_features))
        self.pbest_pos = self.positions.copy()
        self.pbest_fit = np.array([fitness_func(pos > 0.5) for pos in self.positions])
        self.gbest_pos = self.pbest_pos[np.argmax(self.pbest_fit)]
        self.gbest_fit = np.max(self.pbest_fit)

    def run(self):
        for _ in range(self.max_iter):
            # Update velocities
            r1, r2 = np.random.rand(2)
            cognitive = self.c1 * r1 * (self.pbest_pos - self.positions)
            social = self.c2 * r2 * (self.gbest_pos - self.positions)
            self.velocities = self.w * self.velocities + cognitive + social

            # Update positions
            self.positions = self.positions + self.velocities
            self.positions = np.clip(self.positions, 0, 1)

            # Evaluate current positions
            current_fit = np.array([self.fitness_func(pos > 0.5) for pos in self.positions])

            # Update personal best
            improved_idx = current_fit > self.pbest_fit
            self.pbest_pos[improved_idx] = self.positions[improved_idx]
            self.pbest_fit[improved_idx] = current_fit[improved_idx]

            # Update global best
            if np.max(current_fit) > self.gbest_fit:
                self.gbest_fit = np.max(current_fit)
                self.gbest_pos = self.positions[np.argmax(current_fit)]

        # Return best solution
        return (self.gbest_pos > 0.5).astype(int), self.gbest_fit

class BinaryGWO_Optimizer:
    def __init__(self, fitness_func, n_features,
                 population_size=50, max_iter=100):
        
        self.fitness_func = fitness_func
        self.n_features = n_features
        self.population_size = population_size
        self.max_iter = max_iter
        
    def _sigmoid(self, x):
        """Sigmoid function for binary conversion"""
        return 1 / (1 + np.exp(-x))
        
    def run(self):
        # Initialize population
        population = np.random.rand(self.population_size, self.n_features) > 0.5
        fitness = np.array([self.fitness_func(ind) for ind in population])
        
        # Initialize alpha, beta, delta
        alpha_pos = population[np.argmax(fitness)].copy()
        alpha_score = np.max(fitness)
        beta_pos = population[np.argsort(fitness)[-2]].copy()
        beta_score = fitness[np.argsort(fitness)[-2]]
        delta_pos = population[np.argsort(fitness)[-3]].copy()
        delta_score = fitness[np.argsort(fitness)[-3]]
        
        for iter in range(self.max_iter):
            a = 2 - iter * (2 / self.max_iter)  # a decreases linearly from 2 to 0
            
            for i in range(self.population_size):
                # Update positions
                r1 = np.random.rand(self.n_features)
                r2 = np.random.rand(self.n_features)
                
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                
                D_alpha = np.abs(C1 * alpha_pos - population[i])
                X1 = alpha_pos - A1 * D_alpha
                
                r1 = np.random.rand(self.n_features)
                r2 = np.random.rand(self.n_features)
                
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                
                D_beta = np.abs(C2 * beta_pos - population[i])
                X2 = beta_pos - A2 * D_beta
                
                r1 = np.random.rand(self.n_features)
                r2 = np.random.rand(self.n_features)
                
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                
                D_delta = np.abs(C3 * delta_pos - population[i])
                X3 = delta_pos - A3 * D_delta
                
                # Binary conversion using sigmoid
                new_pos = self._sigmoid((X1 + X2 + X3) / 3)
                population[i] = np.random.rand(self.n_features) < new_pos
                
                # Evaluate new solution
                fitness[i] = self.fitness_func(population[i])
                
                # Update alpha, beta, delta
                if fitness[i] > alpha_score:
                    alpha_score = fitness[i]
                    alpha_pos = population[i].copy()
                elif fitness[i] > beta_score:
                    beta_score = fitness[i]
                    beta_pos = population[i].copy()
                elif fitness[i] > delta_score:
                    delta_score = fitness[i]
                    delta_pos = population[i].copy()
                    
        return alpha_pos, alpha_score

class BinaryCS_Optimizer:
    def __init__(self, fitness_func, n_features,
                 population_size=50, pa=0.25, max_iter=100):
       
        self.fitness_func = fitness_func
        self.n_features = n_features
        self.population_size = population_size
        self.pa = pa
        self.max_iter = max_iter
        
    def _levy_flight(self, size):
        """Generate step size using Levy flight"""
        beta = 1.5
        # Calculate sigma for Levy distribution
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / (np.abs(v) ** (1 / beta))
        return step
        
    def _sigmoid(self, x):
        """Sigmoid function for binary conversion"""
        return 1 / (1 + np.exp(-x))
        
    def run(self):
        # Initialize population
        population = np.random.rand(self.population_size, self.n_features) > 0.5
        fitness = np.array([self.fitness_func(ind) for ind in population])
        
        best_solution = population[np.argmax(fitness)].copy()
        best_fitness = np.max(fitness)
        
        for iter in range(self.max_iter):
            # Generate new solutions via Levy flights
            for i in range(self.population_size):
                step_size = 0.01 * self._levy_flight(self.n_features)
                new_solution = population[i] + step_size * np.random.randn(self.n_features)
                
                # Binary conversion using sigmoid
                new_solution = np.random.rand(self.n_features) < self._sigmoid(new_solution)
                new_fitness = self.fitness_func(new_solution)
                
                # Select a random nest to replace
                j = np.random.randint(0, self.population_size)
                if new_fitness > fitness[j]:
                    population[j] = new_solution
                    fitness[j] = new_fitness
                    
                # Update best solution
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_solution = new_solution.copy()
            
            # Abandon worse nests and build new ones
            worst_idx = np.argsort(fitness)[:int(self.pa * self.population_size)]
            for idx in worst_idx:
                population[idx] = np.random.rand(self.n_features) > 0.5
                fitness[idx] = self.fitness_func(population[idx])
                
                # Update best solution if needed
                if fitness[idx] > best_fitness:
                    best_fitness = fitness[idx]
                    best_solution = population[idx].copy()
                    
        return best_solution, best_fitness
