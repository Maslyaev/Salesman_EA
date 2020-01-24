#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:45:02 2019

@author: mike_ubuntu
"""

import copy
import random
import numpy as np
import matplotlib.pyplot as plt

class Individual:
    def __init__(self, points, track = None):
        if not track:
            self.track = np.array(random.sample(points, len(points)))
        else:
            self.track = np.array(track)
           
    @staticmethod
    def evaluate_fitness(track, distances):
        total_dist = 0
        for i in np.arange(len(track)-1):
            total_dist += distances[track[i] - 1, track[i+1] - 1]
        total_dist += distances[track[-1] - 1, track[0] - 1]    
        f_val = 1./total_dist 
        return f_val
        
    def visualize(self, point_coords):
        fig, ax = plt.subplots()
        ax.plot([point_coords[i-1, 0] for i in self.track], [point_coords[i-1, 1] for i in self.track], color = 'k')
        ax.plot([point_coords[self.track[0]-1, 0], point_coords[self.track[-1]-1, 0]], 
                 [point_coords[self.track[0]-1, 1], point_coords[self.track[-1]-1, 1]], color = 'k')
        ax.scatter(point_coords[:, 0], point_coords[:, 1], c = 'r')
        ax.set_title('Final picture of vertices and edges')
        plt.show()

        
    def mutate_gaussian(self):
#        mutated = []
        track_new = copy.copy(self.track)
        #print('track_new.type:', type(track_new))
        for pair_idx in np.arange(0, min(np.int(np.rint(np.random.normal(loc=len(self.track)/4., scale=len(self.track)/12.))), 
                                            np.int(np.rint(len(self.track)/4.)))):    
        
#    for i in np.arange(indiv_1.track.size):
#        if i >= p_left and i <= p_right:
#            child_1_temp_track.append(indiv_2.track[i])
#            child_2_temp_track.append(indiv_1.track[i])
#        else:
#            child_1_temp_track.append(indiv_2.track[i])
            idx_1, idx_2 = random.sample(range(len(self.track)), k=2)
            #print(idx_1, idx_2)
            temp = track_new[idx_1]; track_new[idx_1] = track_new[idx_2]; track_new[idx_2] = temp
        self.track = np.copy(track_new)

    def mutate_reversal(self):
        #track_new = np.empty(self.track)
        p_left = np.random.randint(low = 1, high = self.track.size - 1) #uniform
        p_right = np.random.randint(low = p_left, high = self.track.size) #uniform
        #print(self.track[:p_left].shape, np.flip(self.track[p_left:p_right+1]).shape, self.track[p_right+1:].shape)
        flipped = np.flip(self.track[p_left:p_right+1])
        track_new = np.concatenate((self.track[:p_left], flipped, self.track[p_right+1:]))
        self.track = track_new 
    
class Population:
    def __init__(self, metaparams = {'poolsize':10, 'crossover_rate':0.5, 'mutation_rate':0.5, 'elite':2}, towns_file = 'Input.txt'):
        self.metaparams = metaparams   
        temp = np.loadtxt('Input.txt', delimiter = ' ', dtype = int)
        self.coords = temp[:, 1:]
        self.points = temp[:, 0]
        self.distances = np.zeros((self.coords.shape[0], self.coords.shape[0]))
        for i in np.arange(self.coords.shape[0]):
            for j in np.arange(self.coords.shape[0]):
                self.distances[i, j] = np.linalg.norm(self.coords[i, :] - self.coords[j, :], ord = 2)
        self.pool = [Individual(points=list(self.points)) for i in np.arange(self.metaparams['poolsize'])]     
        self.history = []
        
    def Evolution(self, iterations = None, fitval_target = None):
        fitness_values = []
        if iterations:
            for idx in np.arange(iterations):
                fitness_values.append(self.Step(idx))
        elif fitval_target:
            current_fitness = 0
            idx = 0 
            while current_fitness < fitval_target:
                current_fitness = self.Step(idx)
                fitness_values.append(current_fitness)
                idx += 1
                
        else:
            raise ValueError('Initialize target fitness value of number of epochs')
        
        self.pool = list(reversed([x for x, _ in sorted(
                list(zip(self.pool, [indiv.evaluate_fitness(indiv.track, self.distances) for indiv in self.pool])), key=lambda pair: pair[1])]))
#        plt.plot(fitness_values)
        self.history.extend(fitness_values)
            
    def Plot_History(self):
        fig, ax = plt.subplots()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1)
        ax.plot(self.history, color = 'k', linewidth=2)
        plt.xlabel('epoch'); plt.ylabel('Fitness Value')
        plt.title('maximum fitness value in the population for epochs')
        plt.show()
    
    def Step(self, iter_idx):
        fval_list = []
        for individual in self.pool:
            individual.fitval = individual.evaluate_fitness(individual.track, self.distances)
            #print(individual.fitval)
            fval_list.append(individual.fitval) #np.ceil(f) // 2 * 2 + 1
        probas = np.array(fval_list)/np.sum(fval_list)
        #print('probas', probas)
        
        # Roulette-wheel crossover
        parents_idxs = np.random.choice(np.arange(len(self.pool)), 
                                        size = int(np.ceil(len(self.pool)*self.metaparams['crossover_rate']) // 2 * 2),
                                        replace = False, p = probas)
        #print(len(parents_idxs), parents_idxs)
        new_pool = []
        for idx in np.arange(len(parents_idxs)/2, dtype = np.int16):
            #print('Crossovering individuals in indexes', parents_idxs[2*idx], 'and ', parents_idxs[2*idx+1])
            temp_indiv_1, temp_indiv_2 = crossover(self.pool[parents_idxs[2*idx]], self.pool[parents_idxs[2*idx + 1]])
            new_pool.extend([temp_indiv_1, temp_indiv_2])
            
        # Selection of "survivors"
        total_pool = new_pool + self.pool
        #print(len(total_pool))
        total_pool_ordered = list(reversed([x for x, _ in sorted(
                list(zip(total_pool, [indiv.evaluate_fitness(indiv.track, self.distances) for indiv in total_pool])), key=lambda pair: pair[1])]))
        #print([indiv.evaluate_fitness(indiv.track, self.distances) for indiv in total_pool_ordered])
        self.pool = total_pool_ordered[:self.metaparams['poolsize']]
        best_fitness = self.pool[0].evaluate_fitness(self.pool[0].track, self.distances)
        print('Iteration', iter_idx, 'fitness value:', best_fitness)
        
        for idx in np.arange(self.metaparams['elite'],self.metaparams['poolsize']):
            if np.random.random() < self.metaparams['mutation_rate']:
                self.pool[idx].mutate_reversal()
            
        return best_fitness
#        self.pool[]
#        fval_list = [individual.fitval]
    

#    def Roulette_wheel_selection(self, )

        
def crossover(indiv_1, indiv_2):
    child_1 = copy.copy(indiv_1)
    child_2 = copy.copy(indiv_2)
    
    p_left = np.random.randint(low = 1, high = child_1.track.size - 1) #uniform
    p_right = np.random.randint(low = p_left, high = child_1.track.size) #uniform
    
    #print(p_left, p_right)
#    child_1_temp_track = []; child_2_temp_track = []
    
    child_1_temp_track = [indiv_1.track[i] for i in np.arange(indiv_1.track.size) if indiv_1.track[i] not in indiv_2.track[p_left:p_right+1]]
    child_2_temp_track = [indiv_2.track[i] for i in np.arange(indiv_2.track.size) if indiv_2.track[i] not in indiv_1.track[p_left:p_right+1]]
    
    child_1.track = np.array(list(child_1_temp_track[:p_left]) + list(indiv_2.track[p_left:p_right+1]) + list(child_1_temp_track[p_left:]))
    child_2.track = np.array(list(child_2_temp_track[:p_left]) + list(indiv_1.track[p_left:p_right+1]) + list(child_2_temp_track[p_left:]))
    
    return child_1, child_2
    
if __name__ == "__main__":
    #epochs = 18000
    fitval_target = 0.00161
    temp = Population()
    temp.Evolution(fitval_target=fitval_target)#epochs)
    temp.pool[0].visualize(temp.coords) # Рисуем путь
    temp.Plot_History() # График фитнес-ф-ции