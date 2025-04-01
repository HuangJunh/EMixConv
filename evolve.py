# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import copy
from genotypes import search_space_cifar

def reproduction(population, fitnessSet, params, _log):
    layer_number = params['individual_length']
    modules_length = params['modules_length']
    crossover_prob = params['crossover_prob']
    mutation_prob = params['mutation_prob']

    crossover = Crossover(population, fitnessSet, crossover_prob, _log)
    offspring = crossover.do_crossover_Structure()

    mutation = Mutation(offspring, mutation_prob, _log)
    offspring = mutation.do_mutation_Structure()

    return offspring

def adjustIndi(population, params):
    #make sure the generated individuals are valid, the encoding for each module contains at least a 1
    modules_length = params['modules_length']
    module_length = 3
    num_modules = modules_length // module_length
    new_pop = []
    for individual in population:
        MSModules_code = individual
        MSModules = [MSModules_code[i * module_length:(i + 1) * module_length] for i in range(num_modules)]
        for j, module in enumerate(MSModules):
            if sum(module) == 0:
                # if the updated module has no operation activated, let the first dimension of the module be 1 (connect to 3*3 conv op)
                individual[j * module_length] = 1
        new_pop.append(individual)
    return new_pop


class Crossover(object):
    def __init__(self, individuals, fitnessSet, prob_, _log):
        self.individuals = individuals
        self.prob = prob_
        self.log = _log
        self.fitnessSet = fitnessSet

    def _choose_one_parent(self):
        count_ = len(self.individuals)
        idx1 = np.random.randint(0, count_)
        idx2 = np.random.randint(0, count_)
        while idx2 == idx1:
            idx2 = np.random.randint(0, count_)

        if self.fitnessSet[idx1] >= self.fitnessSet[idx2]:
            return idx1
        else:
            return idx2
    """
    binary tournament selection
    """
    def _choose_two_diff_parents(self):
        idx1 = self._choose_one_parent()
        idx2 = self._choose_one_parent()
        while idx2 == idx1:
            idx2 = self._choose_one_parent()

        assert idx1 < len(self.individuals)
        assert idx2 < len(self.individuals)
        return idx1, idx2

    def do_crossover_Structure(self):
        _stat_param = {'offspring_new': 0, 'offspring_from_parent': 0}
        new_offspring_list = []
        for _ in range(len(self.individuals) // 2):
            index1, index2 = self._choose_two_diff_parents()
            parent1, parent2 = copy.deepcopy(self.individuals[index1]), copy.deepcopy(self.individuals[index2])
            parent1_Structure, parent2_Structure = parent1, parent2
            p_ = np.random.random()

            if p_ < self.prob:
                _stat_param['offspring_new'] += 2
                """
                exchange their units from these parent individuals, the exchanged units must satisfy
                --- if their is no change after this crossover, keep the original acc -- a mutation should be given [to do---]
                """
                # single-point crossover
                offspring1_Structure = []
                offspring2_Structure = []
                num_genes = len(parent1)
                crossPoint = np.random.randint(0, num_genes)
                offspring1_Structure.extend(parent1_Structure[0:crossPoint])
                offspring1_Structure.extend(parent2_Structure[crossPoint:num_genes])
                offspring2_Structure.extend(parent2_Structure[0:crossPoint])
                offspring2_Structure.extend(parent1_Structure[crossPoint:num_genes])

                parent1, parent2 = offspring1_Structure, offspring2_Structure
                self.log.info('Performing single-point crossover for structure')
                new_offspring_list.append(parent1)
                new_offspring_list.append(parent2)
            else:
                _stat_param['offspring_from_parent'] += 2
                new_offspring_list.append(parent1)
                new_offspring_list.append(parent2)

        self.log.info('CROSSOVER_Structure-%d offspring are generated, new:%d, others:%d' % (
        len(new_offspring_list), _stat_param['offspring_new'], _stat_param['offspring_from_parent']))
        return new_offspring_list




class Mutation(object):

    def __init__(self, individuals, prob_, _log):
        self.individuals = individuals
        self.prob = prob_
        self.log = _log

    def do_mutation_Structure(self):
        _stat_param = {'offspring_new': 0, 'offspring_from_parent': 0}

        offspring = []
        for indi in self.individuals:
            names = search_space_cifar['names']
            is_mutated = False
            for i, name in enumerate(names):
                p_ = np.random.random()
                if p_ < self.prob:
                    mutatePoint = i
                    is_mutated = True
                    indi[mutatePoint][0] = int(np.random.choice(search_space_cifar[name][0]))
                    indi[mutatePoint][1] = float(np.random.choice(search_space_cifar[name][2]))
                    indi[mutatePoint][2] = float(np.random.choice(search_space_cifar[name][6]))
                    indi[mutatePoint][3] = int(np.random.choice(search_space_cifar[name][7]))

            if is_mutated:
                _stat_param['offspring_new'] += 1
            else:
                _stat_param['offspring_from_parent'] += 1
            offspring.append(indi)
        self.log.info('MUTATION_Structure-mutated individuals:%d, no_changes:%d' % (
        _stat_param['offspring_new'], _stat_param['offspring_from_parent']))

        return offspring



