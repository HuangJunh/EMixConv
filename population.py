# !/usr/bin/python
# -*- coding: utf-8 -*-

import copy
import numpy as np
from genotypes import search_space_cifar

def initialize_population(params):
    pop_size = params['pop_size']
    individual_length = params['individual_length']
    population = []
    names = search_space_cifar['names']
    assert len(names) == individual_length
    for _ in range(pop_size):
        structure = []
        p_len = np.random.random()
        for i, name in enumerate(names):
            if 0 in search_space_cifar[name][0]:
                p_ = np.random.random()
                if p_ > p_len:
                    e = 0
                else:
                    e_options = copy.deepcopy(search_space_cifar[name][0])
                    e_options.remove(0)
                    e = int(np.random.choice(e_options))
            else:
                e = int(np.random.choice(search_space_cifar[name][0]))

            f = float(np.random.choice(search_space_cifar[name][2]))
            se = float(np.random.choice(search_space_cifar[name][6]))
            k = int(np.random.choice(search_space_cifar[name][7]))
            structure.append([e, f, se, k])

        population.append(structure)
    return population

def test_population():
    params = {}
    params['pop_size'] = 20
    params['individual_length'] = 30
    params['image_channel'] = 3

    pop = initialize_population(params)
    print(pop)
    print(pop[0])


if __name__ == '__main__':
    test_population()

