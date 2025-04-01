# !/usr/bin/python
# -*- coding: utf-8 -*-

from utils import Utils, Log, GPUTools
from population import initialize_population
from evaluate import decode, fitnessEvaluate
from evolve import reproduction
from selection import Selection
import copy, os, time
import numpy as np
import configparser


def create_directory():
    dirs = ['./log', './populations', './scripts', './trained_models']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def fitness_evaluate(population, curr_gen):
    filenames = []
    for i, indi in enumerate(population):
        filename = decode(indi, curr_gen, i, is_test=False)
        filenames.append(filename)
    err_set, num_parameters, flops = fitnessEvaluate(filenames, curr_gen, is_test=False)
    fitnessSet = [calc_fitness(err_set[i]) for i in range(len(err_set))]
    return err_set, num_parameters, flops, fitnessSet

def evolve(population, fitnessSet, params):
    offspring = reproduction(population, fitnessSet, params, Log)
    return copy.deepcopy(offspring)

def environment_selection(parentAndPerformance, offspringAndPerformance, params):
    parents, errs_parent, NumParams_parent, flops_parent, fitnessSet_parent = parentAndPerformance
    offspring, errs_offspring,NumParames_offspring,flops_offspring,fitnessSet_offspring = offspringAndPerformance

    err_list = []
    numParams_list = []
    flops_list = []
    fitness_list = []
    indi_list = []
    for i,indi in enumerate(parents):
        indi_list.append(indi)
        err_list.append(errs_parent[i])
        numParams_list.append(NumParams_parent[i])
        flops_list.append(flops_parent[i])
        fitness_list.append(fitnessSet_parent[i])
    for j,indi in enumerate(offspring):
        indi_list.append(indi)
        err_list.append(errs_offspring[j])
        numParams_list.append(NumParames_offspring[j])
        flops_list.append(flops_offspring[j])
        fitness_list.append(fitnessSet_offspring[j])

    # find the largest one's index
    max_index = np.argmax(fitness_list)
    selection = Selection()
    selected_index_list = selection.binary_tournament_selection(fitness_list, k=params['pop_size'])
    if max_index not in selected_index_list:
        first_selectd_v_list = [fitness_list[i] for i in selected_index_list]
        min_idx = np.argmin(first_selectd_v_list)
        selected_index_list[min_idx] = max_index

    next_individuals = [indi_list[i] for i in selected_index_list]
    next_errs = [err_list[i] for i in selected_index_list]
    next_NumParams = [numParams_list[i] for i in selected_index_list]
    next_flops = [flops_list[i] for i in selected_index_list]
    next_fitnessSet = [fitness_list[i] for i in selected_index_list]

    return next_individuals, next_errs, next_NumParams, next_flops, next_fitnessSet


def calc_fitness(err, numParams=None, flops=None):
    return 1-err

def update_best_individual(population, err_set, num_parameters, flops, gbest):
    fitnessSet = [calc_fitness(err_set[i]) for i in range(len(population))]
    if not gbest:
        pbest_individuals = copy.deepcopy(population)
        pbest_errSet = copy.deepcopy(err_set)
        pbest_params = copy.deepcopy(num_parameters)
        pbest_flops = copy.deepcopy(flops)
        gbest_individual, gbest_err, gbest_params, gbest_flops, gbest_fitness = getGbest([pbest_individuals, pbest_errSet, pbest_params, pbest_flops])
    else:
        gbest_individual, gbest_err, gbest_params, gbest_flops = gbest
        gbest_fitness = calc_fitness(gbest_err)
        for i,fitness in enumerate(fitnessSet):
            if fitness >= gbest_fitness:
                gbest_fitness = copy.deepcopy(fitness)
                gbest_individual = copy.deepcopy(population[i])
                gbest_err = copy.deepcopy(err_set[i])
                gbest_params = copy.deepcopy(num_parameters[i])
                gbest_flops = copy.deepcopy(flops[i])
    return [gbest_individual, gbest_err, gbest_params, gbest_flops, gbest_fitness]

def getGbest(pbest):
    pbest_individuals, pbest_errSet, pbest_params, pbest_flops = pbest
    gbest_err = 10e9
    gbest_params = 10e6
    gbest_flops = 10e9
    gbest = None

    gbest_fitness = calc_fitness(gbest_err)
    pbest_fitnessSet = [calc_fitness(pbest_errSet[i]) for i in range(len(pbest_individuals))]

    for i,indi in enumerate(pbest_individuals):
        if pbest_fitnessSet[i] >= gbest_fitness:
            gbest = copy.deepcopy(indi)
            gbest_err = copy.deepcopy(pbest_errSet[i])
            gbest_params = copy.deepcopy(pbest_params[i])
            gbest_flops = copy.deepcopy(pbest_flops[i])
            gbest_fitness = copy.deepcopy(pbest_fitnessSet[i])
    return gbest, gbest_err, gbest_params, gbest_flops, gbest_fitness

def fitness_test(final_individual):
    final_individual = copy.deepcopy(final_individual)
    filename = Utils.generate_pytorch_file(final_individual, -1, -1, is_test=True)
    err_set, num_parameters, flops = fitnessEvaluate([filename], -1, True, [batch_size], [weight_decay])
    fitnessSet = [calc_fitness(err_set[i]) for i in range(len(err_set))]
    return err_set[0], num_parameters[0], flops[0], fitnessSet[0]

def evolveCNN(params):
    gen_no = 0
    Log.info('Initialize...')
    start = time.time()
    population = initialize_population(params)

    Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (gen_no))
    errs_parent, NumParams_parent, flops_parent, fitnessSet_parent = fitness_evaluate(population, gen_no)
    Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (gen_no))

    [gbest_individual, gbest_err, gbest_params, gbest_flops, gbest_fitness] = update_best_individual(population, errs_parent, NumParams_parent, flops_parent, gbest=None)
    Utils.save_population('population', population, errs_parent, NumParams_parent, flops_parent, gen_no)
    Utils.save_population('gbest', [gbest_individual], [gbest_err], [gbest_params], [gbest_flops], gen_no)

    gen_no += 1

    for curr_gen in range(gen_no, params['num_generations']):
        params['gen_no'] = curr_gen

        Log.info('EVOLVE[%d-gen]-Begin evolution' % (curr_gen))
        offspring = evolve(population, fitnessSet_parent, params)
        Log.info('EVOLVE[%d-gen]-Finish evolution' % (curr_gen))

        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (curr_gen))
        errs_offspring, NumParames_offspring, flops_offspring, fitnessSet_offspring = fitness_evaluate(offspring, curr_gen)
        Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (curr_gen))

        population, errs_parent, NumParams_parent, flops_parent, fitnessSet_parent = environment_selection([population, errs_parent, NumParams_parent, flops_parent, fitnessSet_parent], [offspring, errs_offspring, NumParames_offspring, flops_offspring, fitnessSet_offspring], params)
        Log.info('EVOLVE[%d-gen]-Finish the environment selection' % (curr_gen))

        [gbest_individual, gbest_err, gbest_params, gbest_flops, gbest_fitness] = update_best_individual(population, errs_parent, NumParams_parent, flops_parent, gbest=[gbest_individual, gbest_err, gbest_params, gbest_flops])

        Utils.save_population('population', population, errs_parent, NumParams_parent, flops_parent, curr_gen)
        Utils.save_population('gbest', [gbest_individual], [gbest_err], [gbest_params], [gbest_flops], curr_gen)

    end = time.time()
    Log.info('Total Search Time: %.2f seconds' % (end-start))
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    Log.info("%02dh:%02dm:%02ds" % (h, m, s))

    search_time = str("%02dh:%02dm:%02ds" % (h, m, s))
    equipped_gpu_ids, _ = GPUTools._get_equipped_gpu_ids_and_used_gpu_info()
    num_GPUs = len(equipped_gpu_ids)

    [gbest_individual, gbest_err, gbest_params, gbest_flops, gbest_fitness] = update_best_individual(population, errs_parent, NumParams_parent, flops_parent, gbest=None)
    Utils.save_population('gbest', [gbest_individual], [gbest_err], [gbest_params], [gbest_flops], params['num_generations'], gbest_err, search_time+', GPUs:%d'%num_GPUs)

    proxy_err = copy.deepcopy(gbest_err)
    # final training and test on testset
    gbest_err, num_parameters, flops, fitness = fitness_test(gbest_individual)
    Log.info('Error=[%.5f], #parameters=[%d], FLOPs=[%d]' % (gbest_err, num_parameters, flops))
    Utils.save_population('final_gbest', [gbest_individual], [gbest_err], [num_parameters], [flops], -1, proxy_err, search_time+', GPUs:%d'%num_GPUs)


if __name__ == '__main__':
    create_directory()
    params = Utils.get_init_params()
    batch_size = params['batch_size']
    weight_decay = params['weight_decay']

    evolveCNN(params)

