# !/usr/bin/python
# -*- coding: utf-8 -*-
import configparser
import copy
import logging
import time
import sys
import os
import math
from subprocess import Popen, PIPE
import genotypes
from genotypes import search_space_cifar

kernel_sizes = [3, 5, 7]
class Utils(object):
    @classmethod
    def get_init_params(cls):
        params = {}
        params['pop_size'] = cls.get_params('GA', 'pop_size')
        params['num_generations'] = cls.get_params('GA', 'num_generations')
        params['individual_length'] = cls.get_params('GA', 'individual_length')
        params['modules_length'] = cls.get_params('GA', 'modules_length')
        params['image_channel'] = cls.get_params('DATA', 'image_channel')
        params['min_epoch_eval'] = cls.get_params('NETWORK', 'min_epoch_eval')
        params['epoch_test'] = cls.get_params('NETWORK', 'epoch_test')
        params['batch_size'] = cls.get_params('NETWORK', 'batch_size')
        params['weight_decay'] = float(cls.__read_ini_file('NETWORK', 'weight_decay'))
        params['drop_connect_rate'] = float(cls.__read_ini_file('NETWORK', 'drop_connect_rate'))
        params['crossover_prob'] = float(cls.__read_ini_file('GA', 'crossover_prob'))
        params['mutation_prob'] = float(cls.__read_ini_file('GA', 'mutation_prob'))
        return params

    @classmethod
    def __read_ini_file(cls, section, key):
        config = configparser.ConfigParser()
        config.read('global.ini')
        return config.get(section, key)

    @classmethod
    def get_params(cls, domain, key):
        rs = cls.__read_ini_file(domain, key)
        return int(rs)

    @classmethod
    def save_population(cls, type, population, err_set, num_parameters, flops, gen_no, proxy_err=None, time=None):
        file_name = './populations/' + type + '_%02d.txt' % (gen_no)
        _str = cls.pop2str(population, err_set, num_parameters, flops, proxy_err, time)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def write_to_file(cls, _str, _file):
        f = open(_file, 'w')
        f.write(_str)
        f.flush()
        f.close()

    @classmethod
    def pop2str(cls, population, err_set, num_parameters, flops, proxy_err, time):
        pop_str = []
        for id, individual in enumerate(population):
            _str = []
            _str.append('indi:%02d' % (id))
            _str.append('individual:%s' % (','.join(list(map(str, individual)))))
            _str.append('num_parameters:%d' % (num_parameters[id]))
            _str.append('FLOPs:%d' % (flops[id]))
            if proxy_err:
                _str.append('proxy_err:%.4f' % (proxy_err))
            _str.append('eval_err:%.4f' % (err_set[id]))
            if time:
                _str.append('search time:%s' % (time))

            individual_length = cls.get_params('GA', 'individual_length')
            #individual = [[e, f, se, k]...[e, f, se, k]]
            in_C = 16
            for i, name in enumerate(search_space_cifar['names']):
                expansion_rate = individual[i][0]
                outC = cls._RoundChannels(search_space_cifar[name][1][0]*individual[i][1])
                stride = search_space_cifar[name][5]
                SE = individual[i][2]
                kernels= individual[i][3]
                kernels_combination = []
                for i in range(kernels):
                    kernels_combination.append(kernel_sizes[i])
                _str.append(str(name) + ' - expansion_rate: %d, input: %3d, output: %3d, stride: %d, SE=%f, kernels_combination=[%s]' % (
                expansion_rate, in_C, outC, stride, SE, ", ".join(str(i) for i in kernels_combination)))
                in_C = outC

            indi_str = '\n'.join(_str)
            pop_str.append(indi_str)
            pop_str.append('-' * 100)
        return '\n'.join(pop_str)


    @classmethod
    def generate_forward_list(cls, conv_names):
        forward_list = []
        for i,conv_name in enumerate(conv_names):
            if i == 0:
                forward_string = 'out = ' + conv_name+'(x)'
            else:
                forward_string = 'out = ' + conv_name+'(out, self.drop_connect_rate)'
            forward_list.append(forward_string)

        return forward_list

    @classmethod
    def read_template(cls, test_model = False):
        dataset = str(cls.__read_ini_file('DATA', 'dataset'))
        if test_model:
            _path = './template/' + dataset + '_test.py'
        else:
            _path = './template/' + dataset + '.py'
        part1 = []
        part2 = []
        part3 = []

        f = open(_path)
        f.readline()  # skip this comment
        line = f.readline().rstrip()
        while line.strip() != '#generated_init':
            part1.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part1))

        line = f.readline().rstrip()  # skip the comment '#generated_init'
        while line.strip() != '#generate_forward':
            part2.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part2))

        line = f.readline().rstrip()  # skip the comment '#generate_forward'
        while line.strip() != '"""':
            part3.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part3))
        return part1, part2, part3

    @classmethod
    def _RoundChannels(cls, c, divisor=4, min_value=None):
        if min_value is None:
            min_value = divisor
        new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
        if new_c < 0.9 * c:
            new_c += divisor
        return new_c

    @classmethod
    def generate_pytorch_file(cls, individual, curr_gen, id, is_test, test_model=False):
        dataset = str(cls.__read_ini_file('DATA', 'dataset'))
        img_channel = cls.get_params('DATA', 'image_channel')
        drop_connect_rate = float(cls.__read_ini_file('NETWORK', 'drop_connect_rate'))

        # query convolution unit
        conv_names = []
        conv_list = []

        if dataset=='cifar10' or dataset=='cifar100':
            if is_test:
                scaling_factor = 1.0
                dropRate = drop_connect_rate
                conv_begin1 = 'self.conv_begin1 = ConvBNReLU(%d, %d, kernel_size=3, stride=1)' % (img_channel, cls._RoundChannels(24*individual[0][1]))
                inC = cls._RoundChannels(24*individual[0][1])
            else:
                scaling_factor = 0.25
                dropRate = 0
                conv_begin1 = 'self.conv_begin1 = ConvBNReLU(%d, %d, kernel_size=3, stride=1)' % (img_channel, cls._RoundChannels(8))
                inC = cls._RoundChannels(8)
        elif dataset=='imagenet':
            if is_test:
                scaling_factor = 1.0
                conv_begin1 = 'self.conv_begin1 = ConvBNReLU(%d, %d, kernel_size=3, stride=1)' % (img_channel, cls._RoundChannels(24))
                inC = cls._RoundChannels(24)
            else:
                scaling_factor = 0.2
                conv_begin1 = 'self.conv_begin1 = ConvBNReLU(%d, %d, kernel_size=3, stride=1)' % (img_channel, cls._RoundChannels(8))
                inC = cls._RoundChannels(8)
            dropRate = 0
        conv_list.append(conv_begin1)
        conv_names.append('self.conv_begin1')
        individual_length = cls.get_params('GA', 'individual_length')
        act_func = ["relu", "relu", "hswish"]
        for i, name in enumerate(search_space_cifar['names']):
            #[e, f, se, k]
            stage_idx = int(name.split('_')[1])
            expansion_rate = individual[i][0]
            width_multiplier = individual[i][1]
            outC = cls._RoundChannels(search_space_cifar[name][1][0]*width_multiplier*scaling_factor)
            stride = search_space_cifar[name][5]
            SE = individual[i][2]
            kernels = individual[i][3]
            kernels_combination = []
            MSModule = [0] * len(kernel_sizes)
            for i in range(kernels):
                kernels_combination.append(kernel_sizes[i])
                MSModule[i] = 1
            expand_ksize = search_space_cifar[name][3]
            project_ksize = search_space_cifar[name][4]

            midC = expansion_rate * inC
            conv_name = 'self.' + str(name)
            if not expansion_rate == 0:
                conv = '%s = Block(C_in=%d, C_mid=%d, C_out=%d, expand_ksize="%s", project_ksize="%s", stride=%d, act_func="%s", conn_encoding="%s", attention="%s", drop_connect_rate="%s", is_test=%d)' % (
                    conv_name, inC, midC, outC, " ".join(str(i) for i in expand_ksize),
                    " ".join(str(i) for i in project_ksize), stride, act_func[stage_idx - 1],
                    " ".join(str(i) for i in MSModule), str(SE), str(dropRate), int(is_test))
                inC = outC
                conv_list.append(conv)
                conv_names.append(conv_name)

        # query fully-connect layer, because a global avg_pooling layer is added before the fc layer, so the input size
        # of the fc layer is equal to out channel
        if dataset == 'cifar10':
            factor = 1
            factor1 = 1
            end_channel = inC
            num_class = 10
        elif dataset == 'cifar100':
            factor = 4
            factor1 = 4
            end_channel = inC
            num_class = 100
        elif dataset == 'imagenet':
            factor = 6
            factor1 = 8
            end_channel = inC
            num_class = 1000
        fc_node = factor1 * end_channel
        fc_node1 = factor1 * end_channel

        # add an 1*1 conv to the end of the conv_list, before global avg pooling  2021.03.05
        conv_end = 'self.conv_end = nn.Conv2d(%d, %d, kernel_size=1, stride=1, bias=False)' % (end_channel, fc_node)
        conv_list.append(conv_end)
        bn_end = 'self.bn_end = nn.BatchNorm2d(%d)' % (fc_node)
        conv_list.append(bn_end)
        fully_layer_name = 'self.linear = nn.Linear(%d, %d)' % (fc_node1, num_class)
        drop_conn_rate_name = 'self.drop_connect_rate = %.2f' % (dropRate)

        # generate the forward part
        forward_list = cls.generate_forward_list(conv_names)

        part1, part2, part3 = cls.read_template(test_model)
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('\n        %s' % ('#conv unit'))
        for s in conv_list:
            _str.append('        %s' % (s))
        _str.append('\n        %s' % ('#linear unit'))
        _str.append('        %s' % (fully_layer_name))
        _str.append('        %s' % (drop_conn_rate_name))

        _str.extend(part2)
        for s in forward_list:
            _str.append('        %s' % (s))
        _str.extend(part3)
        # print('\n'.join(_str))
        file_path = './scripts/indi%02d_%02d.py' % (curr_gen, id)
        script_file_handler = open(file_path, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()
        file_name = 'indi%02d_%02d'%(curr_gen, id)
        return file_name


class Log(object):
    _logger = None

    @classmethod
    def __get_logger(cls):
        if Log._logger is None:
            logger = logging.getLogger("MixedScale")
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
            file_handler = logging.FileHandler("main.log")
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
            Log._logger = logger
            return logger
        else:
            return Log._logger

    @classmethod
    def info(cls, _str):
        cls.__get_logger().info(_str)

    @classmethod
    def warn(cls, _str):
        cls.__get_logger().warning(_str)


class GPUTools(object):
    @classmethod
    def _get_equipped_gpu_ids_and_used_gpu_info(cls):
        p = Popen('nvidia-smi', stdout=PIPE)
        output_info = p.stdout.read().decode('UTF-8')
        lines = output_info.split(os.linesep)
        equipped_gpu_ids = []
        for line_info in lines:
            if not line_info.startswith(' '):
                if 'GeForce' in line_info or 'Quadro' in line_info or 'Tesla' in line_info or 'RTX' in line_info or 'A40' in line_info or 'L4' in line_info:
                    equipped_gpu_ids.append(line_info.strip().split(' ', 4)[3])
            else:
                break

        gpu_info_list = []
        for line_no in range(len(lines) - 3, -1, -1):
            if lines[line_no].startswith('|==='):
                break
            else:
                gpu_info_list.append(lines[line_no][1:-1].strip())

        return equipped_gpu_ids, gpu_info_list

    @classmethod
    def get_available_gpu_ids(cls):
        equipped_gpu_ids, gpu_info_list = cls._get_equipped_gpu_ids_and_used_gpu_info()

        used_gpu_ids = []

        for each_used_info in gpu_info_list:
            if 'python' in each_used_info:
                used_gpu_ids.append((each_used_info.strip().split(' ', 1)[0]))

        unused_gpu_ids = []
        for id_ in equipped_gpu_ids:
            if id_ not in used_gpu_ids:
                unused_gpu_ids.append(id_)
        return unused_gpu_ids

    @classmethod
    def detect_available_gpu_id(cls):
        unused_gpu_ids = cls.get_available_gpu_ids()
        if len(unused_gpu_ids) == 0:
            Log.info('GPU_QUERY-No available GPU')
            return None
        else:
            Log.info('GPU_QUERY-Available GPUs are: [%s], choose GPU#%s to use' % (','.join(unused_gpu_ids), unused_gpu_ids[0]))
            return list(map(int, unused_gpu_ids))


    @classmethod
    def all_gpu_available(cls):
        _, gpu_info_list = cls._get_equipped_gpu_ids_and_used_gpu_info()

        used_gpu_ids = []

        for each_used_info in gpu_info_list:
            if 'python' in each_used_info:
                used_gpu_ids.append((each_used_info.strip().split(' ', 1)[0]))
        if len(used_gpu_ids) == 0:
            Log.info('GPU_QUERY-None of the GPU is occupied')
            return True
        else:
            Log.info('GPU_QUERY- GPUs [%s] are occupying' % (','.join(used_gpu_ids)))
            return False


