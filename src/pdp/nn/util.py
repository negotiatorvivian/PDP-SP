# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

# util.py : Defines the utility functionalities for the PDP framework.
import os
import subprocess
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MessageAggregator(nn.Module):
    "Implements a deep set function for message aggregation at variable and function nodes."

    def __init__(self, device, input_dimension, output_dimension, mem_hidden_dimension,
                 mem_agg_hidden_dimension, agg_hidden_dimension, feature_dimension, include_self_message):

        super(MessageAggregator, self).__init__()
        self._device = device
        self._include_self_message = include_self_message
        self._module_list = nn.ModuleList()

        if mem_hidden_dimension > 0 and mem_agg_hidden_dimension > 0:

            self._W1_m = nn.Linear(
                input_dimension, mem_hidden_dimension, bias=True)  # .to(self._device)

            self._W2_m = nn.Linear(
                mem_hidden_dimension, mem_agg_hidden_dimension, bias=False)  # .to(self._device)

            self._module_list.append(self._W1_m)
            self._module_list.append(self._W2_m)

        if agg_hidden_dimension > 0 and mem_agg_hidden_dimension > 0:

            if mem_hidden_dimension <= 0:
                mem_agg_hidden_dimension = input_dimension

            self._W1_a = nn.Linear(
                mem_agg_hidden_dimension + feature_dimension, agg_hidden_dimension, bias=True)  # .to(self._device)

            self._W2_a = nn.Linear(
                agg_hidden_dimension, output_dimension, bias=False)  # .to(self._device)

            self._module_list.append(self._W1_a)
            self._module_list.append(self._W2_a)

        self._agg_hidden_dimension = agg_hidden_dimension
        self._mem_hidden_dimension = mem_hidden_dimension
        self._mem_agg_hidden_dimension = mem_agg_hidden_dimension

    def forward(self, state, feature, mask, mask_transpose, edge_mask=None, var_permutation=None):

        # Apply the pre-aggregation transform
        loss = None
        var_permutation = None
        if self._mem_hidden_dimension > 0 and self._mem_agg_hidden_dimension > 0:
            state = F.logsigmoid(self._W2_m(F.logsigmoid(self._W1_m(state))))
            if var_permutation is not None:
                var_permutation = torch.cat((var_permutation, feature), 1) if edge_mask is None else \
                    torch.cat((var_permutation, edge_mask), 1)
                permute_state = F.logsigmoid(self._W2_m(F.logsigmoid(self._W1_m(var_permutation))))
                permute_state_ = F.softmax(torch.mm(permute_state, torch.t(state)), dim = 0)
                var_degree = torch.sum(torch.abs(permute_state_), dim = 1)
                # loss.append(var_degree)

        if edge_mask is not None:
            state = state * edge_mask

        if mask is not None:
            aggregated_state = torch.mm(mask, state)
        else:
            aggregated_state = state

        if not self._include_self_message:
            aggregated_state = torch.mm(mask_transpose, aggregated_state)

            if edge_mask is not None:
                aggregated_state = aggregated_state - state * edge_mask
                if var_permutation is not None:
                    var_permutation = torch.cat((var_permutation, edge_mask), 1)
            else:
                aggregated_state = aggregated_state - state

        if feature is not None:
            aggregated_state = torch.cat((aggregated_state, feature), 1)
            if var_permutation is not None and edge_mask is None:
                var_permutation = torch.cat((var_permutation, feature), 1)

        # Apply the post-aggregation transform
        if self._agg_hidden_dimension > 0 and self._mem_agg_hidden_dimension > 0:
            aggregated_state = F.logsigmoid(self._W2_a(F.logsigmoid(self._W1_a(aggregated_state))))
            if var_permutation is not None:
                permute_state = F.logsigmoid(self._W2_a(F.logsigmoid(self._W1_a(var_permutation))))
                permute_state_ = F.softmax(torch.mm(permute_state, torch.t(aggregated_state)), dim = 0)
                func_degree = torch.sum(torch.abs(permute_state_), dim = 1)
                loss = func_degree + var_degree

        return aggregated_state, loss


###############################################################

# class MaxpoolAggregator(nn.Module):
#     def __init__(self, device, input_dimension, output_dimension, dropout=0., bias=False, act=nn.relu,**kwags):
#         super(MaxpoolAggregator, self).__init__()
#         self._device = device
#         self.dropout = dropout
#         self.bias = bias
#         self.act = act
#         self._module_list = nn.ModuleList()
###############################################################
class MultiLayerPerceptron(nn.Module):
    "Implements a standard fully-connected, multi-layer perceptron."

    def __init__(self, device, layer_dims):

        super(MultiLayerPerceptron, self).__init__()
        self._device = device
        self._module_list = nn.ModuleList()
        self._layer_num = len(layer_dims) - 1

        self._inner_layers = []
        for i in range(self._layer_num - 1):
            self._inner_layers += [nn.Linear(layer_dims[i], layer_dims[i + 1])]
            self._module_list.append(self._inner_layers[i])

        self._output_layer = nn.Linear(layer_dims[self._layer_num - 1], layer_dims[self._layer_num], bias=False)
        self._module_list.append(self._output_layer)

    def forward(self, inp):
        x = inp

        for layer in self._inner_layers:
            x = F.relu(layer(x))

        return F.sigmoid(self._output_layer(x))


##########################################################################################################################


class SatLossEvaluator(nn.Module):
    "Implements a module to calculate the energy (i.e. the loss) for the current prediction."

    def __init__(self, alpha, device):
        super(SatLossEvaluator, self).__init__()
        self._alpha = alpha
        self._device = device

    @staticmethod
    def safe_log(x, eps):
        # x = torch.tensor(torch.clamp(x, 0), requires_grad=True)
        x = torch.clamp(x, 0).clone().detach().requires_grad_(True)
        a = torch.max(x, eps)
        loss = a.log()
        return loss

    @staticmethod
    def compute_masks(graph_map, batch_variable_map, batch_function_map, edge_feature, device):
        edge_num = graph_map.size(1)
        variable_num = batch_variable_map.size(0)
        function_num = batch_function_map.size(0)
        all_ones = torch.ones(edge_num, device=device)
        edge_num_range = torch.arange(edge_num, dtype=torch.int64, device=device)

        variable_sparse_ind = torch.stack([edge_num_range, graph_map[0, :].long()])
        function_sparse_ind = torch.stack([graph_map[1, :].long(), edge_num_range])

        if device.type == 'cuda':
            variable_mask = torch.cuda.sparse.FloatTensor(variable_sparse_ind, edge_feature.squeeze(1),
                torch.Size([edge_num, variable_num]), device=device)
            function_mask = torch.cuda.sparse.FloatTensor(function_sparse_ind, all_ones,
                torch.Size([function_num, edge_num]), device=device)
        else:
            variable_mask = torch.sparse.FloatTensor(variable_sparse_ind, edge_feature.squeeze(1),
                torch.Size([edge_num, variable_num]), device=device)
            function_mask = torch.sparse.FloatTensor(function_sparse_ind, all_ones,
                torch.Size([function_num, edge_num]), device=device)

        return variable_mask, function_mask

    @staticmethod
    def compute_batch_mask(batch_variable_map, batch_function_map, device):
        variable_num = batch_variable_map.size()[0]
        function_num = batch_function_map.size()[0]
        variable_all_ones = torch.ones(variable_num, device=device)
        function_all_ones = torch.ones(function_num, device=device)
        variable_range = torch.arange(variable_num, dtype=torch.int64, device=device)
        function_range = torch.arange(function_num, dtype=torch.int64, device=device)
        batch_size = (batch_variable_map.max() + 1).long().item()

        variable_sparse_ind = torch.stack([variable_range, batch_variable_map.long()])
        function_sparse_ind = torch.stack([function_range, batch_function_map.long()])

        if device.type == 'cuda':
            variable_mask = torch.cuda.sparse.FloatTensor(variable_sparse_ind, variable_all_ones,
                torch.Size([variable_num, batch_size]), device=device)
            function_mask = torch.cuda.sparse.FloatTensor(function_sparse_ind, function_all_ones,
                torch.Size([function_num, batch_size]), device=device)
        else:
            variable_mask = torch.sparse.FloatTensor(variable_sparse_ind, variable_all_ones,
                torch.Size([variable_num, batch_size]), device=device)
            function_mask = torch.sparse.FloatTensor(function_sparse_ind, function_all_ones,
                torch.Size([function_num, batch_size]), device=device)

        variable_mask_transpose = variable_mask.transpose(0, 1)
        function_mask_transpose = function_mask.transpose(0, 1)

        return (variable_mask, variable_mask_transpose, function_mask, function_mask_transpose)

    def forward(self, variable_prediction, degree_loss, label, graph_map, batch_variable_map,
        batch_function_map, edge_feature, meta_data, global_step, eps, max_coeff, loss_sharpness):

        coeff = torch.min(global_step.pow(self._alpha), torch.tensor([max_coeff], device=self._device))

        signed_variable_mask_transpose, function_mask = \
            SatLossEvaluator.compute_masks(graph_map, batch_variable_map, batch_function_map,
            edge_feature, self._device)

        edge_values = torch.mm(signed_variable_mask_transpose, variable_prediction)
        edge_values = edge_values + (1 - edge_feature) / 2

        weights = (coeff * edge_values).exp()

        nominator = torch.mm(function_mask, weights * edge_values)
        denominator = torch.mm(function_mask, weights)

        clause_value = denominator / torch.max(nominator, eps)
        clause_value = 1 + (clause_value - 1).pow(loss_sharpness)
        if degree_loss is not None:
            temp_loss = torch.sparse.mm(signed_variable_mask_transpose, degree_loss.reshape([len(degree_loss), -1]))
            clause_value = torch.sparse.mm(function_mask, temp_loss) * clause_value

        return torch.mean(SatLossEvaluator.safe_log(clause_value, eps))


##########################################################################################################################


class SatCNFEvaluator(nn.Module):
    "Implements a module to evaluate the current prediction."

    def __init__(self, device):
        super(SatCNFEvaluator, self).__init__()
        self._device = device
        self._increment = 0.6
        self._floor = nn.Parameter(torch.tensor([1], dtype = torch.float, device = self._device), requires_grad = False)
        self._temperature = nn.Parameter(torch.tensor([6], dtype = torch.float, device = self._device),
                                         requires_grad = False)

    def simplify(self, sat_vars, variable_num, function_num, node_adj_lists, variable_prediction, graph_map, edge_feature, is_training):
        variables = list(sat_vars)  # variables 从 0 开始
        functions = np.array([l.numpy() for l in node_adj_lists])[variables]
        # functions = np.array(sat_problem.node_adj_lists)[variables]  # functions 从 1 开始
        symbols = ((variable_prediction[variables] > 0.5).to(torch.float) * 2 - 1).to(torch.long)
        try_times = 3
        ending = ''
        function_num_addition = 0
        flag = 1
        # indices = random.sample(range(len(variables)), math.floor(math.pow(self._temperature, self._increment)))
        if not is_training:
            try_times = 5
        print('\n----------------------')
        while try_times > 0:
            retry = True
            if flag <= 1:
                symbols_ = symbols.cpu().numpy()
                sample_num = max(math.floor(math.pow(self._temperature, self._increment)), 1)
                sample_num = min(sample_num, len(variables))
                degrees = np.array([len(node_adj_lists[i]) for i in range(len(node_adj_lists))])[variables]
                indices = degrees.argsort()[::-1][0: sample_num]
                # indices = random.sample(range(len(variables)), sample_num)
            deactivate_functions = []
            deactivate_varaibles = []
            indices_ = []
            for j in range(len(indices)):
                i = indices[j]
                temp = functions[i][torch.tensor(functions[i]) * symbols[i] > 0]
                temp = temp.cpu() if torch.is_tensor(temp) else temp
                pos_functions = np.array(temp).flatten()
                if len(pos_functions) < len(functions[i]):
                    deactivate_varaibles.append(variables[i])
                    indices_.append(i)
                deactivate_functions.extend(np.abs(pos_functions) - 1)
            deactivate_functions = list(set(deactivate_functions)) if len(deactivate_functions) > 0 else []
            sat_str = 'p cnf ' + str(variable_num) + ' ' + str(
                function_num - len(deactivate_functions) + function_num_addition) + '\n'
            for j in range(function_num):
                if j not in deactivate_functions:
                    clause = ((graph_map[0] + 1) * edge_feature.squeeze().to(torch.int))[graph_map[1] == j]
                    function_str = [i for i in map(str, clause.cpu().numpy()) if abs(int(i)) - 1 not in deactivate_varaibles]
                    if len(function_str) == 0:
                        return False, None
                    sat_str += ' '.join(function_str)
                    sat_str += ' 0\n'
            sat_str += ending

            # print('temperature: ', self._temperature)
            print(np.array(deactivate_varaibles) + 1, symbols[indices_].cpu().numpy().flatten())
            print('function change: ', function_num_addition - len(deactivate_functions))
            res = SatCNFEvaluator.use_solver(sat_str)
            if res:
                print('result: True')
                self._temperature += 1
                try_times -= 1  # return res, (np.array(variables)[indices] + 1, (symbols[indices].squeeze() > 0))
                return res, (np.array(variables)[indices_] + 1, (symbols[indices_].squeeze() > 0))
            elif res is False:
                print('result: False')
                try_times -= 1
                # self._temperature += 0.5
                # self.unsat_core(sat_problem)
                unsat_condition = (np.array(deactivate_varaibles) + 1) * np.array(symbols[indices_].cpu()).flatten() * -1
                ending += ' '.join([str(i) for i in unsat_condition])
                ending += ' 0\n'
                function_num_addition += 1

            else:
                print('result: None')
                if self._temperature > 0:
                    self._temperature -= 1
                try_times -= 1
                # if sat_problem.statistics[2] >= 3:
                #     flag = -1
                # if try_times > 0:
                #     unsat_condition = (np.array(deactivate_varaibles) + 1) * np.array(
                #         symbols[indices].cpu()).flatten() * -1
                #     ending += ' '.join([str(i) for i in unsat_condition])
                #     ending += ' 0\n'
                #     function_num_addition += 1
                #     for item in deactivate_varaibles:
                #         variables.remove(item)
                #     if len(deactivate_varaibles) >= 2:
                #         reverse_array = np.random.rand(len(deactivate_varaibles))
                #     else:
                #         reverse_array = np.ones(1)
                #     left_array = symbols.cpu.numpy() == symbols_
                #     symbols[indices] = torch.tensor(
                #         symbols_[indices] * (2 * (reverse_array > 0.5 and left_array) - 1).reshape(-1, 1) * -1,
                #         dtype = torch.long,
                #         device = self._device)
                if try_times > 0:
                    if flag > 2:
                        for item in deactivate_varaibles:
                            variables.remove(item)
                        flag = 1
                        symbols = torch.tensor(symbols_, device = self._device)
                    else:
                        retry_time = 0
                        while retry and retry_time < 3:
                            left_array = (symbols[indices_].cpu().numpy() == symbols_[indices_]).flatten()
                            if len(deactivate_varaibles) >= 2:
                                reverse_array = np.random.rand(len(deactivate_varaibles))
                            else:
                                reverse_array = np.ones(len(deactivate_varaibles))
                            origin_array = np.array(deactivate_varaibles) * symbols[indices_].cpu().numpy().flatten()

                            symbols[indices_] = torch.tensor(symbols_[indices_] * (2 * (
                                np.any([np.all([reverse_array > 0.5, left_array], axis = 0), np.logical_not(left_array)], axis = 0)) - 1).reshape(-1, 1) * -1, dtype = torch.long, device = self._device)
                            current_array = np.array(deactivate_varaibles) * symbols[indices_].cpu().numpy().flatten()
                            if not np.all(origin_array == current_array):
                                retry = False
                            retry_time += 1
                            flag += 1
        return res, (np.array(variables)[indices_] + 1, (symbols[indices_].squeeze() > 0))

    @staticmethod
    def use_solver(sat_str):
        """使用确定性求解器求解"""
        root_path = os.getcwd()
        file_path = os.path.join(root_path, 'datasets', 'temp.cnf')
        with open(file_path, 'w+') as f:
            f.write(sat_str)
        # process = subprocess.Popen([root_path + '/glucose_release', file_path], stdout = subprocess.PIPE, encoding = 'utf-8')
        process = subprocess.Popen(['z3', file_path], stdout = subprocess.PIPE, encoding = 'utf-8')
        try:
            outs, errs = process.communicate(timeout = 20)
            # if outs.find('s SATISFIABLE') >= 0:
            if outs.find('unsat') >= 0:
                return False
            return True
        except:
            process.kill()
            # print('process killed')
            return None

    def forward(self, variable_prediction, graph_map, batch_variable_map,
        batch_function_map, edge_feature, meta_data, node_adj_list = None):
        # print('graph_map\n', [int(v) for v in graph_map[1].numpy()])

        variable_num = batch_variable_map.size(0)
        function_num = batch_function_map.size(0)
        batch_size = (batch_variable_map.max() + 1).item()
        all_ones = torch.ones(function_num, 1, device=self._device)


        signed_variable_mask_transpose, function_mask = \
            SatLossEvaluator.compute_masks(graph_map, batch_variable_map, batch_function_map,
            edge_feature, self._device)

        b_variable_mask, b_variable_mask_transpose, b_function_mask, b_function_mask_transpose = \
            SatLossEvaluator.compute_batch_mask(
            batch_variable_map, batch_function_map, self._device)

        edge_values = torch.mm(signed_variable_mask_transpose, variable_prediction)
        edge_values = edge_values + (1 - edge_feature) / 2
        edge_values = (edge_values > 0.5).float()
        # print('\n\nedge_values\n', [int(v) for v in edge_values.numpy()])

        clause_values = torch.mm(function_mask, edge_values)
        clause_values = (clause_values > 0).float()
        # print('\n\n\n\n', [(k, int(v)) for k,v in enumerate(clause_values.numpy()) if int(v) == 0])
        unsat_vars = meta_data.to_dense()[np.argwhere(clause_values.squeeze(1) == 0)[0]]
        unsat_vars = set(np.argwhere(unsat_vars > 0)[1].numpy())
        sat_vars = meta_data.to_dense()[np.argwhere(clause_values.squeeze(1) == 1)[0]]
        sat_vars = set(np.argwhere(sat_vars > 0)[1].numpy())
        '''相减后获得差集: 即可以确定值的变量 -> 变量所在的子句均为可满足子句'''
        sat_vars = sat_vars - unsat_vars
        if node_adj_list is not None:
            res, variables = self.simplify(sat_vars, variable_num, function_num, node_adj_list, variable_prediction, graph_map, edge_feature, True)

        max_sat = torch.mm(b_function_mask_transpose, all_ones)
        batch_values = torch.mm(b_function_mask_transpose, clause_values)
        # print ('max_sat\n', max_sat, '\nbatch_values\n', batch_values)

        return (max_sat == batch_values).float(), max_sat - batch_values, graph_map, clause_values, edge_values


##########################################################################################################################


class PerceptronTanh(nn.Module):
    "Implements a 1-layer perceptron with Tanh activaton."

    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        super(PerceptronTanh, self).__init__()
        self._layer1 = nn.Linear(input_dimension, hidden_dimension)
        self._layer2 = nn.Linear(hidden_dimension, output_dimension, bias=False)

    def forward(self, inp):
        return F.tanh(self._layer2(F.relu(self._layer1(inp))))


##########################################################################################################################


def sparse_argmax(x, mask, device):
    "Implements the exact, memory-inefficient argmax operation for a row vector input."

    if device.type == 'cuda':
        dense_mat = torch.cuda.sparse.FloatTensor(mask._indices(), x - x.min() + 1, mask.size(), device=device).to_dense()
    else:
        dense_mat = torch.sparse.FloatTensor(mask._indices(), x - x.min() + 1, mask.size(), device=device).to_dense()

    return torch.argmax(dense_mat, 0)


def sparse_max(x, mask, device):
    "Implements the exact, memory-inefficient max operation for a row vector input."

    if device.type == 'cuda':
        dense_mat = torch.cuda.sparse.FloatTensor(mask._indices(), x - x.min() + 1, mask.size(), device=device).to_dense()
    else:
        dense_mat = torch.sparse.FloatTensor(mask._indices(), x - x.min() + 1, mask.size(), device=device).to_dense()

    return torch.max(dense_mat, 0)[0] + x.min() - 1


def safe_exp(x, device):
    "Implements safe exp operation."

    return torch.min(x, torch.tensor([30.0], device=device)).exp()


def sparse_smooth_max(x, mask, device, alpha=30):
    "Implements the approximate, memory-efficient max operation for a row vector input."

    coeff = safe_exp(alpha * x, device)
    return torch.mm(mask, x * coeff) / torch.max(torch.mm(mask, coeff), torch.ones(1, device=device))
