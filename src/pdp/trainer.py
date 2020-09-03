# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

# PDP_solver_trainer.py : Implements a factor graph trainer for various types of PDP SAT solvers.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import subprocess
from collections import Counter

from pdp.factorgraph import FactorGraphTrainerBase
from pdp.nn import solver, util


##########################################################################################################################


class Perceptron(nn.Module):
    """Implements a 1-layer perceptron."""

    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        super(Perceptron, self).__init__()
        self._layer1 = nn.Linear(input_dimension, hidden_dimension)
        self._layer2 = nn.Linear(hidden_dimension, output_dimension, bias = False)

    def forward(self, inp):
        return F.sigmoid(self._layer2(F.relu(self._layer1(inp))))


##########################################################################################################################

class SatFactorGraphTrainer(FactorGraphTrainerBase):
    """Implements a factor graph trainer for various types of PDP SAT solvers."""

    def __init__(self, config, use_cuda, logger):
        super(SatFactorGraphTrainer, self).__init__(config = config,
                                                    has_meta_data = False, error_dim = config['error_dim'], loss = None,
                                                    evaluator = nn.L1Loss(), use_cuda = use_cuda, logger = logger)

        self._eps = 1e-8 * torch.ones(1, device = self._device, requires_grad = False)
        self._loss_evaluator = util.SatLossEvaluator(alpha = self._config['exploration'], device = self._device)
        self._cnf_evaluator = util.SatCNFEvaluator(device = self._device)
        self._counter = 0
        self._max_coeff = 10.0

    def _build_graph(self, config):
        model_list = []

        if config['model_type'] == 'np-nd-np':
            model_list += [solver.NeuralPropagatorDecimatorSolver(device = self._device, name = config['model_name'],
                                                                  edge_dimension = config['edge_feature_dim'],
                                                                  meta_data_dimension = config['meta_feature_dim'],
                                                                  propagator_dimension = config['hidden_dim'],
                                                                  decimator_dimension = config['hidden_dim'],
                                                                  mem_hidden_dimension = config['mem_hidden_dim'],
                                                                  agg_hidden_dimension = config['agg_hidden_dim'],
                                                                  mem_agg_hidden_dimension = config[
                                                                      'mem_agg_hidden_dim'],
                                                                  prediction_dimension = config['prediction_dim'],
                                                                  variable_classifier = Perceptron(config['hidden_dim'],
                                                                                                   config[
                                                                                                       'classifier_dim'],
                                                                                                   config[
                                                                                                       'prediction_dim']),
                                                                  function_classifier = None,
                                                                  dropout = config['dropout'],
                                                                  local_search_iterations = config[
                                                                      'local_search_iteration'],
                                                                  epsilon = config['epsilon'])]

        elif config['model_type'] == 'p-nd-np':
            model_list += [solver.NeuralSurveyPropagatorSolver(device = self._device, name = config['model_name'],
                                                               edge_dimension = config['edge_feature_dim'],
                                                               meta_data_dimension = config['meta_feature_dim'],
                                                               decimator_dimension = config['hidden_dim'],
                                                               mem_hidden_dimension = config['mem_hidden_dim'],
                                                               agg_hidden_dimension = config['agg_hidden_dim'],
                                                               mem_agg_hidden_dimension = config['mem_agg_hidden_dim'],
                                                               prediction_dimension = config['prediction_dim'],
                                                               variable_classifier = Perceptron(config['hidden_dim'],
                                                                                                config[
                                                                                                    'classifier_dim'],
                                                                                                config[
                                                                                                    'prediction_dim']),
                                                               function_classifier = None, dropout = config['dropout'],
                                                               local_search_iterations = config[
                                                                   'local_search_iteration'],
                                                               epsilon = config['epsilon'])]

        elif config['model_type'] == 'np-d-np':
            model_list += [solver.NeuralSequentialDecimatorSolver(device = self._device, name = config['model_name'],
                                                                  edge_dimension = config['edge_feature_dim'],
                                                                  meta_data_dimension = config['meta_feature_dim'],
                                                                  propagator_dimension = config['hidden_dim'],
                                                                  decimator_dimension = config['hidden_dim'],
                                                                  mem_hidden_dimension = config['mem_hidden_dim'],
                                                                  agg_hidden_dimension = config['agg_hidden_dim'],
                                                                  mem_agg_hidden_dimension = config[
                                                                      'mem_agg_hidden_dim'],
                                                                  classifier_dimension = config['classifier_dim'],
                                                                  dropout = config['dropout'],
                                                                  tolerance = config['tolerance'],
                                                                  t_max = config['t_max'],
                                                                  local_search_iterations = config[
                                                                      'local_search_iteration'],
                                                                  epsilon = config['epsilon'])]

        elif config['model_type'] == 'p-d-p':
            model_list += [solver.SurveyPropagatorSolver(device = self._device, name = config['model_name'],
                                                         tolerance = config['tolerance'], t_max = config['t_max'],
                                                         local_search_iterations = config['local_search_iteration'],
                                                         epsilon = config['epsilon'])]

        elif config['model_type'] == 'walk-sat':
            model_list += [solver.WalkSATSolver(device = self._device, name = config['model_name'],
                                                iteration_num = config['local_search_iteration'],
                                                epsilon = config['epsilon'])]

        elif config['model_type'] == 'reinforce':
            model_list += [solver.ReinforceSurveyPropagatorSolver(device = self._device, name = config['model_name'],
                                                                  pi = config['pi'], decimation_probability = config[
                    'decimation_probability'],
                                                                  local_search_iterations = config[
                                                                      'local_search_iteration'],
                                                                  epsilon = config['epsilon'])]

        if config['verbose']:
            self._logger.info("The model parameter count is %d." % model_list[0].parameter_count())
            self._logger.info("The model list is %s." % model_list)

        return model_list

    def _compute_loss(self, model, loss, prediction, label, graph_map, batch_variable_map,
                      batch_function_map, edge_feature, meta_data, degree_loss = None):

        res = self._loss_evaluator(variable_prediction = prediction[0], degree_loss = degree_loss, label = label,
                                   graph_map = graph_map,
                                   batch_variable_map = batch_variable_map, batch_function_map = batch_function_map,
                                   edge_feature = edge_feature, meta_data = meta_data, global_step = model._global_step,
                                   eps = self._eps, max_coeff = self._max_coeff,
                                   loss_sharpness = self._config['loss_sharpness'])
        # print(res)
        return res

    def _compute_evaluation_metrics(self, model, evaluator, prediction, label, graph_map,
                                    batch_variable_map, batch_function_map, edge_feature, meta_data):

        output, _, _, _, _ = self._cnf_evaluator(variable_prediction = prediction[0], graph_map = graph_map,
                                                 batch_variable_map = batch_variable_map,
                                                 batch_function_map = batch_function_map,
                                                 edge_feature = edge_feature, meta_data = meta_data)

        recall = torch.sum(label * ((output > 0.5).float() - label).abs()) / torch.max(torch.sum(label), self._eps)
        accuracy = evaluator((output > 0.5).float(), label).unsqueeze(0)
        loss_value = self._loss_evaluator(variable_prediction = prediction[0], label = label, graph_map = graph_map,
                                          batch_variable_map = batch_variable_map,
                                          batch_function_map = batch_function_map,
                                          edge_feature = edge_feature, meta_data = meta_data,
                                          global_step = model._global_step,
                                          eps = self._eps, max_coeff = self._max_coeff,
                                          loss_sharpness = self._config['loss_sharpness']).unsqueeze(0)

        return torch.cat([accuracy, recall, loss_value], 0)

    def _post_process_predictions(self, model, prediction, graph_map,
                                  batch_variable_map, batch_function_map, edge_feature, edge_feature_, graph_feat,
                                  label, misc_data, variable_num, function_num):
        """Formats the prediction and the output solution into JSON format."""

        message = ""
        labs = label.detach().cpu().numpy()

        res = self._cnf_evaluator(variable_prediction = prediction[0], graph_map = graph_map,
                                  batch_variable_map = batch_variable_map, batch_function_map = batch_function_map,
                                  edge_feature = edge_feature, meta_data = graph_feat)
        output, unsat_clause_num, graph_map, clause_values, edge_values = [a.detach().cpu().numpy() for a in res]
        unsat_clause_index = [k for k, v in enumerate(clause_values) if int(v) < 1]
        # for i in unsat_clause_index:
        #     cur_clause = []
        #     if len(cur_clause) == 0:
        #         start = 0
        #     index = unsat_clause_index.index(start, i)

        # print('\nclause_values:' + str(clause_values))
        clauses = {}
        j = 0
        clauses[j] = []
        var_base = 0
        func_base = function_num[j]
        for k in unsat_clause_index:
            if k >= func_base:
                var_base += variable_num[j]
                func_base += function_num[j + 1] if len(function_num) > (j + 1) else 0
                j += 1
                clauses[j] = []
            i = k
            index_list = []
            clause_list = []
            while True:
                start = index_list[-1] if len(index_list) > 0 else 0
                try:
                    index = graph_map[1].tolist().index(i, start + 1)
                except:
                    break
                index_list.append(index)
                clause_list.append(
                    (graph_map[0].tolist()[index] + 1 - var_base) * int(edge_feature_.tolist()[index][0]))
            clauses[j].append(clause_list)
        print(clauses)

        for i in range(output.shape[0]):
            # if unsat_clause_num[i] > 0:
            #     # print(graph_map[0])
            #     j = i
            #     input_map = []
            #     s = function_num[j - 1] if j > 0 else 0
            #     t = s + function_num[j]
            #     for k in clause_values[s:t]:
            #         j = k[0]
            #         index_list = []
            #         variable_list = []
            #         while True:
            #             start = index_list[-1] if len(index_list) > 0 else 0
            #             try:
            #                 index = graph_map[1].tolist().index(j, start + 1)
            #             except:
            #                 break
            #             variable_list.append(str(graph_map[0].tolist()[index] + 1 if int(edge_values.tolist()[index][0]) > 0
            #                                                       else -(graph_map[0].tolist()[index] + 1)))
            #             index_list.append(index)
            #         if len(input_map) == 0 and i == 0:
            #             m = index_list[0] - 1
            #             variable_list.append(str(graph_map[0].tolist()[m] + 1 if int(edge_values.tolist()[m][0]) > 0
            #                                                       else -(graph_map[0].tolist()[m] + 1)))
            #         variable_list.append('0')
            #         input_map.extend(variable_list)
            #     input_map.insert(0, 'p cnf ' + str(variable_num[i]) + ' ' + str(function_num[i]))
            #     # print(input_map)

            instance = {
                # 'ID': misc_data[i][0] if len(misc_data[i]) > 0 else "",
                # 'label': int(labs[i, 0]),
                'solved': int(output[i].flatten()[0] == 1),
                'unsat_clauses': int(unsat_clause_num[i].flatten()[0]),
                # 'solution': (prediction[0][batch_variable_map == i, 0].detach().cpu().numpy().flatten() > 0.5).astype(int).tolist()
            }
            message += (str(instance).replace("'", '"') + "\n")
            self._counter += 1

        return message

    def _check_recurrence_termination(self, active, prediction, sat_problem):
        "De-actives the CNF examples which the model has already found a SAT solution for."

        output, _, _, _, _ = self._cnf_evaluator(variable_prediction = prediction[0],
                                                 graph_map = sat_problem._graph_map,
                                                 batch_variable_map = sat_problem._batch_variable_map,
                                                 batch_function_map = sat_problem._batch_function_map,
                                                 edge_feature = sat_problem._edge_feature,
                                                 meta_data = sat_problem._meta_data)  # .detach().cpu().numpy()

        if sat_problem._batch_replication > 1:
            real_batch = torch.mm(sat_problem._replication_mask_tuple[1], (output > 0.5).float())
            dup_batch = torch.mm(sat_problem._replication_mask_tuple[0], (real_batch == 0).float())
            active[active[:, 0], 0] = (dup_batch[active[:, 0], 0] > 0)
        else:
            active[active[:, 0], 0] = (output[active[:, 0], 0] <= 0.5)
        self._sort_vars(prediction[0].tolist(), 10)

    def search_solution(self, input_map):
        "Use other solver to find a solution"

        input_stream = (' ').join(input_map)
        with open('temp_file', 'w') as f:
            f.write(input_stream)
            process = subprocess.run('/home/ziwei/Downloads/', stdin = f, stdout = subprocess.PIPE, encoding = 'utf-8')

    def _sort_vars(self, scores, top_n):
        '''Sort variables according to their predicted value.'''
        score_dict = {}
        for k, v in enumerate(scores):
            score_dict[k] = round(v[0], 4)

        result = Counter(score_dict).most_common(top_n)
        d = {}
        for k, v in result:
            d[k] = v
        print(d)
        return d
