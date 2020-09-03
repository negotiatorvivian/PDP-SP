#!/usr/bin/env python3

import argparse
import random
import traceback
from os import listdir
from os.path import isfile, join, split, splitext
import itertools
from itertools import combinations
import numpy as np
import re
import sp_aggregators

class CompactDimacs:
    "Encapsulates a CNF file given in the DIMACS format."

    def __init__(self, dimacs_file, output, propagate, ans_file = None):

        self.propagate = propagate
        self.file_name = split(dimacs_file)[1]
        self.max_len = 0

        with open(dimacs_file, 'r') as f:
            j = 0
            for line in f:
                seg = line.split(" ")
                if seg[0] == 'c':
                    continue

                if seg[0] == 'p':
                    var_num = int(seg[2])
                    clause_num = int(seg[3])
                    self._clause_mat = np.zeros((clause_num, var_num), dtype=np.int32)

                elif len(seg) <= 1:
                    continue
                else:
                    temp = np.array(seg[:-1], dtype=np.int32)
                    self.max_len = max(self.max_len, len(temp))
                    self._clause_mat[j, np.abs(temp) - 1] = np.sign(temp)
                    j += 1

        ind = np.where(np.sum(np.abs(self._clause_mat), 1) > 0)[0]
        # self.diff = np.array([self.max_len - np.sum(self._clause_mat[i, :] != 0) for i in range(len(ind))], dtype = np.int)
        # self._clause_mat = np.concatenate((self._clause_mat, np.zeros([self._clause_mat.shape[0], max(self.diff)])), axis = 1)
        # for i in range(len(self.diff)):
        #     if self.diff[i] > 0:
        #         self._clause_mat[i, -self.diff[i]:] = 1

        self._clause_mat = self._clause_mat[ind, :]

        if propagate:
            self._clause_mat = self._propagate_constraints(self._clause_mat)

        self._output = output
        # self.edge_feature = sp_aggregators.get_sp_aggregate(self._clause_mat)

    def _propagate_constraints(self, clause_mat):
        n = clause_mat.shape[0]
        if n < 2:
            return clause_mat

        length = np.tile(np.sum(np.abs(clause_mat), 1), (n, 1))
        intersection_len = np.matmul(clause_mat, np.transpose(clause_mat))

        temp = intersection_len == np.transpose(length)
        temp *= np.tri(*temp.shape, k=-1, dtype=bool)
        flags = np.logical_not(np.any(temp, 0))

        clause_mat = clause_mat[flags, :]

        n = clause_mat.shape[0]
        if n < 2:
            return clause_mat

        length = np.tile(np.sum(np.abs(clause_mat), 1), (n, 1))
        intersection_len = np.matmul(clause_mat, np.transpose(clause_mat))

        temp = intersection_len == length
        temp *= np.tri(*temp.shape, k=-1, dtype=bool)
        flags = np.logical_not(np.any(temp, 1))

        return clause_mat[flags, :]

    def to_json(self, base = 0):
        clause_num, var_num = self._clause_mat.shape
        var_num = var_num + base
        ind = np.nonzero(self._clause_mat)
        negative_sample_set = np.argwhere(self._clause_mat == 0)
        relations = ''
        node_type = ''
        test_str = ''
        validate_str = ''
        type_1 = set()
        type_2 = set()
        for i, item in enumerate(ind[1]):
            # if i % 20 == 0 and i > 0:
            #     print(max(type_2), max(type_1))
            var_item = item + base
            temp = str(self._clause_mat[ind[0][i]][item]) + ' ' + str(var_item) + ' ' + str(ind[0][i] + var_num)
            # relations += temp + '\n'
            seed = random.random()
            if seed > 0.95:
                validate_str += temp + ' 1 \n'
                negative_relation = negative_sample_set[random.randint(0, len(negative_sample_set) - 1)]
                edge_type = '1 ' if random.random() > 0.5 else '-1 '
                validate_str += edge_type + str(negative_relation[1] + base) + ' ' + str(negative_relation[0] + var_num) + ' 0\n'
            elif seed > 0.85:
                test_str += temp + ' 1 \n'
                negative_relation = negative_sample_set[random.randint(0, len(negative_sample_set) - 1)]
                edge_type = '1 ' if random.random() > 0.5 else '-1 '
                test_str += edge_type + str(negative_relation[1] + base) + ' ' + str(negative_relation[0] + var_num) + ' 0\n'
            else:
                relations += temp + '\n'

            if var_item in type_1:
                pass
            else:
                node_type += str(var_item) + ' 0\n'
                type_1.add(var_item)
            if (ind[0][i] + var_num) in type_2:
                pass
            else:
                node_type += str(ind[0][i] + var_num) + ' 1\n'
                type_2.add(ind[0][i] + var_num)
        # return [[var_num, clause_num], list(((ind[1] + 1) * self._clause_mat[ind]).astype(np.int)), list(ind[0] + 1), self._output]
        return node_type, relations, test_str, validate_str, max(type_2)


class CompactSatDimacs:
    "Encapsulates a CNF file given in the DIMACS format."

    def __init__(self, dimacs_file, ans_file = None):
        self.file_name = split(dimacs_file)[1]
        self.flag = True
        with open(ans_file, 'r') as ans:
            content = ans.readlines()
            self.content = None
            if content is not None and len(content) == 5:
                if 'UNSATISFIABLE' not in content[3]:
                    self.content = np.array((content[4].strip().split(' ')[1: -1]), dtype=np.int)
        if self.content is None:
            self.flag = False
        else:
            with open(dimacs_file, 'r') as f:
                j = 0
                for line in f:
                    seg = line.split(" ")
                    if seg[0] == 'c':
                        continue

                    if seg[0] == 'p':
                        var_num = int(seg[2])
                        clause_num = int(seg[3])
                        self._clause_mat = np.zeros((clause_num, var_num), dtype=np.int32)
                        self._clause_mat_new = np.zeros((clause_num, var_num), dtype=np.int32)

                    elif len(seg) <= 1:
                        continue
                    else:
                        temp = np.array(seg[:-1], dtype=np.int32)
                        self._clause_mat_new[j, np.abs(temp) - 1] = np.sign(temp)

                        # Use correct answer
                        if self.content is None:
                            edge = temp
                        else:
                            edge = self.content[np.argwhere([np.abs(self.content) == abs(temp[i]) for i in range(len(temp) - 1)])[:, 1]]
                            edge = np.append(edge, [0])
                        self._clause_mat[j, np.abs(temp) - 1] = np.sign(edge)
                        j += 1

            ind = np.where(np.sum(np.abs(self._clause_mat), 1) > 0)[0]
            self._clause_mat = self._clause_mat[ind, :]
            self.pairs = self.generate_pairs()

    def to_json_raw(self, base = 0):
        try:
            clause_num, var_num = self._clause_mat.shape
            ind = np.nonzero(self._clause_mat)
            variables = ((ind[1] + 1) * self._clause_mat[ind]).astype(np.int)
            # variables = variables + abs(min(variables))
            links = []
            nodes = []
            for i in range(self._clause_mat.shape[0]):
                num_index = variables[np.argwhere(ind[0] == i)]
                for item in list(combinations(num_index, 2)):
                    link = {
                        'test_removed': False,
                        'train_removed': False,
                        'target': max(np.abs(item))[0] + base,
                        'source': min(np.abs(item))[0] + base
                    }
                    links.append(link)
            for i in range(var_num):
                node = {
                    'test': False,
                    'id': i + base,
                    'features': np.abs(self._clause_mat[:, i]).tolist(),
                    'val': False,
                    'lable': [1, 0] if self.content[i] < 0 else [0, 1]
                }
                nodes.append(node)
        except Exception as e:
            traceback.print_exc()

        return nodes, links, var_num + base

    def to_json_promote(self, base = 0, count = 0, test = False, val = False):
        try:
            clause_num, var_num = self._clause_mat_new.shape
            ind = np.nonzero(self._clause_mat_new)
            variables = ((ind[1] + 1) * self._clause_mat_new[ind]).astype(np.int)
            # variables = variables + abs(min(variables))
            links = []
            nodes = []
            label = {}

            for item in self.pairs:
                link = {
                    # "test_removed": False,
                    # "train_removed": False,
                    "target": max(item) + base,
                    "source": min(item) + base,
                    "key": str(count)
                }
                links.append(link)
            for i in range(var_num):
                # self._clause_mat_new[np.argwhere(self._clause_mat_new[:, i] == -1).flatten()] = 2
                # node_label = np.zeros(32, dtype=np.int)
                # if self.content[i] < 0:
                #     node_label[:16] = 1
                # else:
                #     node_label[16:] = 1

                node = {
                    "test": test,
                    "id": i + base,
                    "features": np.abs(self._clause_mat_new[:, i]).tolist(),
                    "val": val,
                    "label": [0, 1] if self.content[i] < 0 else [1, 0]
                }
                label[str(node["id"])] = node["label"]
                nodes.append(node)
            walks = self.generate_relations(base)
        except Exception as e:
            traceback.print_exc()

        # return nodes, links, label, walks, clause_num + var_num + base
        return nodes, links, label, walks, var_num + base

    def generate_relations(self, base):
        links = {}
        for index in range(self._clause_mat_new.shape[1]):
            links[index] = []
            func_pos_index = np.where(self._clause_mat_new[:, index] == 1)[0]
            func_neg_index = np.where(self._clause_mat_new[:, index] == -1)[0]
            pos_pairs = [(np.where(self._clause_mat_new[i, :] != 0)[0] + base).tolist() for i in func_pos_index]
            neg_pairs = [(np.where(self._clause_mat_new[i, :] != 0)[0] + base).tolist() for i in func_neg_index]
            for item in pos_pairs:
                if len(item) >= 3:
                    item.remove(index + base)
                    for t in list(combinations(item, 2)):
                        links[index].append(t)
            for item in neg_pairs:
                if len(item) >= 3:
                    item.remove(index + base)
                    for t in list(combinations(item, 2)):
                        links[index].append(t)
            if len(func_pos_index) > 0 and len(func_neg_index) > 0:
                pos_vars = []
                neg_vars = []
                for item in pos_pairs:
                    pos_vars.extend(item)
                for item in neg_pairs:
                    neg_vars.extend(item)

                # index_tuple = list(combinations(range(max(len(pos_vars), len(neg_vars))), 2))
                compete_pairs = list(itertools.product(pos_vars, neg_vars))
                # compete_pairs = [(pos_vars[item[0]], neg_vars[item[1]]) for item in index_tuple if item[0] <
                #                  len(pos_vars) and item[1] < len(neg_vars)]
                # compete_pairs += [(pos_vars[item[1]], neg_vars[item[0]]) for item in index_tuple if item[0] <
                #                  len(neg_vars) and item[1] < len(pos_vars)]
                # compete_pairs += [(pos_vars[item[0]], neg_vars[item[0]]) for item in index_tuple if item[0] <
                #                  len(pos_vars) and item[0] < len(neg_vars)]
                # compete_pairs += [(pos_vars[item[1]], neg_vars[item[1]]) for item in index_tuple if item[1] <
                #                  len(pos_vars) and item[1] < len(neg_vars)]
                links[index] += compete_pairs
        return links

    def generate_pairs(self):
        pairs = []
        for index in range(self._clause_mat_new.shape[0]):
            ind = np.where(self._clause_mat_new[index, :] != 0)[0]
            pos, neg = [], []
            for i in range(len(ind)):
                if self._clause_mat_new[index, ind][i] > 0:
                    pos.append(ind[i])
                else:
                    neg.append(ind[i])

            if len(pos) >= 2:
                pos_com = combinations(pos, 2)
                for pair in list(pos_com):
                    pairs.append((pair[0], pair[1]))
                    pairs.append((pair[1], pair[0]))
            if len(neg) >= 2:
                neg_com = combinations(neg, 2)
                for pair in list(neg_com):
                    pairs.append((pair[0], pair[1]))
                    pairs.append((pair[1], pair[0]))
            if len(pos) > 0 and len(neg) > 0:
                all_com = list(itertools.product(neg, pos))
                pairs.extend(all_com)
        return pairs


def convert_directory(dimacs_dir, answer_dir, output_file, propagate = False, only_positive=False):
    file_list = [join(dimacs_dir, f) for f in listdir(dimacs_dir) if isfile(join(dimacs_dir, f))]
    try:
        base = 0
        if answer_dir is None:
            output_file_list = output_file.split('$')
            if len(output_file_list) < 4:
                raise Exception('Invalid argument output_file', output_file)
            node_type_file = open(output_file_list[0], 'w')
            train_file = open(output_file_list[1], 'w')
            test_file = open(output_file_list[2], 'w')
            validate_file = open(output_file_list[3], 'w')
            for i in range(len(file_list)):
                name, ext = splitext(file_list[i])
                ext = ext.lower()

                if ext != '.dimacs' and ext != '.cnf':
                    continue

                # label = float(name[-1]) if name[-1].isdigit() else -1
                # label = 1 if name.split('=')[-1] == 'True' else -1
                label = -1

                if only_positive and label == 0:
                    continue

                bc = CompactDimacs(file_list[i], label, propagate)
                # f.write(str(bc.to_json_raw()).replace("'", '"') + '\n')
                # print("Generating JSON input file: %6.2f%% complete..." % (
                #     (i + 1) * 100.0 / len(file_list)), end='\r', file=sys.stderr)
                node_type, relations, test_str, validate_str, base = bc.to_json_raw(base)
                train_file.write(relations)
                node_type_file.write(node_type)
                test_file.write(test_str)
                validate_file.write(validate_str)

            node_type_file.close()
            train_file.close()
            test_file.close()
            validate_file.close()
        elif answer_dir is not None:
            ans_list = [join(answer_dir, f) for f in listdir(answer_dir) if isfile(join(answer_dir, f))]
            json_dict = {
                "directed": True,
                "graph": {"name": "disjoint_union( ,  )"},
                "nodes": [],
                "links": [],
                "multigraph": True
            }
            label_dict = {}
            id_map_dict = {}
            walks = []
            test_len = int(0.2 * len(file_list))
            val_len = int(0.1 * len(file_list))
            test_index = random.sample(range(len(file_list)), test_len)
            val_index = random.sample(set(range(len(file_list))) - set(test_index), val_len)
            content = None
            print(test_index, val_index)
            for i, file in enumerate(file_list):
                ans_file = answer_dir + '/' + file.split('/')[-1]
                index = ans_list.index(ans_file)
                bc = CompactSatDimacs(file, ans_list[index])
                if bc.flag is False:
                    continue
                if content is None:
                    content = np.sign(bc.content)
                else:
                    content = np.concatenate((content, np.sign(bc.content)))
                test = True if i in test_index else False
                val = True if i in val_index else False
                node, link, label, walk, base = bc.to_json_promote(base, i, test, val)
                json_dict['nodes'] += node
                json_dict['links'] += link
                label_dict.update(label)
                walks.extend(walk.values())

                # node, link, base = bc.to_json_raw(base)
                # json_dict['nodes'] += node
                # json_dict['links'] += link
            for i in range(base):
                id_map_dict[str(i)] = i
            out_file = open(output_file, 'w')
            out_file.write(str(json_dict).replace("'", '"').lower())
            out_file.close()
            label_file = open('datasets/map_data/sat-class_map.json', 'w')
            label_file.write(str(label_dict).replace("'", '"'))
            label_file.close()
            id_map_file = open('datasets/map_data/sat-id_map.json', 'w')
            id_map_file.write(str(id_map_dict).replace("'", '"'))
            id_map_file.close()
            walk_file = open('datasets/map_data/sat-walks.txt', 'w')
            walks = re.sub('\(|\[|\]\]|\]', '', str(walks))
            walks = re.sub('\), ', '\n', str(walks))
            walks = re.sub(', ', ' ', str(walks))

            walk_file.write(str(walks).replace("'", '"') + '\n')
            walk_file.close()
            np.save('datasets/map_data/right_ans.npy', content)

    except Exception as e:
        print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', action='store', type=str)
    parser.add_argument('ans_file', action='store', type=str, default = 'datasets/SAT_ANS')
    parser.add_argument('out_file', action='store', type=str)
    parser.add_argument('-s', '--simplify', help='Propagate binary constraints', required=False, action='store_true', default=False)
    parser.add_argument('-p', '--positive', help='Output only positive examples', required=False, action='store_true', default=False)
    args = vars(parser.parse_args())

    convert_directory(args['in_dir'], args['ans_file'], args['out_file'], args['simplify'], args['positive'])
