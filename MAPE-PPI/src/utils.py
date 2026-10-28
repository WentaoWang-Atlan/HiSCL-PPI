import os
import dgl
import torch
import shutil
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_writable(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass


# Calculation of evaluation metrics
def evaluat_metrics(output, label):
    # Vectorized micro-F1 — bit-identical to the former per-element Python loop.
    # Global TP/FP/FN over all N*C elements, then P/R/F1 (same 1e-10 eps).
    pre_y = torch.sigmoid(output) > 0.5          # bool
    truth = label                                # raw 0/1 values (float or long)
    is_one = truth == 1
    is_zero = truth == 0
    eq = pre_y == truth                          # pred matches label (numeric)
    TP = int((eq & is_one).sum().item())
    FN = int((~eq & is_one).sum().item())
    FP = int((~eq & is_zero).sum().item())
    Precision = TP / (TP + FP + 1e-10)
    Recall = TP / (TP + FN + 1e-10)
    F1_score = 2 * Precision * Recall / (Precision + Recall + 1e-10)
    return F1_score


# Data splitting by BFS
def get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    candiate_node = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 20:
        random_node = random.randint(0, node_num - 1)
    candiate_node.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = candiate_node.pop(0)
        selected_node.append(cur_node)

        for edge_index in node_to_edge_index[cur_node]:
            if edge_index not in selected_edge_index:
                selected_edge_index.append(edge_index)

                end_node = -1
                if ppi_list[edge_index][0] == cur_node:
                    end_node = ppi_list[edge_index][1]
                else:
                    end_node = ppi_list[edge_index][0]

                if end_node not in selected_node and end_node not in candiate_node:
                    candiate_node.append(end_node)
            else:
                continue


    return selected_edge_index


# Data splitting by DFS
def get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    stack = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 20:
        random_node = random.randint(0, node_num - 1)
    stack.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = stack[-1]

        if cur_node in selected_node:
            flag = True

            for edge_index in node_to_edge_index[cur_node]:
                if flag:
                    end_node = -1

                    if ppi_list[edge_index][0] == cur_node:
                        end_node = ppi_list[edge_index][1]
                    else:
                        end_node = ppi_list[edge_index][0]

                    if end_node in selected_node:
                        continue
                    else:
                        stack.append(end_node)
                        flag = False
                else:
                    break

            if flag:
                stack.pop()
            continue

        else:
            selected_node.append(cur_node)

            for edge_index in node_to_edge_index[cur_node]:
                if edge_index not in selected_edge_index:
                    selected_edge_index.append(edge_index)

    return selected_edge_index
