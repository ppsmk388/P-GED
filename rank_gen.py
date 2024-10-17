import copy
import numpy as np
import argparse
import networkx as nx
from collections import defaultdict
import os
import os
import json
from ranking_utils import RankingUtils, Ranking
from mallows import *
from ws_ranking import WeakSupRanking
import pickle


class DGtoDAG:
    def __init__(self, method='Greedy'):
        self.G = None
        self.weighted = False
        self.method = method
    def out_weight_sum(self, S, u):
        valid_edges = [e for e in self.G.out_edges(u, data=True) if e[1] in S]
        if self.weighted:
            return sum([e[2]['weight'] for e in valid_edges])
        else:
            return len(valid_edges)

    def order_to_fas(self, nodes_order):
        nodes_index = {node: i for i, node in enumerate(nodes_order)}
        feedback_arcs = []
        for node in nodes_order:
            for e in self.G.out_edges(node, data=False):
                if e[1] in nodes_index:
                    if nodes_index[e[1]] < nodes_index[node]:
                        feedback_arcs.append((node, e[1]))
        return feedback_arcs

    def greedy_fas(self, component):
        subG = self.G.subgraph(component).copy()
        ending_sequence = []
        starting_sequence = []
        while len(subG.nodes) > 0:
            while (cur_sinks := [node for node in subG.nodes if subG.out_degree(node) == 0]):
                ending_sequence += cur_sinks
                subG.remove_nodes_from(cur_sinks)
            while (cur_sources := [node for node in subG.nodes if subG.in_degree(node) == 0]):
                starting_sequence += cur_sources
                subG.remove_nodes_from(cur_sources)
            if len(subG.nodes) == 0:
                break
            cur_node_list = list(subG.nodes)
            deltas = []
            for node in cur_node_list:
                if self.weighted:
                    delta = sum([e[2]['weight'] for e in subG.out_edges(node, data=True)]) \
                            - sum([e[2]['weight'] for e in subG.in_edges(node, data=True)])
                else:
                    delta = subG.out_degree(node) - subG.in_degree(node)
                deltas.append(delta)
            greedy_node = cur_node_list[np.argmax(deltas)]

            starting_sequence.append(greedy_node)
            subG.remove_node(greedy_node)
        ending_sequence.reverse()
        nodes_order = starting_sequence + ending_sequence
        return self.order_to_fas(nodes_order)


    def get_dag(self, G):
        self.G = G
        self.weighted = nx.is_weighted(self.G)
        feedback_arcs = []

        scc = list(nx.strongly_connected_components(self.G))
        for component in scc:
            feedback_arcs += self.greedy_fas(component)
        for arc in feedback_arcs:
            self.G.remove_edge(*arc)
        return self.G, len(feedback_arcs)


def weighted_scoring_mMethod(rankings):
    node_scores = defaultdict(int)
    for ranking in rankings:
        for rank, nodes in enumerate(ranking):
            score = len(ranking) - rank
            for node in nodes:
                node_scores[node] += score
    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    combined_ranking = []
    current_score = None
    current_group = []
    for node, score in sorted_nodes:
        if score != current_score:
            if current_group:
                combined_ranking.append(current_group)
            current_group = [node]
            current_score = score
        else:
            current_group.append(node)
    if current_group:
        combined_ranking.append(current_group)
    return combined_ranking




def universalizing_weak_supervision(method_type= 'kemeny',rank_list=[]):
    rank_data = copy.deepcopy(rank_list)
    L = [[Ranking(rank) for rank in rank_data]]
    d = len(rank_data[0])  
    r_utils = RankingUtils(d)
    num_lfs = len(rank_data)  
    label_model = WeakSupRanking(r_utils)
    conf = {"train_method": "median_triplet_opt"}
    label_model.train(conf, L, num_lfs)
    if method_type == 'kemeny':
        mv_conf = {"train_method": "median_triplet_opt", "inference_rule": "kemeny"}
        Y = label_model.infer_ranking(mv_conf, L)
    elif  method_type == 'weighted_kemeny':
        uws_conf = {"train_method": "median_triplet_opt", "inference_rule": "weighted_kemeny"}
        Y = label_model.infer_ranking(uws_conf, L)
    elif  method_type == 'pairwise_majority':
        mv_conf = {"train_method": "median_triplet_opt", "inference_rule": "pairwise_majority"}
        Y = label_model.infer_ranking(mv_conf, L)
    elif  method_type == 'weighted_pairwise_majority':
        uws_conf = {"train_method": "median_triplet_opt", "inference_rule": "weighted_pairwise_majority"}
        Y = label_model.infer_ranking(uws_conf, L)
    aggregated_ranking = [rank.permutation for rank in Y][0]
    final_ranking_resul = [[i] for i in aggregated_ranking]
    return final_ranking_resul




def weighted_descendants_count_sort(G=None):
    descendants_count = {node: len(nx.descendants(G, node)) for node in G.nodes()}
    count_to_nodes = defaultdict(list)
    for node, count in descendants_count.items():
        count_to_nodes[count].append(node)
    sorted_counts = sorted(count_to_nodes.keys(), reverse=True)
    sorted_nodes = [sorted(count_to_nodes[count]) for count in sorted_counts]
    return sorted_nodes




def rank_ensemble_graph_to_rank(graph_list):
    rank_list = []
    for G in graph_list:
        sorted_nodes = weighted_descendants_count_sort(G)
        rank_list.append(sorted_nodes)
    return rank_list



def sort_nodes_by_weighted_out_degree(G):
    weighted_out_degrees = {node: sum(data['weight'] for _, _, data in G.out_edges(node, data=True)) for node in
                            G.nodes()}
    sorted_nodes = sorted(weighted_out_degrees.keys(), key=lambda node: weighted_out_degrees[node],
                          reverse=True)
    return sorted_nodes
def rank_ensemble_raw_graph_to_rank(raw_graph_list):
    all_dag_rank_list = []
    for G in raw_graph_list:
        sorted_nodes = sort_nodes_by_weighted_out_degree(G)
        sorted_nodes = [[i] for i in sorted_nodes]
        all_dag_rank_list.append(sorted_nodes)
    return all_dag_rank_list



def read_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
def update_rank_list(rank_list):
    final_r_l = []
    for r in rank_list:
        tmp_r = []
        for r_i in r:
            tmp_r.extend(r_i)
        final_r_l.append(tmp_r)
    return final_r_l

def DAG_graph_ensemble(graph_list):
    G_ensemble = nx.DiGraph()
    G_ensemble.add_nodes_from(range(len(model_dict)))
    for G in graph_list:
        for u, v, data in G.edges(data=True):
            if G_ensemble.has_edge(u, v):
                G_ensemble[u][v]['weight'] += data['weight']
            else:
                G_ensemble.add_edge(u, v, weight=data['weight'])
        edges_to_remove = []
        for u, v in list(G_ensemble.edges()):
            if (u, v) in edges_to_remove or (v, u) in edges_to_remove:
                continue
            if G_ensemble.has_edge(v, u):
                weight_uv = G_ensemble[u][v]['weight']
                weight_vu = G_ensemble[v][u]['weight']

                if weight_uv > weight_vu:
                    G_ensemble[u][v]['weight'] -= weight_vu
                    edges_to_remove.append((v, u))
                elif weight_uv < weight_vu:
                    G_ensemble[v][u]['weight'] -= weight_uv
                    edges_to_remove.append((u, v))
                else: 
                    edges_to_remove.append((u, v))
                    edges_to_remove.append((v, u))
            else:
                continue 
        G_ensemble.remove_edges_from(edges_to_remove)
    return G_ensemble



def combine_rankings(rankings,type='weight_score'):
    if isinstance(rankings[0][0], list):
        new_rankings = []
        for r_l in rankings:
            r_ll = []
            for kkk in r_l:
                r_ll.extend(kkk)
            new_rankings.append(r_ll)
    else:
        new_rankings = copy.deepcopy(rankings)
        tmp_rank = []
        for r_l in rankings:
            r_ll = []
            for kkk in r_l:
                r_ll.append([kkk])
            tmp_rank.append(r_ll)
    if type == 'weight_score':
        if isinstance(rankings[0][0], list):
            return weighted_scoring_mMethod(rankings)
        else:
            return weighted_scoring_mMethod(tmp_rank)
    elif  type in ['kemeny', 'weighted_kemeny', 'pairwise_majority', 'weighted_pairwise_majority']:
        return  universalizing_weak_supervision(method_type=type, rank_list=new_rankings)

def data_process(ensemble_type, task_name=None,eval_model=None, answer_model=None,all_model_list=None,rank_type=None,data_save_path=None):
    if eval_model != 'all':
        graph_list = read_data(file_path=f'{data_save_path}/{task_name}/{eval_model}/{answer_model}/graph_list.pkl')
    else:
        all_eval_model_dict = {}
        for tmp_eval_model in all_model_list:
            all_eval_model_dict[tmp_eval_model] = {}
            all_eval_model_dict[tmp_eval_model]['graph_list'] = read_data(file_path=f'{data_save_path}/{task_name}/{tmp_eval_model}/{answer_model}/graph_list.pkl')
            all_eval_model_dict[tmp_eval_model]['raw_graph_list'] =  read_data(file_path=f'{data_save_path}/{task_name}/{tmp_eval_model}/{answer_model}/raw_graph_list.pkl')
    if ensemble_type == 'rank_ensemble':
        if eval_model != 'all':
            rank_list = rank_ensemble_graph_to_rank(graph_list=graph_list)
        else:
            rank_list = []
            rank_dict = {}
            for mm in all_model_list:
                rank_dict[mm] = rank_ensemble_graph_to_rank(graph_list=all_eval_model_dict[mm]['graph_list'])
            for i in range(len(rank_dict[all_model_list[0]])):
                r_tmp_list = []
                for mm in all_model_list:
                    r_tmp_list.append(rank_dict[mm])
                rank_list.append(combine_rankings(rankings=r_tmp_list, type=rank_type))
        if isinstance(rank_list[0][0], list):
            rank_list = update_rank_list(rank_list=rank_list)
        else:
            rank_list_result_save[eval_model][f"{rank_type}"][ensemble_type] = rank_list
    elif ensemble_type == 'graph_ensemble':
        if eval_model != 'all':
            rank_list = []
            for g1 in graph_list:
                rank_list.append(weighted_descendants_count_sort(g1))
        else:
            graph_list = []
            rank_list = []
            for i in range(len(all_eval_model_dict['llama3-8b']['graph_list'])):
                g_tmp_list = []
                for mm in all_model_list:
                    g_tmp_list.append( all_eval_model_dict[mm]['graph_list'][i])
                g_ens = DAG_graph_ensemble(graph_list=g_tmp_list)
                graph_list.append(g_ens)
                rank_list.append(weighted_descendants_count_sort(g_ens))
        rank_list = update_rank_list(rank_list=rank_list)
        rank_list_result_save[eval_model][f"{rank_type}"][ensemble_type] = rank_list


parser = argparse.ArgumentParser(description="Example script with arguments")
parser.add_argument("--eval_model", type=str, default='llama3-8b', help="eval_model, ['llama3-8b', 'qwen2-7b', 'mistral-7b', 'all']")
parser.add_argument("--answer_model", type=str, default='qwen1.5-32b', help="answer_model, ['llama3-70b', 'qwen2-72b', 'mixtral-8x7b', 'qwen1.5-32b']")
parser.add_argument("--task_name", type=str, default='10k-ultra', help="task_name, ['gsm8k', 'human_eval', 'alpaca', 'math', 'truthful_qa', 'ultra']")
parser.add_argument("--rank_type", type=str, default='pairwise_majority', help="eval_model, ['weight_score', 'kemeny', 'weighted_kemeny', 'pairwise_majority', 'weighted_pairwise_majority']")
parser.add_argument("--ensemble_type", type=str, default='graph_ensemble', help="ensemble_type, ['graph_ensemble', 'rank_ensemble']")
args = parser.parse_args()
rank_type = args.rank_type
eval_model = args.eval_model
answer_model = args.answer_model
task_name = args.task_name
ensemble_type = args.ensemble_type
folder_path = 'xxx'
data_save_path = 'xxx'
# your all model
all_model_list = ['xxx']
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
file_path = os.path.join(folder_path, f'{rank_type}_rank_list.json')
dir_path = os.path.dirname(file_path)
model_dict = {}
for index, answer_id in enumerate(range(1, 9)):
    answer_model_name = answer_model
    model_name = f"{answer_model_name}_{task_name}_{answer_model_name}_{task_name}_{answer_id}"
    model_dict[index] = model_name
rank_list_result_save = {}
for mm in all_model_list+["all"]:
    rank_list_result_save[mm] = {
        f"{rank_type}": {
            "rank_ensemble": [],
            "graph_ensemble": [],
        }
    }

data_process(ensemble_type=ensemble_type, task_name=task_name, eval_model=eval_model, answer_model=answer_model, all_model_list=all_model_list,rank_type=rank_type)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(rank_list_result_save, file, ensure_ascii=False, indent=4)

