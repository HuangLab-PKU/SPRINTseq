import networkx as nx
import numpy as np
from tqdm import tqdm

def get_var(mis):
    signal_stats = np.zeros((len(mis[0]),2))
    for i,a in enumerate(zip(*mis)):
        signal_stats[i,:] = a.count('C'),a.count('T')
    return np.var(signal_stats),signal_stats

def get_min_var_mis(graph_file,trial_num):
    min_var = 1e9
    C = nx.read_adjlist(graph_file)
    for _ in tqdm(range(trial_num)):
        s = nx.maximal_independent_set(C)
        var,_ = get_var(s)
        if var < min_var:
            min_var_mis = s
            min_var = var 
    return min_var_mis

if __name__ == "__main__":
    mis = get_min_var_mis('/mnt/gpfs/Users/jiangmengcheng/HanWuji/barcode_design/test_graph_210809_G58.txt',10000)
    with open('/mnt/gpfs/Users/jiangmengcheng/HanWuji/barcode_design/min_var_mis_0809_G58.txt','w') as f:
        f.write('\n'.join(mis))
    v,stats = get_var(mis)
    print(f'Final variance: {v}. Stats:\n {stats}')

