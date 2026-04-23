from itertools import product
import numpy as np
import networkx as nx
import time
import os

symbs = ['G','C','T']
def str_from_symbs(symbs,k):
    result = []
    words=list(product(symbs,repeat=k))
    for word in words:
        result.append(''.join(word))
    return result

def hamming(x,y):
    hamm = 0
    for a,b in zip(x,y):
        if a!=b:
            hamm += 1
    return hamm

word_list = str_from_symbs(symbs,10)
word_list = [s for s in word_list if 5 <= s.count('G') <= 8 and s.count('C') >= 1]
word_list = [s for s in word_list if ('C' in s[:4] or 'T' in s[:4])]
G = nx.Graph()
start = time.time()
progress = 1
for i in range(len(word_list)):
    if i*100/len(word_list) >= progress:
        progress += 1
        lap = time.time()
        os.system('echo "{}% completed. {}s elapsed."'.format(progress,lap-start))
    seq_x = word_list[i]
    for seq_y in word_list[i+1:]:
        if hamming(seq_x,seq_y) >= 3:
            G.add_edge(seq_x,seq_y)

C = nx.complement(G)
nx.write_adjlist(C,'/mnt/gpfs/Users/jiangmengcheng/HanWuji/barcode_design/test_graph_210809_1_G58.txt')
