import networkx as nx
import numpy as np
from .crystgraph import quot_gen
from ase.io import read

def bet_cen(G):
    '''
    calculate betweenness centrality
    '''
    bet_dict = nx.betweenness_centrality(G)
    bet_list = [value for value in bet_dict.values()]
    bet_list = np.array(bet_list)

    return [np.max(bet_list), np.mean(bet_list), np.min(bet_list), np.std(bet_list)]

def calc_cons(G):
    '''
    calculate constraints
    '''
    cons_dict = nx.constraint(G)
    cons_list = [value for value in cons_dict.values()]
    cons_list = np.array(cons_list)

    return [np.max(cons_list), np.mean(cons_list), np.min(cons_list), np.std(cons_list)]

def degree_cen(G):
    '''
    calculate degree centrality
    '''
    degree_dict = nx.degree_centrality(G)
    degree_list = [value for value in degree_dict.values()]
    degree_list = np.array(degree_list)

    return [np.max(degree_list), np.mean(degree_list), np.min(degree_list), np.std(degree_list)]

def cycle_sums(G):
    """
    Return the cycle sums count of the crystal quotient graph G.
    G: networkx.MultiGraph
    Return: a (Nx3) matrix
    """
    SG = nx.Graph(G) # Simple graph, maybe with loop.
    cycle_simple = nx.simple_cycles(SG)
    cycle_count = np.zeros(125)

    for cyc in cycle_simple:
        if len(cyc) == 1:
            continue
        cycSum = np.zeros([3])
        for i in range(len(cyc)):
            vector = SG[cyc[i-1]][cyc[i]]['vector']
            direction = SG[cyc[i-1]][cyc[i]]['direction']
            cycDi = (cyc[i-1], cyc[i])
            if cycDi == direction:
                cycSum += vector
            elif cycDi[::-1] == direction:
                cycSum -= vector
            else:
                raise RuntimeError("Error in direction!")
        cycSum += 2
        index = int(cycSum[0] * 25 + cycSum[1] * 5 + cycSum[2])
        cycle_count[index] += 1
    
    for edge in SG.edges():
        # 两个点组成的环没办法通过cycle确定vector，必须通过edge确定
        # 一个点组成的环可能有多个，也需要通过edge确定vector
        numEdges = list(G.edges()).count(edge)
        if edge[0] == edge[1]:
            for i in range(numEdges):
                cycSum = G[edge[0]][edge[1]][i]['vector']
                cycSum += 2
                index = int(cycSum[0] * 25 + cycSum[1] * 5 + cycSum[2])
                cycle_count[index] += 1

        elif numEdges > 1:
            direction0 = G[edge[0]][edge[1]][0]['direction']
            vector0 = G[edge[0]][edge[1]][0]['vector']
            for j in range(1, numEdges):
                directionJ = G[edge[0]][edge[1]][j]['direction']
                vectorJ = G[edge[0]][edge[1]][j]['vector']
                if direction0 == directionJ:
                    cycSum = vector0 - vectorJ
                elif direction0[::-1] == directionJ:
                    cycSum = vector0 + vectorJ
                else:
                    raise RuntimeError("Error in direction!")
                cycSum += 2
                index = int(cycSum[0] * 25 + cycSum[1] * 5 + cycSum[2])
                cycle_count[index] += 1
    return cycle_count

def label_edge(G):
    '''
    labelled edge count.
    '''
    edge_count = np.zeros(34)   # 34 for XMnO, 50 for MP, 36 for c2db, 27 for 2dmd, 41 for oc20
    for edge in G.edges(data=True):
        _, _, data = edge
        vector = data["vector"] + 1
        index = int(vector[0] * 9 + vector[1] * 3 + vector[2])
        edge_count[index] += 1
    return edge_count

def label_edge_check(G):
    '''
    max labelled edge count.
    '''
    max_index = 0
    for edge in G.edges(data=True):
        _, _, data = edge
        vector = data["vector"] + 1
        index = int(vector[0] * 9 + vector[1] * 3 + vector[2])
        if index > max_index:
            max_index = index
    return max_index

def edge_degree(G):
    '''
    edge degree count.
    the number of edges adjacent to e.
    '''
    # TODO: why only 5?
    edge_degree_count = np.zeros(48)    # 48 for XMnO, 87 for MP, 54 for c2db, 17 low 2dmd, 93 for oc20
    edges_list = list(G.edges())
    for i in range(len(edges_list)):
        degree = -1
        edge = edges_list[i]
        for j in range(len(edges_list)):
            m, n = edges_list[j]
            if (m in edge) or (n in edge):
                degree += 1
        edge_degree_count[degree] += 1
    return edge_degree_count

def edge_degree_check(G):
    '''
    max edge degree count.
    the number of edges adjacent to e.
    '''
    max_degree = 0
    edges_list = list(G.edges())
    for i in range(len(edges_list)):
        degree = -1
        edge = edges_list[i]
        for j in range(len(edges_list)):
            m, n = edges_list[j]
            if (m in edge) or (n in edge):
                degree += 1
        if degree > max_degree:
            max_degree = degree
    return max_degree

def egv_cen(G):
    '''
    calculate eigenvector centrality
    '''
    egv_dict = nx.eigenvector_centrality(G)
    egv_list = [value for value in egv_dict.values()]
    egv_list = np.array(egv_list)

    return [np.mean(egv_list), np.std(egv_list)]

def energy_radius(G):
    '''
    calculate graph energy and spectral radius
    '''
    # 图能量为图的邻接矩阵的特征值绝对值之和
    # 谱半径为图的邻接矩阵的特征值绝对值的最大值
    A = nx.adjacency_matrix(G)
    eigenvalues = np.linalg.eigvals(A.toarray())
    graph_energy = np.sum(np.abs(eigenvalues))
    spectral_radius = np.max(np.abs(eigenvalues))

    return graph_energy, spectral_radius

def nei_regular(G):
    '''
    neighbor degree is the same or not
    '''
    is_neighbor_regular = 1
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        degrees = [G.degree(neighbor) for neighbor in neighbors]
        if len(set(degrees)) != 1:
            is_neighbor_regular = 0
            break
    return is_neighbor_regular

def node_regular(G):
    '''
    node degree is the same or not
    '''
    is_node_regular = 1
    degrees = [G.degree[node] for node in G.nodes()]
    if len(set(degrees)) != 1:
        is_node_regular = 0
    return is_node_regular

def lou_com(G):
    '''
    calculate louvain communities.
    '''
    lou_com = nx.community.louvain_communities(G, weight=None)
    lou_size = np.array([len(com) for com in lou_com])
    return [len(lou_size), np.max(lou_size), np.mean(lou_size), np.min(lou_size), np.std(lou_size)]

def max_eccen(G):
    '''
    maximum eccentricity
    '''
    ecc_dict = nx.eccentricity(G)
    ecc_list = [value for value in ecc_dict.values()]
    
    return max(ecc_list)

def nei_degree(G):
    '''
    neighbor degree count
    '''
    # TODO: why only 5?
    nei_degree_count = np.zeros(37) # 37 for XMnO, 64 for MP, 54 for c2db, 17 for 2dmd, 93 for oc20
    for node in G.nodes():
        for nei in G.neighbors(node):
            degree = G.degree[nei]
            nei_degree_count[degree] += 1
    return nei_degree_count

def nei_degree_check(G):
    '''
    max neighbor degree count
    '''
    # TODO: why only 5?
    max_degree = 0
    for node in G.nodes():
        for nei in G.neighbors(node):
            degree = G.degree[nei]
            if degree > max_degree:
                max_degree = degree
    return max_degree

def squ_clu(G):
    '''
    calculate square clustering
    '''
    clu_dict = nx.square_clustering(G)
    clu_list = [value for value in clu_dict.values()]
    clu_list = np.array(clu_list)

    return [np.max(clu_list), np.mean(clu_list), np.min(clu_list), np.std(clu_list)]

def feat_dict(muti_graph, connect=True):
    """
    connect为图是否连通的信息
    """
    simple_graph = nx.Graph(muti_graph)
    if len(simple_graph) == 1:
        ab_con = 0
    else:
        ab_con = nx.algebraic_connectivity(simple_graph, weight=None, method='tracemin_lu')    # algebraic connectivity
    if connect:
        ave_spl = nx.average_shortest_path_length(muti_graph)   # average shortest path lenght
        bary_size = len(nx.barycenter(muti_graph))          # barycenter size
        max_ecc = max_eccen(muti_graph)                     # maximum eccentricity
    else:
        ave_spl = 0
        bary_size = 0
        max_ecc = 0
    try:
        egv_data = egv_cen(simple_graph)                    # eigenvector centrality
    except:
        egv_data = [0, 0]
    bet_cen_data = bet_cen(muti_graph)
    cons_data = calc_cons(muti_graph)
    #cycle_sum_count = cycle_sums(muti_graph)           # cycle sum array
    degree_cen_data = degree_cen(muti_graph)
    edge_degree_count = edge_degree(muti_graph)         # edge degree count array
    global_efficiency = nx.global_efficiency(muti_graph)    # global efficiency
    graph_energy, spectral_radius \
        = energy_radius(muti_graph)                     # graph energy and spectral radius
    is_eulerian = int(nx.is_eulerian(muti_graph))       # is eulerian or not
    is_neighbor_regular = nei_regular(muti_graph)       # is neighbor regular or not
    is_node_regular = node_regular(muti_graph)          # is node regular or not
    is_planar = int(nx.is_planar(muti_graph))           # is planar
    #label_edge_count = label_edge(muti_graph)          # label edge array
    len_dominating_set = len(nx.dominating_set(muti_graph)) # len dominating set
    local_efficiency = nx.local_efficiency(muti_graph)  # local efficiency
    lou_com_data = lou_com(muti_graph)                  # louvain communities array
    nei_degree_count = nei_degree(muti_graph)           # neighbor degree count array
    num_bridge = len(list(nx.bridges(muti_graph)))      # number of bridges
    num_cycle_basis = muti_graph.size() - len(muti_graph) + 1   # number of cycle basis
    num_edge = len(list(muti_graph.edges()))            # number of edges
    num_node = len(list(muti_graph.nodes()))            # number of nodes
    squ_clu_data = squ_clu(muti_graph)                  # square clustering
    wiener_index = nx.wiener_index(muti_graph)          # wiener index

    data = {
        "algebraic_connectivity":       np.array([ab_con]),
        "average_shortest_path_length": np.array([ave_spl]),
        "barycenter_size":              np.array([bary_size]),
        "betweenness_centrality_data":  np.array(bet_cen_data),
        "constraints_data":             np.array(cons_data),
        #"cycle_sum_count":              cycle_sum_count,
        "degree_centrality_data":       np.array(degree_cen_data),
        "edge_degree_count":            edge_degree_count,
        "eigenvector_centrality_data":  np.array(egv_data),
        "global_efficiency":            np.array([global_efficiency]),
        "graph_energy":                 np.array([graph_energy]),
        "is_eulerian":                  np.array([is_eulerian]),
        "is_neighbor_regular":          np.array([is_neighbor_regular]),
        "is_node_regular":              np.array([is_node_regular]),
        "is_planar":                    np.array([is_planar]),
        #"labelled_edge_count":          label_edge_count,
        "len_dominating_set":           np.array([len_dominating_set]),
        "local_efficiency":             np.array([local_efficiency]),
        "louvain_communities_data":     np.array(lou_com_data),
        "maximum_eccentricity":         np.array([max_ecc]),
        "neighbor_degree_count":        nei_degree_count,
        "num_bridge":                   np.array([num_bridge]),
        "num_cycle_basis":              np.array([num_cycle_basis]),
        "num_edge":                     np.array([num_edge]),
        "num_node":                     np.array([num_node]),
        "spectral_radius":              np.array([spectral_radius]),
        "square_clustering_data":       np.array(squ_clu_data),
        "wiener_index":                 np.array([wiener_index]),
    }

    return data

if __name__ == "__main__":
    
    atoms = read("test.cif", index=0, format="cif")
    muti_graph, connect = quot_gen(atoms, "voronoi", 0.5)
    feat_dict(muti_graph, connect)
    '''
    G = nx.MultiGraph()
    G.add_edge(0, 0, vector=np.array([0, 0, 1]), direct=(0, 0))
    G.add_edge(0, 1, vector=np.array([0, 0, 0]), direct=(0, 1))
    G.add_edge(0, 2, vector=np.array([0, 0, 0]), direct=(0, 2))
    G.add_edge(1, 2, vector=np.array([0, 0, 0]), direct=(2, 1))
    G.add_edge(0, 2, vector=np.array([0, 1, 0]), direct=(2, 0))
    G.add_edge(1, 2, vector=np.array([1, 0, 0]), direct=(1, 2))
    G.add_edge(2, 2, vector=np.array([1, 0, 1]), direct=(2, 2))
    degrees = [G.degree[node] for node in G.nodes()]
    print(degrees)
    '''
