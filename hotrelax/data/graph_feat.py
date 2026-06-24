from typing import Callable, Dict, List, Optional

import networkx as nx
import numpy as np
from ase.io import read

from .crystgraph import quot_gen


FEATURE_REGISTRY: Dict[str, Callable] = {}


def register_feature(name: str):
    """
    Register a graph feature function with a string name.

    Args:
        name: Public feature name used for lookup.

    Returns:
        Decorator that stores the feature function in the registry.
    """
    def decorator(func: Callable):
        FEATURE_REGISTRY[name] = func
        return func

    return decorator


@register_feature("betweenness_centrality_data")
def bet_cen(G):
    """
    Calculate betweenness centrality summary statistics.

    Args:
        G: Input graph.

    Returns:
        Max/mean/min/std of betweenness centrality.
    """
    bet_dict = nx.betweenness_centrality(G)
    bet_list = [value for value in bet_dict.values()]
    bet_list = np.array(bet_list)

    return np.array([np.max(bet_list), np.mean(bet_list), np.min(bet_list), np.std(bet_list)])


@register_feature("constraints_data")
def calc_cons(G):
    """
    Calculate graph constraint summary statistics.

    Args:
        G: Input graph.

    Returns:
        Max/mean/min/std of node constraints.
    """
    cons_dict = nx.constraint(G)
    cons_list = [value for value in cons_dict.values()]
    cons_list = np.array(cons_list)

    return np.array([np.max(cons_list), np.mean(cons_list), np.min(cons_list), np.std(cons_list)])


@register_feature("degree_centrality_data")
def degree_cen(G):
    """
    Calculate degree centrality summary statistics.

    Args:
        G: Input graph.

    Returns:
        Max/mean/min/std of degree centrality.
    """
    degree_dict = nx.degree_centrality(G)
    degree_list = [value for value in degree_dict.values()]
    degree_list = np.array(degree_list)

    return np.array([np.max(degree_list), np.mean(degree_list), np.min(degree_list), np.std(degree_list)])


def cycle_sums(G):
    """
    Return the cycle sums count of the crystal quotient graph.

    Args:
        G: Input networkx MultiGraph.

    Returns:
        Cycle-count histogram indexed by cycle offset.
    """
    SG = nx.Graph(G)
    cycle_simple = nx.simple_cycles(SG)
    cycle_count = np.zeros(125)

    for cyc in cycle_simple:
        if len(cyc) == 1:
            continue
        cycSum = np.zeros([3])
        for i in range(len(cyc)):
            vector = SG[cyc[i - 1]][cyc[i]]["vector"]
            direction = SG[cyc[i - 1]][cyc[i]]["direction"]
            cycDi = (cyc[i - 1], cyc[i])
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
        numEdges = list(G.edges()).count(edge)
        if edge[0] == edge[1]:
            for i in range(numEdges):
                cycSum = G[edge[0]][edge[1]][i]["vector"]
                cycSum += 2
                index = int(cycSum[0] * 25 + cycSum[1] * 5 + cycSum[2])
                cycle_count[index] += 1

        elif numEdges > 1:
            direction0 = G[edge[0]][edge[1]][0]["direction"]
            vector0 = G[edge[0]][edge[1]][0]["vector"]
            for j in range(1, numEdges):
                directionJ = G[edge[0]][edge[1]][j]["direction"]
                vectorJ = G[edge[0]][edge[1]][j]["vector"]
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
    """
    Count labelled edges.

    Args:
        G: Input graph.

    Returns:
        Labelled-edge histogram.
    """
    edge_count = np.zeros(34)
    for edge in G.edges(data=True):
        _, _, data = edge
        vector = data["vector"] + 1
        index = int(vector[0] * 9 + vector[1] * 3 + vector[2])
        edge_count[index] += 1
    return edge_count


def label_edge_check(G):
    """
    Get the maximum labelled-edge index.

    Args:
        G: Input graph.

    Returns:
        Maximum labelled-edge index.
    """
    max_index = 0
    for edge in G.edges(data=True):
        _, _, data = edge
        vector = data["vector"] + 1
        index = int(vector[0] * 9 + vector[1] * 3 + vector[2])
        if index > max_index:
            max_index = index
    return max_index


@register_feature("edge_degree_count")
def edge_degree(G):
    """
    Count edge-adjacent degree values.

    Args:
        G: Input graph.

    Returns:
        Edge-degree histogram.
    """
    edge_degree_count = np.zeros(48)
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
    """
    Get the maximum edge-adjacent degree value.

    Args:
        G: Input graph.

    Returns:
        Maximum edge degree.
    """
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


@register_feature("eigenvector_centrality_data")
def egv_cen(G):
    """
    Calculate eigenvector centrality summary statistics.

    Args:
        G: Input graph.

    Returns:
        Mean and std of eigenvector centrality.
    """
    egv_dict = nx.eigenvector_centrality(G)
    egv_list = [value for value in egv_dict.values()]
    egv_list = np.array(egv_list)

    return np.array([np.mean(egv_list), np.std(egv_list)])


def energy_radius(G):
    """
    Calculate graph energy and spectral radius.

    Args:
        G: Input graph.

    Returns:
        Graph energy and spectral radius.
    """
    A = nx.adjacency_matrix(G)
    eigenvalues = np.linalg.eigvals(A.toarray())
    graph_energy = np.sum(np.abs(eigenvalues))
    spectral_radius = np.max(np.abs(eigenvalues))

    return graph_energy, spectral_radius


@register_feature("is_neighbor_regular")
def nei_regular(G):
    """
    Check whether all neighbors of each node have the same degree.

    Args:
        G: Input graph.

    Returns:
        Integer flag indicating neighbor-regularity.
    """
    is_neighbor_regular = 1
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        degrees = [G.degree(neighbor) for neighbor in neighbors]
        if len(set(degrees)) != 1:
            is_neighbor_regular = 0
            break
    return np.array([is_neighbor_regular])


@register_feature("is_node_regular")
def node_regular(G):
    """
    Check whether all nodes have the same degree.

    Args:
        G: Input graph.

    Returns:
        Integer flag indicating node-regularity.
    """
    is_node_regular = 1
    degrees = [G.degree[node] for node in G.nodes()]
    if len(set(degrees)) != 1:
        is_node_regular = 0
    return np.array([is_node_regular])


@register_feature("louvain_communities_data")
def lou_com(G):
    """
    Calculate Louvain community summary statistics.

    Args:
        G: Input graph.

    Returns:
        Community-count and community-size summary statistics.
    """
    lou_coms = nx.community.louvain_communities(G, weight=None)
    lou_size = np.array([len(com) for com in lou_coms])
    return np.array([len(lou_size), np.max(lou_size), np.mean(lou_size), np.min(lou_size), np.std(lou_size)])


def max_eccen(G):
    """
    Calculate maximum eccentricity.

    Args:
        G: Input graph.

    Returns:
        Maximum eccentricity.
    """
    ecc_dict = nx.eccentricity(G)
    ecc_list = [value for value in ecc_dict.values()]

    return max(ecc_list)


@register_feature("neighbor_degree_count")
def nei_degree(G):
    """
    Count neighbor degree values.

    Args:
        G: Input graph.

    Returns:
        Neighbor-degree histogram.
    """
    nei_degree_count = np.zeros(37)
    for node in G.nodes():
        for nei in G.neighbors(node):
            degree = G.degree[nei]
            nei_degree_count[degree] += 1
    return nei_degree_count


def nei_degree_check(G):
    """
    Get the maximum neighbor degree.

    Args:
        G: Input graph.

    Returns:
        Maximum neighbor degree.
    """
    max_degree = 0
    for node in G.nodes():
        for nei in G.neighbors(node):
            degree = G.degree[nei]
            if degree > max_degree:
                max_degree = degree
    return max_degree


@register_feature("square_clustering_data")
def squ_clu(G):
    """
    Calculate square clustering summary statistics.

    Args:
        G: Input graph.

    Returns:
        Max/mean/min/std of square clustering.
    """
    clu_dict = nx.square_clustering(G)
    clu_list = [value for value in clu_dict.values()]
    clu_list = np.array(clu_list)

    return np.array([np.max(clu_list), np.mean(clu_list), np.min(clu_list), np.std(clu_list)])


def compute_feature(name: str, muti_graph, simple_graph, connect: bool = True) -> np.ndarray:
    """
    Compute one feature by its string name.

    Args:
        name: Registered feature name.
        muti_graph: Input multigraph.
        simple_graph: Simple graph converted from the multigraph.
        connect: Whether graph-connected-only features should be evaluated.

    Returns:
        One feature array.
    """
    if name == "algebraic_connectivity":
        if len(simple_graph) == 1:
            return np.array([0])
        return np.array([
            nx.algebraic_connectivity(simple_graph, weight=None, method="tracemin_lu")
        ])
    if name == "average_shortest_path_length":
        value = nx.average_shortest_path_length(muti_graph) if connect else 0
        return np.array([value])
    if name == "barycenter_size":
        value = len(nx.barycenter(muti_graph)) if connect else 0
        return np.array([value])
    if name == "global_efficiency":
        return np.array([nx.global_efficiency(muti_graph)])
    if name == "graph_energy":
        graph_energy, _ = energy_radius(muti_graph)
        return np.array([graph_energy])
    if name == "is_eulerian":
        return np.array([int(nx.is_eulerian(muti_graph))])
    if name == "is_planar":
        return np.array([int(nx.is_planar(muti_graph))])
    if name == "len_dominating_set":
        return np.array([len(nx.dominating_set(muti_graph))])
    if name == "local_efficiency":
        return np.array([nx.local_efficiency(muti_graph)])
    if name == "maximum_eccentricity":
        value = max_eccen(muti_graph) if connect else 0
        return np.array([value])
    if name == "num_bridge":
        return np.array([len(list(nx.bridges(muti_graph)))])
    if name == "num_cycle_basis":
        return np.array([muti_graph.size() - len(muti_graph) + 1])
    if name == "num_edge":
        return np.array([len(list(muti_graph.edges()))])
    if name == "num_node":
        return np.array([len(list(muti_graph.nodes()))])
    if name == "spectral_radius":
        _, spectral_radius = energy_radius(muti_graph)
        return np.array([spectral_radius])
    if name == "wiener_index":
        return np.array([nx.wiener_index(muti_graph)])
    if name == "eigenvector_centrality_data":
        try:
            return FEATURE_REGISTRY[name](simple_graph)
        except Exception:
            return np.array([0, 0])
    if name in {
        "betweenness_centrality_data",
        "constraints_data",
        "degree_centrality_data",
        "edge_degree_count",
        "is_neighbor_regular",
        "is_node_regular",
        "louvain_communities_data",
        "neighbor_degree_count",
        "square_clustering_data",
    }:
        return FEATURE_REGISTRY[name](muti_graph)

    return FEATURE_REGISTRY[name](muti_graph)


if __name__ == "__main__":
    default_names = [
        "algebraic_connectivity",
        "average_shortest_path_length",
        "barycenter_size",
        "betweenness_centrality_data",
        "constraints_data",
        "degree_centrality_data",
        "edge_degree_count",
        "eigenvector_centrality_data",
        "global_efficiency",
        "graph_energy",
        "is_eulerian",
        "is_neighbor_regular",
        "is_node_regular",
        "is_planar",
        "len_dominating_set",
        "local_efficiency",
        "louvain_communities_data",
        "maximum_eccentricity",
        "neighbor_degree_count",
        "num_bridge",
        "num_cycle_basis",
        "num_edge",
        "num_node",
        "spectral_radius",
        "square_clustering_data",
        "wiener_index",
    ]

    atoms = read("test.cif", index=0, format="cif")
    muti_graph = quot_gen(atoms, "voronoi", 0.5)
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
