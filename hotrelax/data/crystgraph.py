import itertools
import networkx as nx
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import MinimumDistanceNN, EconNN, VoronoiNN


def get_neighbors_of_site_with_index(struct, n, approach, delta, cutoff):

    if approach == "min_dist":
        neighs_list = MinimumDistanceNN(tol=delta, cutoff=cutoff).get_nn_info(struct, n)
    if approach == "econ":
        neighs_list = EconNN(tol=delta, cutoff=10).get_nn_info(struct, n)
    if approach == "voronoi":
        neighs_list = VoronoiNN(tol=delta, cutoff=cutoff).get_nn_info(struct, n)

    return neighs_list

def quot_gen(ats, approach, delta):
    struct = AseAtomsAdaptor.get_structure(ats)
    cutoff = ats.cell.cellpar()[:3].mean()
    G = nx.MultiGraph()
    for i in range(len(struct)):
        G.add_node(i)

    for i in range(len(struct)):
        site = struct[i]
        neighs_list = get_neighbors_of_site_with_index(struct,i,approach,delta,cutoff)
        for nn in neighs_list:
            j = nn['site_index']
            if i <= j:
                G.add_edge(i,j, vector=np.array(nn['image']), direction=(i,j))

    if nx.number_connected_components(G) != 1:
        return False
    else:
        return G

def cycle_sums(G):
    """
    Return the cycle basis and offset of the crystal quotient graph G.
    G: networkx.MultiGraph
    Return: a (Nx3) matrix
    """
    SG = nx.Graph(G) # Simple graph, maybe with loop.
    length = G.size()
    cycBasis_simple = nx.cycle_basis(SG)
    cycleBasis = np.empty((0, length))
    cycleSums = np.empty((0, 3))

    for cyc in cycBasis_simple:
        cycSum = np.zeros([3])
        cycBasis = np.zeros([length])
        for i in range(len(cyc)):
            vector = SG[cyc[i-1]][cyc[i]]['vector']
            direction = SG[cyc[i-1]][cyc[i]]['direction']
            index = SG[cyc[i-1]][cyc[i]]['index']
            cycDi = (cyc[i-1], cyc[i])
            if cycDi == direction:
                cycBasis[index] = 1
                cycSum += vector
            elif cycDi[::-1] == direction:
                cycBasis[index] = -1
                cycSum -= vector
            else:
                raise RuntimeError("Error in direction!")
        cycleBasis = np.concatenate((cycleBasis, cycBasis[None, :]))
        cycleSums = np.concatenate((cycleSums, cycSum[None, :]))

    for edge in SG.edges():
        numEdges = list(G.edges()).count(edge)
        if numEdges > 1:
            direction0 = G[edge[0]][edge[1]][0]['direction']
            vector0 = G[edge[0]][edge[1]][0]['vector']
            index0 = G[edge[0]][edge[1]][0]['index']
            for j in range(1, numEdges):
                cycBasis = np.zeros([length])
                directionJ = G[edge[0]][edge[1]][j]['direction']
                vectorJ = G[edge[0]][edge[1]][j]['vector']
                indexJ = G[edge[0]][edge[1]][j]['index']
                if direction0 == directionJ:
                    cycBasis[[index0, indexJ]] = [1, -1]
                    cycSum = vector0 - vectorJ
                elif direction0[::-1] == directionJ:
                    cycBasis[[index0, indexJ]] = [1, 1]
                    cycSum = vector0 + vectorJ
                else:
                    raise RuntimeError("Error in direction!")
                cycleBasis = np.concatenate((cycleBasis, cycBasis[None, :]))
                cycleSums = np.concatenate((cycleSums, cycSum[None, :]))
    
    if len(cycleBasis) != length - len(G) + 1:
        raise RuntimeError("Error in finding linearly independent cycle basis!")

    return cycleBasis, cycleSums

def cocycle_sums(G):
    length = G.size()
    cocycleBasis = np.empty((0, length))
    cocycle_atoms = []
    for node in G.nodes():
        cocycBasis = np.zeros([length])
        for edge in G.edges(node, data=True):
            _, _, data = edge
            direction = data["direction"]
            index = data["index"]
            if direction[0] == node and direction[1] != node:
                cocycBasis[index] = 1
            elif direction[0] != node and direction[1] == node:
                cocycBasis[index] = -1
            else:
                cocycBasis[index] = 0
        cocycleBasis_new = np.concatenate((cocycleBasis, cocycBasis[None, :]))
        if np.linalg.matrix_rank(cocycleBasis_new) == cocycleBasis_new.shape[0]:
            cocycleBasis = cocycleBasis_new
            cocycle_atoms.append(node)
    if len(cocycleBasis) != len(G) - 1:
        raise RuntimeError("Error in finding linearly independent cocycle basis!")
    
    for node in G.nodes():
        if node not in cocycle_atoms:
            cocycBasis = np.zeros([length])
            for edge in G.edges(node, data=True):
                _, _, data = edge
                direction = data["direction"]
                index = data["index"]
                if direction[0] == node and direction[1] != node:
                    cocycBasis[index] = 1
                elif direction[0] != node and direction[1] == node:
                    cocycBasis[index] = -1
                else:
                    cocycBasis[index] = 0
            cocycleBasis = np.concatenate((cocycleBasis, cocycBasis[None, :]))
            cocycle_atoms.append(node)
    return cocycleBasis, cocycle_atoms

def get_coord(Nat, edges, graph, cell=None):
    paths = nx.single_source_shortest_path(graph, 0)
    pos = np.zeros((Nat, 3))
    offset = np.zeros((Nat, 3))
    for node in paths.keys():
        if node == 0:
            continue
        else:
            path = paths[node]
            for l in range(len(path)-1):
                index = graph[path[l]][path[l+1]][0]['index']
                direction = graph[path[l]][path[l+1]][0]['direction']
                vector = graph[path[l]][path[l+1]][0]['vector']
                if (path[l], path[l+1]) == direction:
                    pos[node] += edges[index]
                    offset[node] -= vector
                else:
                    pos[node] -= edges[index]
                    offset[node] += vector
    if cell is not None:
        pos += np.dot(offset, cell)
    else:
        pos += offset
    return pos


class HotRelax:
    """
    Extract forces and stress information from dataset
    for hotpp training and validation by barycentric embedding.
    """
    def __init__(self, ats_unrelax, ats_relax):
        self.pos_unrelax = ats_unrelax.get_positions()
        self.spos_unrelax = ats_unrelax.get_scaled_positions(wrap=False)
        self.pos_relax = ats_relax.get_positions()
        self.spos_relax = ats_relax.get_scaled_positions(wrap=False)
        self.cell_unrelax = np.array(ats_unrelax.get_cell())
        self.cell_relax = np.array(ats_relax.get_cell())
        self.edges_bary = None      # barycentric edges

        self.forces = None
        self.rotate_matrix = None   # rotate matrix to make strain symmetric
        self.strain = None
        self.edges_pred = None      # predicted edges
        self.pos_pred = None
        self.cell_pred = None

        self.pos_edge = None
        self.pos_offset = None

    def read_graph(self, ats_unrelax, approach="voronoi", delta=0.5, ori_G=None):
        """
        Order of edges in graph building process can differ from graph reading process.
        Here we take the order of edges in graph reading process and add index.
        """
        if not ori_G:
            ori_G = quot_gen(ats_unrelax, approach, delta)

        self.graph = nx.MultiGraph()
        self.edge_index = np.empty((0, 2))
        self.cell_offsets = np.empty((0 ,3))
        for node in range(len(ats_unrelax)):
            self.graph.add_node(node)
        for n, edge in enumerate(ori_G.edges(data=True)):
            _, _, data = edge
            i, j = data['direction']
            self.edge_index = np.concatenate((self.edge_index, np.array([[j, i]])), axis=0)
            self.cell_offsets = np.concatenate((self.cell_offsets, data["vector"][None, :]), axis=0)
            self.graph.add_edge(i, j, vector=data['vector'], direction=(i, j), index=n)
        self.edge_index = self.edge_index.T
        return True
    
    def get_pos_edge(self):
        '''
        简单图中的每个坐标矢量 OA_i 对应的边矩阵及其 offset
        '''
        paths = nx.single_source_shortest_path(self.graph, 0)
        M = self.graph.size()
        Nat = len(self.pos_unrelax)
        self.pos_edge = np.zeros((Nat, M))
        self.pos_offset = np.zeros((Nat, 3))     # edges = pos[j] + offset - pos[i]
        for node in paths.keys():
            if node == 0:
                continue
            else:
                path = paths[node]
                for l in range(len(path)-1):
                    index = self.graph[path[l]][path[l+1]][0]['index']
                    direction = self.graph[path[l]][path[l+1]][0]['direction']
                    vector = self.graph[path[l]][path[l+1]][0]['vector']
                    if (path[l], path[l+1]) == direction:
                        self.pos_edge[node][index] += 1
                        self.pos_offset[node] += vector      # 为了和神经网络图保持一致，需要变号
                    else:
                        self.pos_edge[node][index] -= 1
                        self.pos_offset[node] -= vector

    def get_graph_matrix(self, edge_index):
        '''
        计算从简单图到神经网络图的边转移矩阵及 offset 的差
        Args:
            pos_edge: 简单图中每个坐标矢量对应的边矩阵
            offset: 简单图中每个坐标矢量对应的 offset
            edge_index: 神经网络图中边对应的起始点和终点矩阵
        '''
        j, i = edge_index
        return self.pos_edge[j] - self.pos_edge[i], self.pos_offset[j] - self.pos_offset[i]

    def get_cycle(self):
        self.cycle_basis, self.cycle_offset = cycle_sums(self.graph)

    def get_cocycle(self):
        self.cocycle_basis, self.cocycle_atoms = cocycle_sums(self.graph)
        self.cocycle_offset = np.zeros((len(self.cocycle_basis), 3))     # required by barycentric embedding

    def get_barycentric_edge(self):
        """
        Get barycentric edges based on cycle and cocycles.
        """
        self.get_cycle()
        self.get_cocycle()
        coeff = np.concatenate((self.cycle_basis, self.cocycle_basis[:-1]), axis=0)
        
        if np.linalg.matrix_rank(coeff) != coeff.shape[0]:
            raise RuntimeError("Error in finding linearly independent cycle-cocycle basis!")

        result = np.concatenate((self.cycle_offset, self.cocycle_offset[:-1]), axis=0)
        self.edges_bary = np.dot(np.linalg.inv(coeff), result)

    def get_barycentric_pos(self):
        """
        Get barycentric edges based on cycle and cocycles.
        """
        self.pos_bary = get_coord(len(self.pos_unrelax), self.edges_bary, self.graph)

    def get_relaxed_offset(self, pos_pred=None):
        """
        Relaxed structures may cross the cell boundary.
        We choose the closest atoms to the unrelaxed structure atoms
        among all the atoms of the neighboring cell.
        """
        offset_all = np.array(list(itertools.product(range(-1, 2),range(-1, 2),range(-1, 2))))
        pos_relax_super = self.pos_relax[:, None] + np.dot(offset_all, self.cell_relax)
        if pos_pred is not None:
            delta_pos = np.linalg.norm(pos_relax_super - pos_pred[:, None], axis=2)
        else:
            delta_pos = np.linalg.norm(pos_relax_super - self.pos_unrelax[:, None], axis=2)
        offset = offset_all[np.argmin(delta_pos, axis=1)]
        self.spos_relax += offset
        self.pos_relax += np.dot(offset, self.cell_relax)
    
    def get_train_strain(self):
        """
        Get (strain + I) from unrelaxed cell to relaxed cell.
        Perform singular value decomposition on (strain + I) to get symmetric (strain + I).
        """
        strain = np.dot(np.linalg.inv(self.cell_unrelax), self.cell_relax)
        W, s, VT = np.linalg.svd(strain)
        self.rotate_matrix = np.dot(W, VT).T
        self.strain = np.dot(strain, self.rotate_matrix) - np.eye(3)

    def get_train_forces_rb(self):
        """
        Take the difference between relaxed edges and barycentric edges as deviation.
        Compute the product of cocycle and deviation as forces.
        """
        edges_relax = np.zeros((self.graph.size(), 3))
        for edge in self.graph.edges(data=True):
            _, _, data = edge
            i, j = data["direction"]
            n = data["index"]
            edges_relax[n][:] = self.spos_relax[j] - self.spos_relax[i] + data["vector"]

        coeff = np.concatenate((self.cycle_basis, self.cocycle_basis), axis=0)
        forces = np.dot(coeff, edges_relax - self.edges_bary)[-len(self.pos_relax):]
        sorted_indices = np.argsort(self.cocycle_atoms)
        forces = forces[sorted_indices]     # make force vector's order coincide with atomic order
        self.forces = np.dot(forces, self.cell_unrelax)

    def get_train_forces_ru_cartesion(self):
        """
        Take the difference between relaxed edges and unrelaxed edges as deviation.
        Compute the product of cocycle and deviation as forces.
        """
        edges_relax = np.zeros((self.graph.size(), 3))
        edges_unrelax = np.zeros((self.graph.size(), 3))
        for edge in self.graph.edges(data=True):
            _, _, data = edge
            i, j = data["direction"]
            n = data["index"]
            edges_relax[n][:] = self.pos_relax[j] - self.pos_relax[i] + np.dot(data["vector"], self.cell_relax)
            edges_unrelax[n][:] = self.pos_unrelax[j] - self.pos_unrelax[i] + np.dot(data["vector"], self.cell_unrelax)
        
        coeff = np.concatenate((self.cycle_basis, self.cocycle_basis), axis=0)
        forces_unrelax = np.dot(coeff, edges_unrelax)[-len(self.pos_relax):]
        forces_relax = np.dot(coeff, edges_relax)[-len(self.pos_relax):]
        sorted_indices = np.argsort(self.cocycle_atoms)
        self.forces_unrelax = forces_unrelax[sorted_indices]     # make force vector's order coincide with atomic order
        self.forces_relax = forces_relax[sorted_indices]
        self.forces = self.forces_relax - self.forces_unrelax
        self.cocycle_basis_sorted = self.cocycle_basis[sorted_indices]
    
    def get_train_forces_ru_frac(self):
        """
        Take the difference between relaxed edges and unrelaxed edges as deviation.
        Compute the product of cocycle and deviation as forces.
        """
        edges_relax = np.zeros((self.graph.size(), 3))
        edges_unrelax = np.zeros((self.graph.size(), 3))
        for edge in self.graph.edges(data=True):
            _, _, data = edge
            i, j = data["direction"]
            n = data["index"]
            edges_relax[n][:] = self.spos_relax[j] - self.spos_relax[i] + data["vector"]
            edges_unrelax[n][:] = self.spos_unrelax[j] - self.spos_unrelax[i] + data["vector"]
        
        coeff = np.concatenate((self.cycle_basis, self.cocycle_basis), axis=0)
        forces_unrelax = np.dot(coeff, edges_unrelax)[-len(self.spos_relax):]
        forces_relax = np.dot(coeff, edges_relax)[-len(self.spos_relax):]
        sorted_indices = np.argsort(self.cocycle_atoms)
        forces_unrelax = forces_unrelax[sorted_indices]     # make force vector's order coincide with atomic order
        forces_relax = forces_relax[sorted_indices]
        self.forces_unrelax = np.dot(forces_unrelax, self.cell_unrelax)
        self.forces_relax = np.dot(forces_relax, self.cell_unrelax)
        self.forces = self.forces_relax - self.forces_unrelax
        self.cocycle_basis_sorted = self.cocycle_basis[sorted_indices]
    
    def get_pred_pos_rb(self, forces_pred, cell_pred):
        """Inverse of get_train_forces_rb."""
        forces_frac = np.dot(forces_pred, np.linalg.inv(self.cell_unrelax))
        results_pred = np.concatenate((np.zeros_like(self.cycle_offset), forces_frac[self.cocycle_atoms][:-1]), axis=0)
        coeff = np.concatenate((self.cycle_basis, self.cocycle_basis[:-1]), axis=0)
        self.edges_pred = np.dot(np.linalg.inv(coeff), results_pred) + self.edges_bary
        pos_frac = get_coord(len(self.pos_relax), self.edges_pred, self.graph)
        self.pos_pred = np.dot(pos_frac, cell_pred)
    
    def get_pred_pos_ru_cartesion(self, forces_pred, cell_pred):
        """Inverse of get_train_forces_ru."""
        edges_unrelax = np.zeros((self.graph.size(), 3))
        for edge in self.graph.edges(data=True):
            _, _, data = edge
            i, j = data["direction"]
            n = data["index"]
            edges_unrelax[n][:] = self.pos_unrelax[j] - self.pos_unrelax[i] + np.dot(data["vector"], self.cell_unrelax)
        
        results_unrelax = np.dot(np.concatenate((self.cycle_basis, self.cocycle_basis), axis=0), edges_unrelax)
        results_pred = np.concatenate((np.dot(self.cycle_offset, self.cell_relax), \
                        forces_pred[self.cocycle_atoms] + results_unrelax[-len(self.pos_unrelax):]), axis=0)
        coeff = np.concatenate((self.cycle_basis, self.cocycle_basis[:-1]), axis=0)
        self.edges_pred = np.dot(np.linalg.inv(coeff), results_pred[:-1])
        self.pos_pred = get_coord(len(self.pos_relax), self.edges_pred, self.graph, cell_pred)
    
    def get_pred_pos_ru_frac(self, forces_pred, cell_pred):
        """Inverse of get_train_forces_ru."""
        forces_frac = np.dot(forces_pred, np.linalg.inv(self.cell_unrelax))
        edges_unrelax = np.zeros((self.graph.size(), 3))
        for edge in self.graph.edges(data=True):
            _, _, data = edge
            i, j = data["direction"]
            n = data["index"]
            edges_unrelax[n][:] = self.spos_unrelax[j] - self.spos_unrelax[i] + data["vector"]
        
        results_unrelax = np.dot(np.concatenate((self.cycle_basis, self.cocycle_basis), axis=0), edges_unrelax)
        results_pred = np.concatenate((self.cycle_offset, forces_frac[self.cocycle_atoms] + results_unrelax[-len(self.spos_unrelax):]), axis=0)
        coeff = np.concatenate((self.cycle_basis, self.cocycle_basis[:-1]), axis=0)
        self.edges_pred = np.dot(np.linalg.inv(coeff), results_pred[:-1])
        pos_frac = get_coord(len(self.pos_relax), self.edges_pred, self.graph)
        self.pos_pred = np.dot(pos_frac, cell_pred)
    
    def get_pred_cell(self, strain_pred):
        self.cell_pred = np.dot(self.cell_unrelax, strain_pred)