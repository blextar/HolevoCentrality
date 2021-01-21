from __future__ import division
import collections
import operator
import math as math
import networkx as nx
import numpy as np

def von_neumann_entropy(G,entropy_type="lap"):
    # normalise Laplacian by its trace to get a density matrix
    if entropy_type == "nlap":
        L = nx.normalized_laplacian_matrix(G).A/(nx.number_of_nodes(G))
    else:
        L = nx.laplacian_matrix(G).A/(2*nx.number_of_edges(G))
    eigs = np.real(np.linalg.eigvals(L))
    eigs[np.abs(eigs)<1e-8] = 1
    # the von nuemann entropy of the density matrix is the shannon entropy of its eigenvalues
    shannon_entropy = -sum([e*np.log(e)/np.log(2) for e in eigs])
    return shannon_entropy

def approximate_von_neumann_entropy(G,entropy_type="lap"):
    # quadratic approximation of the entropy: -tr(rho log(rho)) = tr(rho (1-rho))
    n = nx.number_of_nodes(G)
    m = nx.number_of_edges(G)
    if entropy_type == "nlap":
        avne = 1 - 1/n
        for e in G.edges():
            avne -= 1/(n*n)*1/(G.degree(e[0])*G.degree(e[1]))
    else:
        avne = 1
        for v in G.nodes():
            avne -= 1/(4*m*m)*G.degree(v)*(G.degree(v)+1)
    return avne

def compute_holevo_edge_centrality(G,edge,entropy_type="lap"):
    m = nx.number_of_edges(G)
    # von neumann entropy of the original graph
    hec = von_neumann_entropy(G,entropy_type)
    # von neumann entropy of the original graph
    hec_scaled = von_neumann_entropy(G,entropy_type)
    # approximate vne of the original graph
    hec_approx = approximate_von_neumann_entropy(G,entropy_type)
    # create a copy of G without the edge e
 
    Ge = G.copy()
    try:
        Ge.remove_edge(*edge)
    except Exception as e:
        print(e)
        return
    #exact
    hec -= (m-1)/m*von_neumann_entropy(Ge,entropy_type)
    # without scaling factor
    hec_scaled -= von_neumann_entropy(Ge,entropy_type)
    # approximate
    hec_approx -= approximate_von_neumann_entropy(Ge,entropy_type)
    return hec, hec_scaled, hec_approx

def holevo_edge_centrality(G,entropy_type="lap"):
    hec, hec_scaled, hec_approx = {}, {}, {}
    # for each edge of G, compute the holevo centrality
    for e in G.edges():
        hec[e], hec_scaled[e], hec_approx[e] = compute_holevo_edge_centrality(G,e,entropy_type)
    return hec, hec_scaled, hec_approx
    
def holevo_node_centrality(G,entropy_type="lap"):
    hc = {}
    # von neumann entropy of the original graph
    vneG = von_neumann_entropy(G,entropy_type)
    m = nx.number_of_edges(G)
    for node in G.nodes():
        # von neumann entropy of the egonet
        EGO = nx.star_graph(G.degree(node))
        vneEGO = von_neumann_entropy(EGO,entropy_type)
        # von neumann entropy of the rest
        REST = G.copy()
        REST.remove_node(node)
        vneREST = von_neumann_entropy(REST,entropy_type)
        hc[node] = vneG - (nx.number_of_edges(EGO)/m*vneEGO + nx.number_of_edges(REST)/m*vneREST)
    return hc
