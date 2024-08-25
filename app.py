import streamlit as st
import requests
import streamlit.components.v1 as components
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Funções para cálculo de métricas e plotagem
def plot_heatmap_adjacency_matrix(G):
    adj_matrix = nx.adjacency_matrix(G).todense()
    adj_matrix = np.array(adj_matrix)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(adj_matrix, cmap='Blues', cbar=True, square=True)
    plt.title("Matriz de Adjacência")
    st.pyplot(plt)

def compute_and_plot_betweenness_centrality(G):
    betweenness_centrality = nx.betweenness_centrality(G)
    nx.set_node_attributes(G, betweenness_centrality, 'betweenness_centrality')
    betweenness_df = pd.DataFrame(G.nodes(data='betweenness_centrality'), columns=['node', 'betweenness_centrality'])
    betweenness_df = betweenness_df.sort_values(by='betweenness_centrality', ascending=False)
    num_nodes_to_inspect = 15
    betweenness_df[:num_nodes_to_inspect].plot(x='node', y='betweenness_centrality', kind='barh').invert_yaxis()
    st.pyplot(plt)

def compute_and_plot_weighted_degree(G, weight='weight', top_n=15):
    weighted_degrees = dict(nx.degree(G, weight=weight))
    nx.set_node_attributes(G, weighted_degrees, 'weighted_degree')
    weighted_degree_df = pd.DataFrame(G.nodes(data='weighted_degree'), columns=['node', 'weighted_degree'])
    weighted_degree_df = weighted_degree_df.sort_values(by='weighted_degree', ascending=False)
    weighted_degree_df[:top_n].plot(x='node', y='weighted_degree', color='orange', kind='barh').invert_yaxis()
    plt.title(f'Top {top_n} Nós por Grau Ponderado')
    st.pyplot(plt)

def compute_and_plot_degree_centrality(G, num_nodes_to_inspect=15):
    degrees = dict(nx.degree(G))
    nx.set_node_attributes(G, name='degree', values=degrees)
    degree_df = pd.DataFrame(G.nodes(data='degree'), columns=['node', 'degree'])
    degree_df = degree_df.sort_values(by='degree', ascending=False)
    degree_df[:num_nodes_to_inspect].plot(x='node', y='degree', kind='barh').invert_yaxis()
    plt.title('Top Nodes by Degree Centrality')
    st.pyplot(plt)

def compute_and_plot_clustering_coefficients(G):
    G_simple = nx.Graph(G)
    local_clustering = nx.clustering(G_simple)
    global_clustering = nx.transitivity(G_simple)
    nx.set_node_attributes(G, local_clustering, 'clustering_coefficient')
    clustering_df = pd.DataFrame(G.nodes(data='clustering_coefficient'), columns=['node', 'clustering_coefficient'])
    clustering_df = clustering_df.sort_values(by='clustering_coefficient', ascending=False)
    num_nodes_to_inspect = 15
    clustering_df[:num_nodes_to_inspect].plot(x='node', y='clustering_coefficient', kind='barh').invert_yaxis()
    st.write(f"Coeficiente de Clustering Global: {global_clustering}")
    st.pyplot(plt)

def compute_and_plot_eigenvector_centrality(G):
    G_simple = nx.Graph(G)
    eigenvector_centrality = nx.eigenvector_centrality(G_simple, max_iter=1000, tol=1e-06)
    nx.set_node_attributes(G, eigenvector_centrality, 'eigenvector_centrality')
    eigenvector_df = pd.DataFrame(G.nodes(data='eigenvector_centrality'), columns=['node', 'eigenvector_centrality'])
    eigenvector_df = eigenvector_df.sort_values(by='eigenvector_centrality', ascending=False)
    num_nodes_to_inspect = 15
    eigenvector_df[:num_nodes_to_inspect].plot(x='node', y='eigenvector_centrality', kind='barh').invert_yaxis()
    st.pyplot(plt)

def compute_and_plot_closeness_centrality(G):
    closeness_centrality = nx.closeness_centrality(G)
    nx.set_node_attributes(G, closeness_centrality, 'closeness_centrality')
    closeness_df = pd.DataFrame(G.nodes(data='closeness_centrality'), columns=['node', 'closeness_centrality'])
    closeness_df = closeness_df.sort_values(by='closeness_centrality', ascending=False)
    num_nodes_to_inspect = 15
    closeness_df[:num_nodes_to_inspect].plot(x='node', y='closeness_centrality', kind='barh').invert_yaxis()
    st.pyplot(plt)

def plot_degree_distribution(G):
    degrees = [d for n, d in G.degree()]
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), edgecolor='black')
    st.pyplot(plt)

def compute_assortativity(G):
    assortativity = nx.degree_assortativity_coefficient(G)
    st.write(f"Assortatividade Geral: {assortativity}")

def compute_sparsity_density(G):
    density = nx.density(G)
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    sparsity = 1 - density
    st.write(f"Densidade da Rede: {density}")
    st.write(f"Esparsidade da Rede: {sparsity}")

def compute_strongly_weakly_connected_components(G):
    if isinstance(G, nx.DiGraph):
        strongly_connected = list(nx.strongly_connected_components(G))
        weakly_connected = list(nx.weakly_connected_components(G))
        st.write(f"Componentes Conectados Fortemente: {strongly_connected}")
        st.write(f"Componentes Conectados Fracamente: {weakly_connected}")
    else:
        st.write("O grafo precisa ser dirigido para calcular componentes conectados fortemente.")

# Função para carregar o dataset e construir o grafo
def load_data_and_build_graph(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()

    G = nx.MultiDiGraph()

    for index, row in df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['quantity'])
    
    return G

# Função para exibir HTML
def display_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
        components.html(html_content, height=800, scrolling=True)
    else:
        st.write("Failed to load HTML content.")

# Interface do Streamlit
st.title("Análise de Redes - Métricas e Visualizações")

# Carregar dataset e construir o grafo
file_path = 'https://raw.githubusercontent.com/Dimitri-Code56/network_analysis/main/Datasets/global_arms_transfer_2000_2023.csv'  
G = load_data_and_build_graph(file_path)

# Seletor de métricas
metric = st.sidebar.selectbox(
    "Escolha uma métrica para visualizar",
    ["Matriz de Adjacência", "Betweenness Centrality", "Weighted Degree",
     "Degree Centrality", "Clustering Coefficient", "Eigenvector Centrality",
     "Closeness Centrality", "Distribuição de Grau", "Assortatividade", 
     "Esparsidade/Densidade", "Componentes Conectados Fortemente/Fracamente",
     "Visualização HTML"]
)

# Mostrar a métrica selecionada
if metric == "Matriz de Adjacência":
    plot_heatmap_adjacency_matrix(G)
elif metric == "Betweenness Centrality":
    compute_and_plot_betweenness_centrality(G)
elif metric == "Weighted Degree":
    compute_and_plot_weighted_degree(G)
elif metric == "Degree Centrality":
    compute_and_plot_degree_centrality(G)
elif metric == "Clustering Coefficient":
    compute_and_plot_clustering_coefficients(G)
elif metric == "Eigenvector Centrality":
    compute_and_plot_eigenvector_centrality(G)
elif metric == "Closeness Centrality":
    compute_and_plot_closeness_centrality(G)
elif metric == "Distribuição de Grau":
    plot_degree_distribution(G)
elif metric == "Assortatividade":
    compute_assortativity(G)
elif metric == "Esparsidade/Densidade":
    compute_sparsity_density(G)
elif metric == "Componentes Conectados Fortemente/Fracamente":
    compute_strongly_weakly_connected_components(G)
elif metric == "Visualização HTML":
    html_url = 'https://raw.githubusercontent.com/Dimitri-Code56/network_analysis/main/Vizualizations/comercio_todos%20(1).html'
    display_html(html_url)
