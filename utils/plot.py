import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

def _plotLastMile(model, x, y, I, J, c, S, k, a, case_idx=1, save_fig=False, filename=None):
    '''
    Visualize the LastMile delivery optimization model as a network graph
    
    Parameters:
    -----------
    model : pySCIPopt model object
    x : decision variables for nodes to packages
    y : decision variables for Service Center
    I : set of delivery nodes
    J : set of packages
    c : fixed cost per package from node i
    S : average cost per package from Service Center
    k : capacity of each node
    a : availability matrix (node i can deliver package j)
    case_idx : case index for title
    save_fig : whether to save the figure
    filename : filename for saving (if save_fig=True)
    '''
    
    # Create the graph
    G = nx.Graph()
    
    # Define positions for better layout
    pos = {}
    
    # Add Service Center node
    SC_node = 'SC'
    G.add_node(SC_node)
    pos[SC_node] = (0, 0)  # Center position
    
    # Add delivery nodes in a circle around SC
    n_nodes = len(I)
    if n_nodes > 0:
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        radius = 3
        
        for idx, i in enumerate(I):
            node_name = f'Node_{i}'
            G.add_node(node_name)
            x_coord = radius * np.cos(angles[idx])
            y_coord = radius * np.sin(angles[idx])
            pos[node_name] = (x_coord, y_coord)
    
    # Add package nodes in an outer ring
    n_packages = len(J)
    if n_packages > 0:
        package_angles = np.linspace(0, 2*np.pi, n_packages, endpoint=False)
        package_radius = 5
        
        for idx, j in enumerate(J):
            package_name = f'Pkg_{j}'
            G.add_node(package_name)
            x_coord = package_radius * np.cos(package_angles[idx])
            y_coord = package_radius * np.sin(package_angles[idx])
            pos[package_name] = (x_coord, y_coord)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Draw nodes with different styles
    # Service Center
    nx.draw_networkx_nodes(G, pos, nodelist=[SC_node], 
                          node_color='gold', node_size=1500, 
                          node_shape='s', ax=ax, alpha=0.9)
    
    # Delivery nodes
    delivery_nodes = [f'Node_{i}' for i in I]
    node_colors = plt.cm.Set3(np.linspace(0, 1, len(I)))
    
    for idx, node in enumerate(delivery_nodes):
        nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                              node_color=[node_colors[idx]], node_size=800, 
                              node_shape='o', ax=ax, alpha=0.8)
    
    # Package nodes
    package_nodes = [f'Pkg_{j}' for j in J]
    nx.draw_networkx_nodes(G, pos, nodelist=package_nodes, 
                          node_color='lightcoral', node_size=400, 
                          node_shape='^', ax=ax, alpha=0.7)
    
    # Draw edges for optimal assignments
    edge_colors = []
    edge_styles = []
    edge_widths = []
    
    # Edges from delivery nodes to packages (x variables)
    for i in I:
        for j in J:
            if model.getVal(x[i, j]) == 1:
                G.add_edge(f'Node_{i}', f'Pkg_{j}')
                edge_colors.append('blue')
                edge_styles.append('-')
                edge_widths.append(3)
    
    # Edges from SC to packages (y variables)
    for j in J:
        if model.getVal(y[j]) == 1:
            G.add_edge(SC_node, f'Pkg_{j}')
            edge_colors.append('red')
            edge_styles.append('--')
            edge_widths.append(2)
    
    # Draw all edges
    edges = list(G.edges())
    if edges:
        for idx, edge in enumerate(edges):
            nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                                  edge_color=edge_colors[idx], 
                                  style=edge_styles[idx],
                                  width=edge_widths[idx], ax=ax, alpha=0.7)
    
    # Add labels
    # Node labels with capacity and cost info
    node_labels = {}
    node_labels[SC_node] = f'SC\n(Cost: {S})'
    
    for i in I:
        node_labels[f'Node_{i}'] = f'N{i}\nCap:{k[i]}\nCost:{c[i]}'
    
    for j in J:
        node_labels[f'Pkg_{j}'] = f'P{j}'
    
    nx.draw_networkx_labels(G, pos, node_labels, font_size=8, ax=ax)
    
    # Add title and information
    obj_val = model.getObjVal()
    total_packages = len(J)
    packages_from_nodes = sum(model.getVal(x[i, j]) for i in I for j in J)
    packages_from_sc = sum(model.getVal(y[j]) for j in J)
    
    title = f'Last Mile Delivery Optimization - Case {case_idx}\n'
    title += f'Objective Value: {obj_val:.2f} | '
    title += f'Packages from Nodes: {packages_from_nodes:.0f} | '
    title += f'Packages from SC: {packages_from_sc:.0f}'
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color='gold', label='Service Center'),
        mpatches.Patch(color='lightblue', label='Delivery Nodes'),
        mpatches.Patch(color='lightcoral', label='Packages'),
        plt.Line2D([0], [0], color='blue', linewidth=3, label='Node → Package'),
        plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='SC → Package')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Add constraint satisfaction info as text box
    info_text = "Constraint Status:\n"
    info_text += f"• Total packages delivered: {packages_from_nodes + packages_from_sc:.0f}/{total_packages}\n"
    
    # Check capacity constraints
    for i in I:
        used_capacity = sum(model.getVal(x[i, j]) for j in J)
        info_text += f"• Node {i} capacity: {used_capacity:.0f}/{k[i]}\n"
    
    # Add availability info
    available_assignments = sum(a[i][j] for i in I for j in J)
    made_assignments = sum(model.getVal(x[i, j]) for i in I for j in J)
    info_text += f"• Available slots: {available_assignments}, Used: {made_assignments:.0f}"
    
    # Create text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(1.02, 0.5, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', bbox=props)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig and filename:
        plt.savefig(f"{filename}_case_{case_idx}.png", dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}_case_{case_idx}.png")