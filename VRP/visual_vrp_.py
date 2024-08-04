import matplotlib.pyplot as plt
import networkx as nx

def plot_2e_vrp(DC, satellites, customers, primary_routes, secondary_routes):
    G = nx.DiGraph()

    # Add nodes for depot, satellites, and customers
    G.add_node('DC', pos=(DC.xcoord, DC.ycoord), color='#1384ff')
    for i, satellite in enumerate(satellites):
        G.add_node(f'BigWin{i+len(customers)}', 
                   pos=(satellite.xcoord, satellite.ycoord), 
                   color='#de425b')
        
    for i, customer in enumerate(customers):
        G.add_node(f'SmallWin{i}', 
                   pos=(customer.xcoord, customer.ycoord), 
                   color='orange')

    # Add primary routes
    for route in primary_routes:
        G.add_edge('DC', f'BigWin{route[1]}', color='black')
        G.add_edge(f'BigWin{route[-1]}', 'DC', color='black')
        for i in range(1, len(route) - 1):
            print(f"BigWin{route[i]}")
            G.add_edge(f'BigWin{route[i]}', f'BigWin{route[i+1]}', color='black')

    # Add secondary routes
    for route in secondary_routes:
        G.add_edge(f'BigWin{route[0]}', f'SmallWin{route[1]}', color='black', linestyle='dashed')
        for i in range(1, len(route) - 1):
            G.add_edge(f'SmallWin{route[i]}', f'SmallWin{route[i+1]}', color='black', linestyle='dashed')

    pos = nx.get_node_attributes(G, 'pos')
    node_colors = [G.nodes[node].get('color', 'blue') for node in G.nodes()]
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    edge_styles = [G[u][v].get('linestyle', '-') for u, v in G.edges()]

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=False, node_size=200, 
            node_color=node_colors, edge_color=edge_colors, style=edge_styles, arrowsize=10)
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='DC', markerfacecolor='#1384ff', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='BigWin', markerfacecolor='#de425b', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='SmallWin', markerfacecolor='orange', markersize=10),
        plt.Line2D([0], [0], color='black', lw=2, linestyle='-', label='Truck Route'),
        plt.Line2D([0], [0], color='black', lw=2, linestyle='--', label='Motorbike Route')
    ]
    plt.legend(handles=legend_elements)
    plt.title('2E-VRP Solution')
    plt.show()


def plot_2e_vrp_sample(DC, satellites, customers, primary_routes, secondary_routes):
    G = nx.DiGraph()

    # Add nodes for depot, satellites, and customers
    G.add_node('DC', pos=DC, color='#1384ff')
    for i, satellite in enumerate(satellites):
        G.add_node(f'BigWin{i}', 
                   pos=satellite, 
                   color='#de425b')
        
    for i, customer in enumerate(customers):
        G.add_node(f'SmallWin{i}', 
                   pos=customer, 
                   color='orange')

    # Add primary routes
    for route in primary_routes:
        G.add_edge('DC', f'BigWin{route[1]}', color='black')
        G.add_edge(f'BigWin{route[-1]}', 'DC', color='black')
        for i in range(1, len(route) - 1):
            print(f"BigWin{route[i]}")
            G.add_edge(f'BigWin{route[i]}', f'BigWin{route[i+1]}', color='black')

    # Add secondary routes
    for route in secondary_routes:
        G.add_edge(f'BigWin{route[0]-22}', f'SmallWin{route[1]}', color='black', linestyle='dashed')
        for i in range(1, len(route) - 1):
            if route[i+1] >= 22:
                G.add_edge(f'SmallWin{route[i]}', f'BigWin{route[i+1]-22}', color='black', linestyle='dashed')
            elif route[i] >= 21:
                G.add_edge(f'BigWin{route[i]-22}', f'SmallWin{route[i+1]}', color='black', linestyle='dashed')
            else:
                G.add_edge(f'SmallWin{route[i]}', f'SmallWin{route[i+1]}', color='black', linestyle='dashed')

    pos = nx.get_node_attributes(G, 'pos')
    node_colors = [G.nodes[node].get('color', 'blue') for node in G.nodes()]
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    edge_styles = [G[u][v].get('linestyle', '-') for u, v in G.edges()]

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=False, node_size=200, 
            node_color=node_colors, edge_color=edge_colors, style=edge_styles, arrowsize=10)
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='DC', markerfacecolor='#1384ff', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='BigWin', markerfacecolor='#de425b', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='SmallWin', markerfacecolor='orange', markersize=10),
        plt.Line2D([0], [0], color='black', lw=2, linestyle='-', label='Truck Route'),
        plt.Line2D([0], [0], color='black', lw=2, linestyle='--', label='Motorbike Route')
    ]
    plt.legend(handles=legend_elements)
    plt.title('2E-VRP Solution')
    plt.show()


def plot_1e_vrp_sample(DC, customers, routes):
    G = nx.DiGraph()

    # Add nodes for depot and customers
    G.add_node('DC', pos=DC, color='red', label='DC')
    for i, customer in enumerate(customers):
        G.add_node(f'Customer{i}', 
                   pos=customer, 
                   color='orange',
                   label=f'H{i+1}')

    # Add routes
    for route in routes:
        G.add_edge('DC', f'Customer{route[0]}', color='darkblue')
        G.add_edge(f'Customer{route[-1]}', 'DC', color='darkblue')
        for i in range(len(route) - 1):
            G.add_edge(f'Customer{route[i]}', f'Customer{route[i+1]}', color='darkblue')

    pos = nx.get_node_attributes(G, 'pos')
    node_colors = [G.nodes[node].get('color', 'blue') for node in G.nodes()]
    node_labels = {node: G.nodes[node].get('label', '') for node in G.nodes()}
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=400, labels=node_labels,
            node_color=node_colors, edge_color=edge_colors, arrowsize=10)
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='DC', markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Hospital', markerfacecolor='orange', markersize=10),
        plt.Line2D([0], [0], color='black', lw=2, linestyle='-', label='Route')
    ]
    plt.legend(handles=legend_elements)
    plt.title('1E-VRP Solution')
    plt.show()

if __name__ == "__main__":
    # Example data
    depot = (9, 10)
    satellites = [(5, 7), (10, 8), (12, 12), (7, 11)]
    customers = [(2, 8), (3, 5), (1, 3), (4, 3), (6, 2), (7, 4), (11, 3), (14, 3), (16, 5), (15, 7),
                 (2, 11), (1, 13), (2, 15), (6, 15), (8, 14), (9, 15), (12, 16), (14, 17), (16, 15), (18, 12),
                 (20, 13), (18, 17)]
    print(len(customers))

    primary_routes = [[0, 0, 1], [0, 3, 2]]
    secondary_routes = [[22, 0, 1, 2, 3, 23, 8, 9], [22, 4, 5, 23, 6, 7], 
                        [25, 10, 11, 12, 24, 18, 19, 20, 21], [25, 13, 14, 15, 24, 16, 17]]

    # plot_2e_vrp_sample(depot, satellites, customers, primary_routes, secondary_routes)

    DC_position = (5, 7)
    customer_positions = [(2, 4), (5, 2), (6, 3), (5, 10), (4, 5), (6, 8), (10, 7), (7, 3), (9, 9)]
    # routes = [[5, 3, 0, 4, 1], [2, 7, 6, 8]]
    # routes = [[3, 0, 4,], [2, 1, 4], [8, 6, 7]]

    plot_1e_vrp_sample(DC_position, customer_positions, routes)
