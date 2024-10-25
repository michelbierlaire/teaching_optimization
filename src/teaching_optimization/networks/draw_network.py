"""Function to draw a network

Michel Bierlaire
Wed Jun 12 09:25:53 2024
"""

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from networkx.classes.digraph import DiGraph
import networkx as nx


def draw_network(
    the_network: DiGraph, attr_edge_labels: str | None = None, ax: Axis | None = None
) -> Axis:
    """Plots the network, bending the double arcs

    :param the_network: network to plot
    :param attr_edge_labels: name of the edge attribute used for labels.
    :param ax: matplotlib axis object, in case the plot must be included in an existing plot.

    """
    if ax is None:
        fig, ax = plt.subplots()

    positions = nx.get_node_attributes(the_network, name='pos')
    nx.draw_networkx_nodes(
        the_network.nodes, positions, node_color='skyblue', margins=0.1
    )

    nx.draw_networkx_nodes(the_network, positions, node_color='skyblue')
    nx.draw_networkx_labels(the_network, positions)

    # Draw edges with and without curvature
    curved_edges = [
        (u, v) for u, v in the_network.edges() if the_network.has_edge(v, u)
    ]
    straight_edges = [
        (u, v) for u, v in the_network.edges() if not the_network.has_edge(v, u)
    ]

    # Draw edge labels with adjustments for curved edges

    # Draw edges with curved connections
    nx.draw_networkx_edges(
        the_network, positions, edgelist=curved_edges, connectionstyle='arc3,rad=-0.2'
    )
    nx.draw_networkx_edges(the_network, positions, edgelist=straight_edges)

    # Draw labels for straight edges
    if attr_edge_labels:
        edge_labels = nx.get_edge_attributes(the_network, name=attr_edge_labels)
        nx.draw_networkx_edge_labels(
            the_network,
            positions,
            edge_labels={e: edge_labels[e] for e in straight_edges},
        )

        def draw_curved_edge_label(edge: tuple[Any, Any], label, offset=0.6) -> None:
            """Draw the label of one edge, with offset"""
            # Calculate midpoint
            upstream_node = edge[0]
            upstream_x, upstream_y = positions[upstream_node]
            downstream_node = edge[1]
            downstream_x, downstream_y = positions[downstream_node]

            # Calculate the direction of the edge
            main_direction = np.array(
                [float(downstream_x - upstream_x), float(downstream_y - upstream_y)]
            )

            # Calculate perpendicular vector (rotated 90 degrees counterclockwise)
            perpendicular_vector = np.array([-main_direction[1], main_direction[0]])

            # Normalize perpendicular vector
            perpendicular_vector /= np.linalg.norm(perpendicular_vector)
            scaled_perpendicular_vector = offset * perpendicular_vector
            # New upstream node
            new_upstream_node_coord = (
                upstream_x + scaled_perpendicular_vector[0],
                upstream_y + scaled_perpendicular_vector[1],
            )

            # New downstream node
            new_downstream_node_coord = (
                downstream_x + scaled_perpendicular_vector[0],
                downstream_y + scaled_perpendicular_vector[1],
            )

            relative_positions = {
                upstream_node: new_upstream_node_coord,
                downstream_node: new_downstream_node_coord,
            }
            nx.draw_networkx_edge_labels(
                the_network, relative_positions, edge_labels={edge: edge_labels[edge]}
            )

        for e in curved_edges:
            draw_curved_edge_label(e, edge_labels[e])

    return ax
