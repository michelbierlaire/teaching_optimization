"""Plot the objective function

Code to plot a function.

Michel Bierlaire
Sun Mar 10 12:30:54 2024
"""

from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axis import Axis


def plot_function(
    my_function: Callable[[float], float],
    label: str,
    x_label: str,
    y_label: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    ax: Axis | None = None,
) -> Axis:
    """Plot an objective function"""
    if ax is None:
        fig, ax = plt.subplots()

    # Generate x values
    x_values = np.linspace(start=x_min, stop=x_max, num=400)

    # Generate y values using the objective function
    y_values = [my_function(x) for x in x_values]

    # Filter out values of y outside of the range.
    filtered_values = [
        (x, y) for x, y in zip(x_values, y_values) if y_min <= y <= y_max
    ]
    x_values_filtered, y_values_filtered = zip(
        *filtered_values
    )  # This separates the filtered pairs back into two lists

    # Create the plot
    plt.plot(x_values_filtered, y_values_filtered, color='blue', label=label)

    # Add title and labels
    plt.title(label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Set y-axis limits
    plt.ylim(y_min, y_max)

    # Setting the ticks on axes
    x_range = max(x_values) - min(x_values)
    x_start = min(x_values)
    x_stop = max(x_values) + 0.01 * x_range
    plt.xticks(
        np.arange(start=x_start, stop=x_stop, step=x_range / 10)
    )  # Adjust x-axis tick interval if needed
    y_range = max(y_values) - min(y_values)
    y_start = min(y_values)
    y_stop = max(y_values) + 0.01 * y_range

    plt.yticks(
        np.arange(start=y_start, stop=y_stop, step=y_range / 10)
    )  # Adjust y-axis tick interval if needed

    # Add a grid
    plt.grid(visible=True, color='gray', linestyle='-', linewidth=0.5)
    plt.legend()
    return ax
