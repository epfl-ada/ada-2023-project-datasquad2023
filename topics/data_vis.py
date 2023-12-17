# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime



def plot_product_hist(x, y, release_date, title):
    # convert x from list of string to list of datetime
    x = [datetime.strptime(date, '%Y-%m-%d') for date in x]
    start_date = x[0]

    # Define the specific time ranges as 1 month before and after the release date
    time_before_specific = datetime.strptime(release_date, '%Y-%m-%d') - datetime.timedelta(days=30)
    time_after_specific = datetime.strptime(release_date, '%Y-%m-%d') + datetime.timedelta(days=30)

    # Convert dates to days
    days = np.array([(date - start_date).days for date in x])

    # Categorize the days into three groups based on time
    colors = np.zeros(len(x), dtype=int)
    colors[days < (time_before_specific - start_date).days] = 0  # before
    colors[(days >= (time_before_specific - start_date).days) & 
        (days < (time_after_specific - start_date).days)] = 1  # during
    colors[days >= (time_after_specific - start_date).days] = 2  # after

    # Set a color palette for the plot
    color_palette = ['firebrick', 'gold', 'deepskyblue']

    # Plot the histogram with colored bars
    fig, ax = plt.subplots()
    for col in np.unique(colors):
        # add a bar plot for each color
        ax.bar(days[colors == col], y[colors == col], color=color_palette[col], edgecolor='black', alpha=0.7, label=f'Color {col}', width=1)

    # add a grey vertical line for the release date
    plt.axvline((datetime.strptime(release_date, '%Y-%m-%d') - start_date).days, color='red', linewidth=1, linestyle='--', label=f'Release date: {release_date}')

    # Customize the plot
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Frequency')

    # add legend for colors
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color_palette[col]) for col in np.unique(colors)]
    legend_elements.append(plt.Line2D([0], [0], color='grey', linewidth=1, linestyle='--'))
    plt.legend(legend_elements, ['Before release', 'Release', 'After release', 'Release date'])

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Add background color
    ax.set_facecolor('#f4f4f4')

    # Add horizontal grid lines
    plt.axhline(0, color='black',linewidth=0.5)

    # Add a subtle border to the bars
    for rect in ax.patches:
        rect.set_linewidth(0.2)
        rect.set_edgecolor('black')

    # Show the plot
    plt.show()
