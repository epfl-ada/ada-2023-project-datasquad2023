# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import datetime as dt
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_product_hist(x, y, release_date, title):
    # convert x from list of string to list of datetime
    # x = [dt.datetime.strptime(date, '%Y-%m-%d') for date in x]
    start_date = x[0]

    # Define the specific time ranges as 1 month before and after the release date
    time_before_specific = dt.datetime.strptime(release_date, '%d-%m-%Y') - dt.timedelta(days=30)
    time_after_specific = dt.datetime.strptime(release_date, '%d-%m-%Y') + dt.timedelta(days=30)

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
        ax.bar(x=np.array(x)[colors == col], height=np.array(y)[colors == col], color=color_palette[col], edgecolor='black', alpha=0.7, label=f'Color {col}', width=1)


    # add a grey vertical line for the release date
    plt.axvline(dt.datetime.strptime(release_date, '%d-%m-%Y'), color='red', linewidth=1, linestyle='--', label=f'Release date: {release_date}')

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

def plot_line(data_dict: dict, release: str, title: str):
    """
    This function plots the data given by data_dict as an interactive line plot using Plotly
    :param data_dict: a dictionary where keys are datetimes and values are floats
    :param release_date is a special date that is highlighted with a vertical dotted line in the plot
    :param title: is the title of the plot
    Keep in mind that this function creates a very beautiful and appealing plot with beautiful colours
    and follows all best practices of Python line plotting
    """
    # Extracting dates and values from data_dict
    dates = list(data_dict.keys())
    values = list(data_dict.values())
    print('these are the dates', dates)

    # Convert release_date to timestamp
    release = datetime.strptime(release, '%d-%m-%Y')

    # Selecting a fraction of dates to display on the x-axis
    tick_fraction = 20
    subset_dates = dates[::tick_fraction]

    # Creating a trace for the data line
    trace = go.Scatter(x=dates, y=values, mode='lines+markers', name='Data Line', marker=dict(color='skyblue', size=1.5))

    # Creating a trace for the release date line
    release_trace = go.Scatter(x=[release, release], y=[min(values), max(values)],
                               mode='lines', name='Release Date', line=dict(color='red', dash='dash'))

    # Creating the layout
    layout = go.Layout(title=title, xaxis=dict(title='Date', tickangle=45, tickmode='array', tickvals=subset_dates,
                                               ticktext=[date for date in subset_dates],
                                               showgrid=False),
                       yaxis=dict(title='Values'), showlegend=True, hovermode='closest',
                       plot_bgcolor='rgba(240, 240, 240, 0.6)')
    
    # add vertical lines for release date
    layout.update(dict(shapes=[dict(type='line', xref='x', yref='paper', x0=release, y0=0, x1=release, y1=1,
                                    line=dict(color='red', width=2, dash='dash'))]))

    # Combining traces and layout into a figure
    fig = go.Figure(data=[trace, release_trace], layout=layout)

    # Displaying the interactive plot
    fig.show()
