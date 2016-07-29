# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:47:15 2016

@author: anna
"""

import plotly
import plotly.graph_objs
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
import six
import numpy
from plotly.tools import FigureFactory as FF
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.graph_objs import *
from matplotlib import colors
#from datetime import datetime
from operator import itemgetter
from scipy import stats
from numpy import *

def trace_Box(x_data, y_data, color_, name, text):
    track=go.Bar(
        x=x_data,
        y=y_data,
        name=name,
        text=text,
        marker=dict(
            color=color_,
            line=dict(
                color=color_,
                width=1.5,
            )
        ),
        opacity=0.6,
    )
    return track;

def plot_data(table):
    x_range=[table['Server_class'][i].split(".")[1]  if "." in table['Server_class'][i] else table['Server_class'][i] for i in range(len(table['Server_class'])) ]

    tr=Scatter({
            'x': x_range,
            'y': table['Mem_avg_clust'],
            'name': 'Total Cluster load', 
            'mode': 'markers',
            'line': {
                'color': "red",
                'width': 1,
            },
            'marker': {
                'color': "red",
                'symbol': 24,
                'size' : 30,
            },
        })


    min_trace=trace_Box(x_range, table['Mem_min_utilized'], "green", "min server load", table['Mem_min_utilized'])

    max_trace_diff=[table['Mem_max_utilized'][i]-table['Mem_min_utilized'][i] for i in range(len(table['Server_class']))]
    max_trace=trace_Box(x_range, max_trace_diff, "blue", "max server load", table['Mem_max_utilized'])

    trace_100_per=[100-table['Mem_max_utilized'][i] for i in range(len(table['Server_class']))]
    trace_100=trace_Box(x_range, trace_100_per, "gray", "Free capacity", trace_100_per)

    data=[min_trace, max_trace, trace_100, tr]

    layout = {
        'barmode': 'stack',
        'xaxis': {
            'showgrid': True,
            'title' : 'Server_class',
        },
        'yaxis': {
            'showgrid': True,
            'title' : '% of Utilized',
        }
        }

    fig = {
        'data': data,
        'layout': layout,
    }

    plot(fig, filename="1.html")
    x=["A.d", "H.z", "Y.f"]