import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.express as px
from dash import dcc
from dash import html
from dash_bootstrap_templates import load_figure_template

from dataset import Dataset

load_figure_template('LUX')

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "24rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
    [
        html.H1("Filters"),
        html.Hr(),
        dbc.Nav(
            [
                html.P(
                    "View Images Samples", className="lead"
                ),
                html.Div([
                    dcc.RadioItems([' with any boxes', ' without boxes'], ' with any boxes', inline=False,
                                   labelStyle={'display': 'block'})], style={'marginLeft': '4%'}),
                html.Br(),
                html.Div([
                    daq.BooleanSwitch(id='show-boxes-switch', on=True, label='Draw Boxes',
                                      labelPosition='Right', color=px.colors.qualitative.G10[3]),
                ], style={'marginLeft': '2%', 'width': '45%', 'marginBottom': '2%'}),
                html.Br(),
                html.Div(
                    [
                        dbc.Button("Prev", outline=True, color="secondary", className="me-1"),
                        dbc.Button(
                            "Next", outline=True, color="secondary", className="me-1"
                        )], style={'marginBottom': '7%', 'marginLeft': '4%'}),
                html.Hr(),
                html.P(
                    "View Training Results", className="lead"
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

dataset = Dataset()

img_fig_0 = dataset.figures[0]
img_fig_1 = dataset.figures[0]

# Hide the axes and the tooltips
img_fig_0.update_layout(plot_bgcolor='white', paper_bgcolor='white', margin=dict(t=20, b=0, l=0, r=0),
                        xaxis=dict(showgrid=False, showticklabels=False, linewidth=0),
                        yaxis=dict(showgrid=False, showticklabels=False, linewidth=0),
                        hovermode=False
                        )
img_fig_1.update_layout(plot_bgcolor='white', paper_bgcolor='white', margin=dict(t=20, b=0, l=0, r=0),
                        xaxis=dict(showgrid=False, showticklabels=False, linewidth=0),
                        yaxis=dict(showgrid=False, showticklabels=False, linewidth=0),
                        hovermode=False
                        )

app.layout = html.Div(children=[

    dbc.Row([
        dbc.Col(),
        dbc.Col(html.H3('Statistics'), width=9, style={'marginTop': '2%'}),
    ]),
    dbc.Row(
        [dbc.Col(),
         dbc.Col(html.Div("Images with and without labels"), style={'marginLeft': '4.5%', 'width': '40%'}),
         dbc.Col(html.Div("Types of labels"), style={'marginLeft': '4.5%', 'width': '40%'}),
         ],
        className="g-0", style={'marginTop': '2%'}
    ),
    dbc.Row(
        [
            dbc.Col(dcc.Graph(id='graph1', figure=px.pie(dataset.labeled_and_non_labeled_data_num, values='num_images',
                                                         names='labeled_or_not', hole=.3)), width=9,
                    style={'width': '40%', 'marginLeft': '7%', 'display': 'inline-block'}),

            dbc.Col(dcc.Graph(id='graph2', figure=px.pie(dataset.objects_id_counter, names='class',
                                                         values='boxes_num', hole=.0)), width=9,
                    style={'width': '40%', 'display': 'inline-block'})
        ], style={'marginLeft': '20%'}),

    dbc.Row([
        dbc.Col(),
        dbc.Col(html.H3('Dataset Viewer'), width=9, style={'marginTop': '1.5%'}),
    ]),
    dbc.Row(
        [dbc.Col(sidebar, style={'display': 'inline-block'}),
         dbc.Col(dcc.Graph(id=f'image-{0}', figure=img_fig_0),
                 style={'display': 'inline-block'}),
         dbc.Col(dcc.Graph(id=f'image-{1}', figure=img_fig_1),
                 style={'display': 'inline-block', 'marginRight': '10%'})
         ]),

]
)
if __name__ == '__main__':
    app.run_server(debug=True)
