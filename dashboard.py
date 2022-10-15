import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.express as px
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash_bootstrap_templates import load_figure_template


class Dashboard:
    def __init__(self, cfg, logger, dataset, evaluation_results):
        self.cfg = cfg['dashboard']
        self.logger = logger
        self.dataset = dataset
        self.evaluation_results = evaluation_results

        self.logger.info('initializing dashboard for EDA and training results visualization')
        load_figure_template('LUX')
        self.app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

        self.init_sidebar()
        self.init_main_part_div()

        # visualize images samples TODO: remove
        img_fig_0 = self.dataset.figures[0]
        img_fig_1 = self.dataset.figures[0]
        # Hide the axes and the tooltips
        images_layout = {'plot_bgcolor': 'white', 'paper_bgcolor': 'white', 'margin': dict(t=20, b=0, l=0, r=0),
                         'xaxis': dict(showgrid=False, showticklabels=False, linewidth=0),
                         'yaxis': dict(showgrid=False, showticklabels=False, linewidth=0),
                         'hovermode': False}
        img_fig_0.update_layout(**images_layout)
        img_fig_1.update_layout(**images_layout)

        @self.app.callback(
            Output("main-part-div", "children"),
            Output("dropdown_0", "value"),
            Output("settings-addition", "children"),
            Input("dropdown_0", "value")
        )
        def show_dropdown_result(value):
            if value is None:
                settings_addition = dbc.Nav(id='settings-addition')
                return self.app.layout, value, settings_addition
            elif value == 'EDA':
                updated_div = html.Div(children=[
                    dbc.Row([
                        dbc.Col(self.sidebar, style={'display': 'inline-block'}),
                        dbc.Col(html.H1('Object Detection Problem, EDA'), width=9, style={'marginTop': '1.5%'},
                                id='main-title'),
                    ]),
                    dbc.Row([
                        dbc.Col(),
                        dbc.Col(html.H3('Dataset Common Info'), width=9, style={'marginTop': '2%',
                                                                                'marginBottom': '2%'}),
                        html.P(f"Number of Images: {self.dataset.size}", style={'marginLeft': '25%',
                                                                                'marginBottom': '0.5%'}),
                        html.P(f"Number of Classes: {self.dataset.classes_num}", style={'marginLeft': '25%',
                                                                                        'marginBottom': '0.5%'}),
                        html.P(f"Number of Bounding Boxes: {self.dataset.bboxes_num}", style={'marginLeft': '25%',
                                                                                              'marginBottom': '0.5%'}),
                        html.P(f"Images Height: "
                               f"Avg={self.dataset.images_sizes_stats['avg_h']}, "
                               f"Min={self.dataset.images_sizes_stats['min_h']}, "
                               f"Max={self.dataset.images_sizes_stats['max_h']} ",
                               style={'marginLeft': '25%', 'marginBottom': '0.5%'}),
                        html.P(f"Images Width: "
                               f"Avg={self.dataset.images_sizes_stats['avg_w']}, "
                               f"Min={self.dataset.images_sizes_stats['min_w']}, "
                               f"Max={self.dataset.images_sizes_stats['max_w']} ",
                               style={'marginLeft': '25%', 'marginBottom': '0.5%'}),
                        html.P(f"Boxes Height: "
                               f"Avg={self.dataset.bboxes_sizes_stats['avg_h']}, "
                               f"Min={self.dataset.bboxes_sizes_stats['min_h']}, "
                               f"Max={self.dataset.bboxes_sizes_stats['max_h']} ",
                               style={'marginLeft': '25%', 'marginBottom': '0.5%'}),
                        html.P(f"Boxes Width: "
                               f"Avg={self.dataset.bboxes_sizes_stats['avg_w']}, "
                               f"Min={self.dataset.bboxes_sizes_stats['min_w']}, "
                               f"Max={self.dataset.bboxes_sizes_stats['max_w']} ",
                               style={'marginLeft': '25%', 'marginBottom': '0.5%'}),
                    ]),
                    dbc.Row([
                        dbc.Col(),
                        dbc.Col(html.H3('Dataset Analysis'), width=9, style={'marginTop': '2%'}),
                    ]),
                    dbc.Row(
                        [dbc.Col(),
                         dbc.Col(html.Div("Images with and without labels"),
                                 style={'marginLeft': '4.5%', 'width': '40%'}),
                         dbc.Col(html.Div("Types of labels"), style={'marginLeft': '4.5%', 'width': '40%'}),
                         ],
                        className="g-0", style={'marginTop': '2%'}
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(id='graph1',
                                          figure=px.pie(self.dataset.labeled_and_non_labeled_data_num,
                                                        values='num_images',
                                                        names='labeled_or_not', hole=.3)), width=9,
                                style={'width': '40%', 'marginLeft': '7%', 'display': 'inline-block'}),

                            dbc.Col(dcc.Graph(id='graph2', figure=px.pie(self.dataset.objects_id_counter, names='class',
                                                                         values='boxes_num', hole=.0)), width=9,
                                    style={'width': '40%', 'display': 'inline-block'})
                        ], style={'marginLeft': '20%'}),

                    # aspect ratios
                    dbc.Row([
                        dbc.Col(
                            dcc.Graph(id='img-aspect-ratio-hist', figure=px.histogram(self.dataset.images_aspect_ratios,
                                                                                      x='Images aspect ratio',
                                                                                      nbins=1, range_x=(-1, 2))),
                            width=9, style={'width': '40%', 'display': 'inline-block', 'marginRight': '7%'}),
                        dbc.Col(
                            dcc.Graph(id='box-aspect-ratio-hist', figure=px.histogram(self.dataset.bboxes_aspect_ratios,
                                                                                      x='Boxes aspect ratio')),
                            width=9, style={'width': '40%', 'display': 'inline-block'})
                    ], style={'marginLeft': '25.7%'}),

                    dbc.Row([
                        dbc.Col(),
                        dbc.Col(html.H3('Dataset Viewer'), width=9, style={'marginTop': '1.5%'}),
                    ]),
                    dbc.Row(
                        [dbc.Col(self.sidebar, style={'display': 'inline-block'}),
                         dbc.Col(dcc.Graph(id=f'image-{0}', figure=img_fig_0),
                                 style={'display': 'inline-block'}),
                         dbc.Col(dcc.Graph(id=f'image-{1}', figure=img_fig_1),
                                 style={'display': 'inline-block', 'marginRight': '10%'})
                         ]),
                ], id='main-part-div')

                settings_addition = dbc.Nav(
                    [
                        html.P(
                            "View Images Samples", className="lead", style={'marginTop': '80%', 'marginBottom': '15%'}
                        ),
                        html.Div([
                            dcc.RadioItems([' with any boxes', ' without boxes'], ' with any boxes', inline=False,
                                           labelStyle={'display': 'block'})],
                            style={'marginLeft': '45%', 'width': '100%', 'marginBottom': '3%'}),
                        # html.Br(),
                        html.Div([
                            daq.BooleanSwitch(id='show-boxes-switch', on=True, label='Draw Boxes',
                                              labelPosition='Right', color=px.colors.qualitative.G10[3]),
                        ], style={'marginLeft': '50%', 'width': '55%', 'marginBottom': '15%'}),
                        # html.Br(),
                        html.Div(
                            [
                                dbc.Button("Prev", outline=True, color="secondary", className="me-1"),
                                dbc.Button(
                                    "Next", outline=True, color="secondary", className="me-1"
                                )], style={'marginBottom': '7%', 'marginLeft': '35%', 'width': '100%'}),
                    ],
                    vertical=True,
                    pills=True,
                    id='settings-addition',
                    # style={'marginLeft': '5%'}
                ),
            else:
                updated_div = html.Div(children=[
                    dbc.Row([
                        dbc.Col(self.sidebar, style={'display': 'inline-block'}),
                        dbc.Col(html.H1('Object Detection Problem, Training Results'), width=9,
                                style={'marginTop': '1.5%'},
                                id='main-title'),
                    ])
                ], id='main-part-div')
                settings_addition = dbc.Nav(id='settings-addition')

            return updated_div, value, settings_addition

    def init_sidebar(self):
        self.sidebar = html.Div(
            [
                html.H3("Settings", style={'marginBottom': '2%'}),
                html.Hr(),
                html.P("Choose what to view"),
                dcc.Dropdown(
                    ['EDA', 'Training Results'],
                    searchable=False, style={'marginBottom': '95%'}, id="dropdown_0",
                ),
                dbc.Nav(id='settings-addition'),
            ],
            style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "24rem", "padding": "2rem 1rem",
                   "background-color": "#f8f9fa"},
        )

    def init_main_part_div(self):
        self.app.layout = html.Div(children=[
            dbc.Row([
                dbc.Col(self.sidebar, style={'display': 'inline-block'}),
                dbc.Col(html.H1('Object Detection Problem'), width=9, style={'marginTop': '1.5%'}, id='main-title'),
            ]),
        ], id='main-part-div')

    def __call__(self, *args, **kwargs):
        self.logger.info('running dashboard for EDA and training results visualization')
        self.app.run_server(debug=self.cfg['debug_mode'])
