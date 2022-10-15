import os

import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import numpy as np
import plotly.express as px
from PIL import Image
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash_bootstrap_templates import load_figure_template


class Dashboard:
    def __init__(self, cfg, logger, dataset, evaluation_results):
        self.cfg = cfg['dashboard']
        self.dataset_cfg = cfg['dataset']
        self.logger = logger
        self.dataset = dataset
        self.evaluation_results = evaluation_results
        self.slider_size = len(self.dataset)

        self.logger.info('initializing dashboard for EDA and training results visualization')
        load_figure_template('LUX')
        self.app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

        self.slider_data = self.dataset.dataset
        self.get_images_figures(idx=0)
        self.init_sidebar()
        self.init_main_part_div()

        @self.app.callback(
            Output("dataset-image-0", "figure"),
            Output("dataset-image-1", "figure"),
            Input("dataset-viewer-slider", "value"),
            Input("show-boxes-switch", "on"),
        )
        def slide_through_dataset_samples(slider_value, switch_value):
            self.get_images_figures(idx=slider_value // 2, show_boxes=switch_value)
            return self.figures[0], self.figures[1]

        @self.app.callback(
            Output("main-part-div", "children"),
            Output("dropdown_0", "value"),
            Input("dropdown_0", "value")
        )
        def show_dropdown_result(value):
            if value is None:
                return self.app.layout, value
            elif value == 'EDA':
                updated_div = html.Div(children=[
                    dbc.Row([
                        dbc.Col(self.sidebar, style={'display': 'inline-block'}),
                        dbc.Col(html.H1('Object Detection Problem, EDA'), width=9, style={'marginTop': '1.5%'},
                                id='main-title'),
                    ]),
                    dbc.Row([
                        dbc.Col(),
                        dbc.Col(html.H3('Dataset Common Info'), width=9, style={'marginTop': '4%',
                                                                                'marginBottom': '1.5%'}),
                        html.P(f"Number of Images: {self.dataset.size}", style={'marginLeft': '27%',
                                                                                'marginBottom': '0.5%'}),
                        html.P(f"Number of Classes: {self.dataset.classes_num}", style={'marginLeft': '27%',
                                                                                        'marginBottom': '0.5%'}),
                        html.P(f"Number of Bounding Boxes: {self.dataset.bboxes_num}", style={'marginLeft': '27%',
                                                                                              'marginBottom': '0.5%'}),
                        html.P(f"Images Height: "
                               f"Avg={self.dataset.images_sizes_stats['avg_h']}, "
                               f"Min={self.dataset.images_sizes_stats['min_h']}, "
                               f"Max={self.dataset.images_sizes_stats['max_h']} ",
                               style={'marginLeft': '27%', 'marginBottom': '0.5%'}),
                        html.P(f"Images Width: "
                               f"Avg={self.dataset.images_sizes_stats['avg_w']}, "
                               f"Min={self.dataset.images_sizes_stats['min_w']}, "
                               f"Max={self.dataset.images_sizes_stats['max_w']} ",
                               style={'marginLeft': '27%', 'marginBottom': '0.5%'}),
                        html.P(f"Boxes Height: "
                               f"Avg={self.dataset.bboxes_sizes_stats['avg_h']}, "
                               f"Min={self.dataset.bboxes_sizes_stats['min_h']}, "
                               f"Max={self.dataset.bboxes_sizes_stats['max_h']} ",
                               style={'marginLeft': '27%', 'marginBottom': '0.5%'}),
                        html.P(f"Boxes Width: "
                               f"Avg={self.dataset.bboxes_sizes_stats['avg_w']}, "
                               f"Min={self.dataset.bboxes_sizes_stats['min_w']}, "
                               f"Max={self.dataset.bboxes_sizes_stats['max_w']} ",
                               style={'marginLeft': '27%'}),
                    ]),
                    dbc.Row([
                        dbc.Col(),
                        dbc.Col(html.H3('Dataset Analysis'), width=9, style={'marginTop': '2%'}),
                        dbc.Col(html.P('1. Labels'), width=9,
                                style={'marginTop': '1.5%', 'marginLeft': '27%'}),
                    ]),
                    dbc.Row(
                        [dbc.Col(),
                         dbc.Col(html.Div("Images with and without labels"),
                                 style={'marginLeft': '4.5%', 'width': '40%'}),
                         dbc.Col(html.Div("Classes balance"), style={'marginLeft': '4.5%', 'width': '40%'}),
                         ],
                        className="g-0", style={'marginTop': '1%'}
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
                    dbc.Row([dbc.Col(html.P('2. Images and Boxes aspect ratios'), width=9,
                                     style={'marginTop': '1%', 'marginLeft': '27%'})]),
                    dbc.Row([
                        dbc.Col(
                            dcc.Graph(id='img-aspect-ratio-hist', figure=px.histogram(self.dataset.images_aspect_ratios,
                                                                                      x='Images aspect ratio',
                                                                                      nbins=1, range_x=(-1, 2))),
                            width=9, style={'width': '40%', 'display': 'inline-block', 'marginRight': '7%'}),
                        dbc.Col(
                            dcc.Graph(id='box-aspect-ratio-hist', figure=px.histogram(self.dataset.bboxes_aspect_ratios,
                                                                                      x='Boxes aspect ratio',
                                                                                      color='Class')),
                            width=9, style={'width': '40%', 'display': 'inline-block'})
                    ], style={'marginLeft': '25.7%'}),

                    # distribution of boxes across images
                    dbc.Row([dbc.Col(html.P('3. Distribution of boxes across images'), width=9,
                                     style={'marginTop': '1%', 'marginLeft': '27%'})]),
                    dbc.Row([
                        dbc.Col(
                            dcc.Graph(id='boxes-per-image-hist', figure=px.histogram(self.dataset.num_boxes_per_image,
                                                                                     x='Number of Boxes per Image')),
                            width=9, style={'width': '35%', 'display': 'inline-block', 'marginLeft': '27%'}),
                        dbc.Col(
                            dcc.Graph(id='boxes-per-image-hist', figure=px.histogram(self.dataset.boxes_area,
                                                                                     x='Boxes Area', color='Class')),
                            width=9, style={'width': '35%', 'display': 'inline-block'})
                    ]),

                    # viewing dataset samples
                    dbc.Row([
                        dbc.Col(),
                        dbc.Col(html.H3('Dataset Viewer'), width=9, style={'marginTop': '1.5%', 'marginBottom': '1%'}),
                        dbc.Col(
                            [html.P('You can go through dataset samples by moving slider below'),
                             html.P('Turn off the switch if you don`t want to see boxes on images')],
                            width=9, style={'marginBottom': '1%', 'marginLeft': '30%'}),
                    ]),
                    dbc.Nav([
                        html.Div([
                            daq.BooleanSwitch(id='show-boxes-switch', on=True, label='Draw Boxes',
                                              labelPosition='Right', color=px.colors.qualitative.G10[3],
                                              style={'width': '10%', 'marginTop': '1%'})]),
                    ],
                        vertical=True, pills=True, id='settings-addition',
                        style={'marginLeft': '30%', 'width': '100%', 'marginBottom': '3%'}
                    ),

                    dbc.Row([dcc.Slider(0, self.slider_size, 2, id='dataset-viewer-slider',
                                        marks={i: '{}'.format(i) for i in range(0, self.slider_size + 1,
                                                                                self.slider_size // 6)},
                                        value=0, updatemode='drag',
                                        tooltip={"placement": "bottom", "always_visible": True})
                             ], style={'width': '60%', 'marginLeft': '30%'}),

                    dbc.Row(
                        [dbc.Col(self.sidebar, style={'display': 'inline-block'}),
                         dbc.Col(dcc.Graph(id='dataset-image-0', figure=self.figures[0]),
                                 style={'display': 'inline-block'}),
                         dbc.Col(dcc.Graph(id='dataset-image-1', figure=self.figures[1]),
                                 style={'display': 'inline-block', 'marginRight': '10%',
                                        'marginBottom': '10%'})
                         ]),
                ], id='main-part-div')
            else:
                updated_div = html.Div(children=[
                    dbc.Row([
                        dbc.Col(self.sidebar, style={'display': 'inline-block'}),
                        dbc.Col(html.H1('Object Detection Problem, Training Results'), width=9,
                                style={'marginTop': '1.5%'},
                                id='main-title'),
                    ])
                ], id='main-part-div')

                # TODO: add training details (hyperparams etc., training time on kaggle and colab, metrics on test set)

            return updated_div, value

    def get_images_figures(self, idx=0, img_count=2, show_boxes=True):
        self.figures = []
        for row in self.slider_data[idx:idx + img_count].iterrows():
            image = Image.open(os.path.join(self.dataset_cfg['data_path'], row[1]['img_path']))
            fig = px.imshow(image)
            image_w, image_h = image.size

            images_layout = {'plot_bgcolor': 'white', 'paper_bgcolor': 'white', 'margin': dict(t=20, b=0, l=0, r=0),
                             'xaxis': dict(showgrid=False, showticklabels=False, linewidth=0),
                             'yaxis': dict(showgrid=False, showticklabels=False, linewidth=0),
                             'hovermode': False}
            fig.update_layout(**images_layout)

            if show_boxes:
                # draw boxes
                if np.isnan(row[1]['label']).any():
                    self.figures.append(fig)
                    continue

                for label in row[1]['label']:
                    object_class, x, y, width, height = label
                    x_coord, y_coord = image_w * x, image_h * y  # center of box
                    width_half, height_half = image_w * width / 2, image_h * height / 2

                    fig.add_shape(type="rect",
                                  x0=x_coord - width_half, x1=x_coord + width_half,
                                  y0=y_coord - height_half, y1=y_coord + height_half,
                                  line=dict(color=px.colors.qualitative.Plotly[object_class], width=2))
                    fig.update_shapes(dict(xref='x', yref='y'))
            self.figures.append(fig)

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
