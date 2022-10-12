import json
import os
from collections import Counter

import dash_bootstrap_components as dbc
import dash_daq as daq
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from dash import Dash, dcc, html

from config import args


class Dataset:
    def __init__(self):

        with open('dataset_config.json') as json_file:
            self.dataset_cfg = json.load(json_file)

        self.dataset = self.get_dataset_as_dataframe()
        self.get_eda()
        print(args.data_path)

    @staticmethod
    def get_dashboard(figures):
        app = Dash(external_stylesheets=[dbc.themes.LUX])
        # TODO: prev, next; Boxes (turn_on); show labelled, unlabelled data;

        # graphs_list = [dcc.Graph(id=f'image-{fig_id}', figure=fig) for fig_id, fig in enumerate(figures)]

        app.layout = html.Div([

            html.Div([
                html.Label('Show Images Sample')],
                style={'fontSize': 18, 'margin': '2%'}),

            html.Div([
                dcc.RadioItems(['with any boxes', 'without boxes'], 'with any boxes', inline=False,
                               labelStyle={'display': 'block'})], style={'margin': '2%'}),

            html.Div([
                daq.BooleanSwitch(id='show-boxes-switch', on=True, label='Show Boxes',
                                  labelPosition='Right', color=px.colors.qualitative.Prism[4]),
            ], style={'margin': '2%', 'marginTop': '3%', 'width': '35%'})
        ],
            style={
                # 'background-color': px.colors.qualitative.Prism[10], 'fontSize': 18, 'font-family': 'monospace', 'color': 'white'
                'width': '20%', 'margin': '2%', 'padding': '0.5%'})
        # px.colors.qualitative.G10[9]
        app.run_server()

        pass

    def get_eda(self):
        """
        EDA is an iterative cycle. You:
        1. Generate questions about your data.
        2. Search for answers by visualising, transforming, and modelling your data.
        3. Use what you learn to refine your questions and/or generate new questions.
        """
        # get number of images without label
        isnan_labels_num = self.dataset.label.isna().sum()

        # visualize images sample
        labeled_data_sample = self.dataset[~self.dataset.label.isna()]
        non_labeled_data_sample = self.dataset[self.dataset.label.isna()]

        figures = []
        for row in labeled_data_sample.head(10).iterrows():
            image = Image.open(os.path.join(args.data_path, row[1]['img_path']))
            image_w, image_h = image.size
            fig = px.imshow(image)

            # draw boxes
            for label in row[1]['label']:
                object_class, x, y, width, height = label
                x_coord, y_coord = image_w * x, image_h * y  # center of box
                width_half, height_half = image_w * width / 2, image_h * height / 2

                fig.add_shape(type="rect",
                              x0=x_coord - width_half, x1=x_coord + width_half,
                              y0=y_coord - height_half, y1=y_coord + height_half,
                              line=dict(color=px.colors.qualitative.Plotly[object_class], width=2),
                              )
                fig.update_shapes(dict(xref='x', yref='y'))
                # fig.show()
                # a = 1
            figures.append(fig)

        # self.get_dashboard(figures)
        self.figures = figures
        self.labeled_and_non_labeled_data_num = pd.DataFrame()
        self.labeled_and_non_labeled_data_num['labeled_or_not'] = ['With',
                                                                   'Without']
        self.labeled_and_non_labeled_data_num['num_images'] = [len(labeled_data_sample),
                                                               len(non_labeled_data_sample)]
        objects_id_counter = Counter(
            np.concatenate(self.dataset.label[~self.dataset.label.isna()].to_list())[:, 0].astype(int))
        objects_id_dict = {
            f'{self.dataset_cfg["class_id_2_class_name_mapping"][str(object_id)]}': boxes_num for
            object_id, boxes_num in objects_id_counter.items()}

        # self.objects_id_counter = objects_id_dict

        self.objects_id_counter = pd.DataFrame()
        self.objects_id_counter['class'] = list(objects_id_dict.keys())
        self.objects_id_counter['boxes_num'] = list(objects_id_dict.values())
        pass

    def get_dataset_as_dataframe(self):
        files_list = np.asarray(os.listdir(args.data_path))
        files_num = len(files_list)
        jpg_files_id = np.asarray([idx for idx, file_name in enumerate(files_list) if file_name.endswith('.jpg')])
        txt_files_id = np.setdiff1d(np.arange(files_num), jpg_files_id)

        # read labels
        txt_files = files_list[txt_files_id]
        labels = self.read_labels(txt_files)

        # make a pandas dataframe with jpeg paths and labels
        dataset_df = pd.DataFrame()
        dataset_df['img_path'] = files_list[jpg_files_id]
        dataset_df['label'] = labels
        return dataset_df

    def read_labels(self, txt_files):
        labels = []
        # self.object_ids = []
        for txt_file in txt_files:
            with open(os.path.join(args.data_path, txt_file), 'r') as f:
                text = f.readlines()
            if text:
                # boxes = []
                # for line in text:
                #     for num_id, num in enumerate(line.replace('\n', '').split(' ')):
                #         if num_id == 0:
                #             object_id = int(num)
                #             boxes.append(object_id)
                #             self.object_ids.append(object_id)
                #         else:
                #             boxes.append(float(num))
                boxes = [[int(num) if num_id == 0 else float(num) for num_id, num in
                          enumerate(line.replace('\n', '').split(' '))] for line in text]
            else:
                boxes = np.nan
            labels.append(boxes)
        return labels


if __name__ == '__main__':
    dataset = Dataset()
