import os

import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from dash import Dash, dcc, html

from config import args


class Dataset:
    def __init__(self):
        self.dataset = self.get_dataset_as_dataframe()
        self.get_eda()
        print(args.data_path)

    @staticmethod
    def get_dashboard(figures):
        app = Dash(__name__)  # TODO: prev, next; Boxes (turn_on); show labelled, unlabelled data;

        graphs_list = [dcc.Graph(id=f'image-{fig_id}', figure=fig) for fig_id, fig in enumerate(figures)]
        app.layout = html.Div([
            *graphs_list
        ])

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
        labeled_data_sample = self.dataset[~self.dataset.label.isna()].head(3)
        non_labeled_data_sample = self.dataset[self.dataset.label.isna()].head(3)

        figures = []
        for row in labeled_data_sample.iterrows():
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
            figures.append(fig)

        self.get_dashboard(figures)

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

    @staticmethod
    def read_labels(txt_files):
        labels = []
        for txt_file in txt_files:
            with open(os.path.join(args.data_path, txt_file), 'r') as f:
                text = f.readlines()
            if text:
                boxes = [[int(num) if num_id == 0 else float(num) for num_id, num in
                          enumerate(line.replace('\n', '').split(' '))] for line in text]
            else:
                boxes = np.nan
            labels.append(boxes)
        return labels


if __name__ == '__main__':
    dataset = Dataset()
