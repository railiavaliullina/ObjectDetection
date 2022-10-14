import json
import os
import shutil
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
from PIL import Image


class Dataset:
    def __init__(self):

        with open('dataset_config.json') as json_file:
            self.dataset_cfg = json.load(json_file)

        self.dataset = self.get_dataset_as_dataframe()
        self.prepare_data_for_eda()
        # self.split_data()

    def get_dataset_as_dataframe(self):
        files_list = np.asarray(os.listdir(self.dataset_cfg['data_path']))
        files_num = len(files_list)
        jpg_files_id = np.asarray([idx for idx, file_name in enumerate(files_list) if file_name.endswith('.jpg')])
        txt_files_id = np.setdiff1d(np.arange(files_num), jpg_files_id)

        # read labels
        txt_files = files_list[txt_files_id]
        labels = self.read_labels(txt_files)

        # make a pandas dataframe with images paths and labels
        dataset_df = pd.DataFrame()
        dataset_df['img_path'] = files_list[jpg_files_id]
        dataset_df['label'] = labels
        return dataset_df

    def read_labels(self, txt_files):
        labels = []
        for txt_file in txt_files:
            with open(os.path.join(self.dataset_cfg['data_path'], txt_file), 'r') as f:
                text = f.readlines()
            if text:
                boxes = [[int(num) if num_id == 0 else float(num) for num_id, num in
                          enumerate(line.replace('\n', '').split(' '))] for line in text]
            else:
                boxes = np.nan
            labels.append(boxes)
        return labels

    def prepare_data_for_eda(self):
        # collect images samples to visualize in dashboard
        labeled_data_sample = self.dataset[~self.dataset.label.isna()]
        non_labeled_data_sample = self.dataset[self.dataset.label.isna()]

        figures = []
        for row in labeled_data_sample.head(10).iterrows():
            image = Image.open(os.path.join(self.dataset_cfg['data_path'], row[1]['img_path']))
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

    def split_data(self):
        """
        Splits data into train and valid sets.
        :return:
        """
        # split dataset ids
        np.random.seed(int(self.dataset_cfg['seed']))
        dataset_size = len(self.dataset)
        sets_division_thr = np.ceil(dataset_size * float(self.dataset_cfg["train_set_part"])).astype(int)
        shuffled_ids = np.random.permutation(np.arange(dataset_size))
        set_ids_dict = {'train': shuffled_ids[:sets_division_thr],
                        'test': shuffled_ids[sets_division_thr:]}

        # generate files with train/valid sets paths which will be used while training
        images_names = self.dataset.img_path.to_numpy()
        for set_type, set_ids in set_ids_dict.items():
            set_img_names = [os.path.join(self.dataset_cfg['path_prefix'] + f'{set_type}_set/', img_name) + '\n' for
                             img_name in images_names[set_ids]]
            with open(os.path.join(self.dataset_cfg['dir_to_save'], f'{set_type}.txt'), 'w') as f:
                f.writelines(set_img_names)

            if not os.path.exists(self.dataset_cfg[f'{set_type}_set_path']):
                os.makedirs(self.dataset_cfg[f'{set_type}_set_path'])

            for img_name in images_names[set_ids]:
                shutil.copyfile(os.path.join(self.dataset_cfg['data_path'], img_name),
                                os.path.join(self.dataset_cfg[f'{set_type}_set_path'], img_name))
                txt_name = img_name.replace('.jpg', '.txt')
                shutil.copyfile(os.path.join(self.dataset_cfg['data_path'], txt_name),
                                os.path.join(self.dataset_cfg[f'{set_type}_set_path'], txt_name))


if __name__ == '__main__':
    dataset = Dataset()
