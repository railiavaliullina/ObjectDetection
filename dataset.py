import os
import shutil
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image


class Dataset:
    def __init__(self, cfg, logger):
        """
        class for collecting dataset info and stats, which are used for visualization
        :param cfg: 'dataset' part of config
        :param logger: logger object
        """
        self.cfg = cfg['dataset']
        self.logger = logger

        self.logger.info('initializing dataset as dataframe')
        self.dataset = self.get_dataset_as_dataframe()

        self.logger.info('getting dataset stats for EDA')
        # collect dataset stats and info for eda_plots
        self.get_common_stats()

        if self.cfg['split_data']:
            self.logger.info('splitting dataset for training')
            self.split_data()

    def __len__(self):
        """
        :return: size of dataset
        """
        return len(self.dataset)

    def get_dataset_as_dataframe(self):
        """
        saves dataset as dataframe with columns (img_path, label), each row is associated with image
        :return: dataset dataframe
        """
        files_list = np.asarray(os.listdir(self.cfg['data_path']))
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
        self.size = len(dataset_df)
        return dataset_df

    def read_labels(self, txt_files):
        """
        reads bboxes info
        :param txt_files:
        :return: bboxes info
        """
        labels = []
        for txt_file in txt_files:
            with open(os.path.join(self.cfg['data_path'], txt_file), 'r') as f:
                text = f.readlines()
            if text:
                boxes = [[int(num) if num_id == 0 else float(num) for num_id, num in
                          enumerate(line.replace('\n', '').split(' '))] for line in text]
            else:
                boxes = np.nan
            labels.append(boxes)
        return labels

    @staticmethod
    def get_sizes_stats(sizes):
        """
        Gets stats about images and boxes sizes
        :param sizes: list of sizes (w, h)
        :return: aggregated stats
        """
        sizes = np.asarray(sizes)
        widths, heights = sizes[:, 0], sizes[:, 1]
        sizes_stats = {'avg_w': int(round(widths.mean())), 'avg_h': int(round(heights.mean())),
                       'min_w': int(round(widths.min())), 'min_h': int(round(heights.min())),
                       'max_w': int(round(widths.max())), 'max_h': int(round(heights.max()))}
        return sizes_stats

    def get_common_stats(self):
        """
        collect images samples with bboxes to visualize in dashboard
        and get common stats about dataset
        """
        images_sizes, bboxes_sizes = [], []
        images_aspect_ratios, bboxes_aspect_ratios = [], []
        num_boxes_per_image, boxes_area, classes_id = [], [], []

        self.labeled_data_sample = self.dataset[~self.dataset.label.isna()]
        self.non_labeled_data_sample = self.dataset[self.dataset.label.isna()]

        for row in tqdm(self.dataset.iterrows()):
            image = Image.open(os.path.join(self.cfg['data_path'], row[1]['img_path']))
            image_w, image_h = image.size
            images_sizes.append(image.size)
            images_aspect_ratios.append(image_h / image_w)

            # draw boxes
            if np.isnan(row[1]['label']).any():
                continue

            num_boxes_per_image.append(len(row[1]['label']))
            for label in row[1]['label']:
                object_class, x, y, width, height = label
                bbox_w, bbox_h = image_w * width, image_h * height
                bboxes_sizes.append((bbox_w, bbox_h))
                bboxes_aspect_ratios.append(bbox_h / bbox_w)
                boxes_area.append(bbox_w * bbox_h)
                classes_id.append(self.cfg['class_id_2_class_name_mapping'][str(object_class)])

        self.images_sizes_stats = self.get_sizes_stats(images_sizes)
        self.bboxes_sizes_stats = self.get_sizes_stats(bboxes_sizes)
        self.images_aspect_ratios = pd.DataFrame({'Images aspect ratio': images_aspect_ratios})
        self.bboxes_aspect_ratios = pd.DataFrame({'Boxes aspect ratio': bboxes_aspect_ratios, 'Class': classes_id})

        self.labeled_and_non_labeled_data_num = pd.DataFrame()
        self.labeled_and_non_labeled_data_num['labeled_or_not'] = ['With', 'Without']
        self.labeled_and_non_labeled_data_num['num_images'] = [len(self.labeled_data_sample),
                                                               len(self.non_labeled_data_sample)]
        objects_id_counter = Counter(
            np.concatenate(self.dataset.label[~self.dataset.label.isna()].to_list())[:, 0].astype(int))
        objects_id_dict = {
            f'{self.cfg["class_id_2_class_name_mapping"][str(object_id)]}': boxes_num for
            object_id, boxes_num in objects_id_counter.items()}

        self.objects_id_counter = pd.DataFrame()
        self.objects_id_counter['class'] = list(objects_id_dict.keys())
        self.objects_id_counter['boxes_num'] = list(objects_id_dict.values())

        self.classes_num = len(objects_id_dict)
        self.bboxes_num = sum(self.objects_id_counter['boxes_num'])
        self.num_boxes_per_image = pd.DataFrame({'Number of Boxes per Image': num_boxes_per_image})
        self.boxes_area = pd.DataFrame({'Boxes Area': boxes_area, 'Class': classes_id})

    def split_data(self):
        """
        Splits data into train and valid sets
        """
        # split dataset ids
        np.random.seed(int(self.cfg['seed']))
        dataset_size = len(self.dataset)
        sets_division_thr = np.ceil(dataset_size * float(self.cfg["train_set_part"])).astype(int)
        shuffled_ids = np.random.permutation(np.arange(dataset_size))
        set_ids_dict = {'train': shuffled_ids[:sets_division_thr],
                        'test': shuffled_ids[sets_division_thr:]}

        # generate files with train/valid sets paths which will be used while training
        images_names = self.dataset.img_path.to_numpy()
        for set_type, set_ids in set_ids_dict.items():
            set_img_names = [os.path.join(self.cfg['path_prefix'] + f'{set_type}_set/', img_name) + '\n' for
                             img_name in images_names[set_ids]]
            with open(os.path.join(self.cfg['dir_to_save'], f'{set_type}.txt'), 'w') as f:
                f.writelines(set_img_names)

            if not os.path.exists(self.cfg[f'{set_type}_set_path']):
                os.makedirs(self.cfg[f'{set_type}_set_path'])

            for img_name in images_names[set_ids]:
                shutil.copyfile(os.path.join(self.cfg['data_path'], img_name),
                                os.path.join(self.cfg[f'{set_type}_set_path'], img_name))
                txt_name = img_name.replace('.jpg', '.txt')
                shutil.copyfile(os.path.join(self.cfg['data_path'], txt_name),
                                os.path.join(self.cfg[f'{set_type}_set_path'], txt_name))
