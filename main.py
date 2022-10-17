import pickle
import argparse

from utils import init_logger, init_config
from dataset import Dataset
from evaluation import Evaluation
from dashboard import Dashboard


if __name__ == '__main__':

    # define logger and config
    logger = init_logger()
    cfg = init_config(logger)

    # if running from terminal with needed data-path
    if cfg["use_argparse"]:
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--data-path', required=True, type=str, help="Path to dataset dir")
        args = parser.parse_args()
        cfg["dataset"]["data_path"] = args.data_path

    # saving and loading dataset stats for speeding up
    if cfg["dataset"]["load_dataset_obj"]:
        dataset = pickle.load(open('/dataset', 'rb'))
    else:
        dataset = Dataset(cfg=cfg, logger=logger)
        if cfg["dataset"]["save_dataset_obj"]:
            pickle.dump(dataset, open('/dataset', 'wb'))

    # parse logs from training, get info about metrics
    evaluation = Evaluation(cfg=cfg, logger=logger)

    # visualization tool for eda and training results viewing and analysis
    dashboard_obj = Dashboard(cfg=cfg, logger=logger, dataset=dataset, evaluation=evaluation)
    dashboard_obj()
