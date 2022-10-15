import json
import logging
import sys

from dataset import Dataset
from evaluation import Evaluation
from dashboard import Dashboard


def init_logger():
    # define logger
    logger_obj = logging.getLogger()
    logger_obj.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger_obj.addHandler(handler)
    return logger_obj


def init_config():
    logger.info('initializing config')
    with open('config.json') as json_file:
        config = json.load(json_file)
    return config


if __name__ == '__main__':
    logger = init_logger()
    cfg = init_config()
    dataset = Dataset(cfg=cfg, logger=logger)
    evaluation_results = Evaluation(cfg=cfg, logger=logger).parsed_logs
    dashboard_obj = Dashboard(cfg=cfg, logger=logger, dataset=dataset, evaluation_results=evaluation_results)
    dashboard_obj()
