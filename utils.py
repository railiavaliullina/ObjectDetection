import json
import logging
import sys


def init_logger():
    """
    Defines logger
    :return: logger object
    """
    logger_obj = logging.getLogger()
    logger_obj.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger_obj.addHandler(handler)
    return logger_obj


def init_config(logger):
    """
    Reads json config
    :return: config
    """
    logger.info('initializing config')
    with open('config.json') as json_file:
        config = json.load(json_file)
    return config
