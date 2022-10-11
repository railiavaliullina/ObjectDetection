import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-path', default='D:\dataset', type=str, help="Path to dataset directory")
parser.add_argument('-bs', '--batch-size', default=32, type=int, help="Batch size")
parser.add_argument('-e', '--epochs-num', default=10, type=int, help="Number of epochs")
parser.add_argument('-s', '--seed', default=42, type=int, help="Seed for reproducibility")
args = parser.parse_args()
