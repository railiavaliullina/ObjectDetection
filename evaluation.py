import os
import pickle
import numpy as np


class Evaluation:
    def __init__(self, cfg, logger):
        """
        class for parsing logs from training, collecting metrics and loss values
        :param cfg: "evaluation" part of config
        :param logger: logger object
        """

        self.cfg = cfg['evaluation']
        self.logger = logger
        self.parse_logs()
        self.predictions_results_paths = np.asarray(os.listdir(os.path.join(self.cfg["predictions_results_main_path"],
                                                                            f'thr_{self.cfg["default_thr"]}')))
        self.thresholds_num = len(self.cfg['thresholds'])
        self.get_meanAP_and_loss_values()
        self.read_train_and_inference_details()

    def parse_logs(self):
        """
        parses logs from training, collects all metrics anf losses values
        """
        if self.cfg['load_parsed_logs']:
            self.parsed_logs = pickle.load(open(os.path.join(self.cfg['logs_dir']['parsed'], 'parsed_logs'), 'rb'))
            self.logger.info(f'successfully loaded parsed logs from {self.cfg["logs_dir"]["parsed"]}')
        else:
            self.logger.info('parsing logs to get training and validation info')
            self.parsed_logs = {}

            for log_id, log_path in enumerate(os.listdir(self.cfg['logs_dir']['raw'])):
                f = open(os.path.join(self.cfg['logs_dir']['raw'], log_path))
                lines = [line.rstrip("\n").strip() for line in f.readlines()]
                loss, validation_results, training_iteration = {}, {}, None

                for line in lines:
                    if line == '':
                        continue
                    line_splitted = line.split(':')
                    # check if there is info about training
                    try:
                        training_iteration = int(line_splitted[0])
                        avg_loss = float(line_splitted[1].split(',')[1].replace('avg loss', '').strip())
                        loss[training_iteration] = avg_loss

                    except ValueError:
                        if training_iteration:
                            # check if there is info about validation
                            if line.startswith('class_id = '):
                                parts = line.split(',')

                                if validation_results.get(training_iteration, None) is None:
                                    validation_results[training_iteration] = []

                                validation_results[training_iteration].append(
                                    {'class_id': int(parts[0].replace('class_id = ', '')),
                                     'class_name': parts[1].replace('name = ', ''),
                                     'ap': float(parts[2].split(' ')[3].replace('%', '')),
                                     'TP': int(parts[2].split(' ')[-1]),
                                     'FP': int(parts[3].replace('FP = ', '').replace(')', '').strip())
                                     }
                                )
                            elif line.startswith('for conf_thresh = '):
                                assert validation_results.get(training_iteration, None) is not None
                                parts = line.split(',')

                                try:
                                    results = {part.split(' ')[1]: float(part.split(' ')[-1]) for part in parts}
                                except ValueError:
                                    results = {part.split(' ')[1]: float(part.split(' ')[-1]) for part in parts[:-1]}
                                    results.update({'average_IoU': float(parts[-1].split(' ')[-2])})

                                validation_results[training_iteration].append(results)
                            elif line.startswith('mean average precision (mAP@0.50) = '):

                                assert validation_results.get(training_iteration, None) is not None
                                parts = line.split(',')
                                validation_results[training_iteration].append(
                                    {'mAP@0.50': float(parts[0].split(' ')[-1])})

                self.parsed_logs[log_id] = {'loss': loss, 'validation_results': validation_results}

            if self.cfg['save_parsed_logs']:
                pickle.dump(self.parsed_logs, open(os.path.join(self.cfg['logs_dir']['parsed'], 'parsed_logs'), 'wb'))
                self.logger.info(f'successfully saved parsed logs to {self.cfg["logs_dir"]["parsed"]}')

    def get_meanAP_and_loss_values(self):
        """
        prepares lists with loss and mAP values for visualization in dashboard
        """
        mean_ap_list = [(0, .0)]
        losses_list = []
        for log_id, log_dict in self.parsed_logs.items():
            for training_iter, validation_results in log_dict['validation_results'].items():
                if mean_ap_list and training_iter in np.asarray(mean_ap_list)[:, 0]:
                    continue
                mean_ap = validation_results[-1]['mAP@0.50'] * 100
                mean_ap_list.append((training_iter, mean_ap))
            for training_iter, loss in log_dict['loss'].items():
                if losses_list and training_iter in np.asarray(losses_list)[:, 0]:
                    continue
                losses_list.append((training_iter, loss))
        self.mean_ap_list = np.asarray(mean_ap_list)
        self.losses_list = np.asarray(losses_list)

    def read_train_and_inference_details(self):
        """
        reads info from 'best_result_info', 'training_details' files to show it in dashboard
        """
        self.best_result_info = open(self.cfg['best_result_info_path']).read()
        self.training_details = open(self.cfg['training_details_path']).read()
