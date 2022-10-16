import os
import pickle
import numpy as np


class Evaluation:
    def __init__(self, cfg, logger):

        self.cfg = cfg['evaluation']
        self.logger = logger
        self.parse_logs()
        self.predictions_results_paths = np.asarray(os.listdir(os.path.join(self.cfg["predictions_results_main_path"],
                                                                 f'thr_{self.cfg["default_thr"]}')))
        self.thresholds_num = len(self.cfg['thresholds'])

    def parse_logs(self):
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
                                validation_results[training_iteration].append({'mAP@0.50': float(parts[0].split(' ')[-1])})

                self.parsed_logs[log_id] = {'loss': loss, 'validation_results': validation_results}

            if self.cfg['save_parsed_logs']:
                pickle.dump(self.parsed_logs, open(os.path.join(self.cfg['logs_dir']['parsed'], 'parsed_logs'), 'wb'))
                self.logger.info(f'successfully saved parsed logs to {self.cfg["logs_dir"]["parsed"]}')
