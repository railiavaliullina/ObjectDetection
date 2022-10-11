import numpy as np
import torch

from config import args


class Trainer:
    def __init__(self):
        self.get_data()
        self.get_model()
        self.build()

    def get_data(self):
        pass

    def get_model(self):
        self.model = None
        # print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
        # TODO: logging lib
        pass

    def build(self):
        self.get_optimizer()
        self.get_criterion()

    def get_optimizer(self):
        # self.opt = torch.optim.Adam(self.model.parameters(), 1e-3, weight_decay=1e-4)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=45, gamma=0.1)
        pass

    def get_criterion(self):
        pass

    def run_epoch(self, mode, is_train):
        torch.set_grad_enabled(is_train)
        self.model.train() if is_train else self.model.eval()

    def __call__(self):
        # start the training
        for epoch in range(args.epochs_num):
            print("epoch %d" % (epoch,))
            # self.scheduler.step(epoch)
            self.run_epoch(mode='test', is_train=False)
            self.run_epoch(mode='train', is_train=True)
        pass


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer()
    trainer()
