from neuralnet import VDSR_model
from utils.common import exists
import torch
import numpy as np

# -----------------------------------------------------------
#  SRCNN
# -----------------------------------------------------------

class VDSR:
    def __init__(self, device):
        self.device = device
        self.model = VDSR_model().to(device)
        self.optimizer = None
        self.loss =  None
        self.metric = None
        self.model_path = None
        self.ckpt_path = None
        self.ckpt_man = None

    def setup(self, optimizer, loss, metric, model_path, ckpt_path):
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.model_path = model_path
        self.ckpt_path = ckpt_path

    def load_checkpoint(self, ckpt_path):
        if not exists(ckpt_path):
            return
        self.ckpt_man = torch.load(ckpt_path)
        self.optimizer.load_state_dict(self.ckpt_man['optimizer'])
        self.model.load_state_dict(self.ckpt_man['model'])

    def load_weights(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=torch.device(self.device)))

    def predict(self, lr):
        self.model.train(False)
        sr = self.model(lr)
        return sr

    def evaluate(self, dataset, batch_size=64):
        losses, metrics = [], []
        isEnd = False
        self.model.eval()
        with torch.no_grad():
            while isEnd == False:
                lr, hr, isEnd = dataset.get_batch(batch_size, shuffle_each_epoch=False)
                lr, hr = lr.to(self.device), hr.to(self.device)
                sr = self.predict(lr)
                loss = self.loss(hr, sr).cpu()
                metric = self.metric(hr, sr).cpu()
                losses.append(loss)
                metrics.append(metric)

        metric = torch.mean(torch.tensor(metrics))
        loss = torch.mean(torch.tensor(losses))
        return loss, metric

    def train(self, train_set, valid_set, batch_size, 
              epochs, save_best_only=False):

        cur_epoch = 0
        if self.ckpt_man is not None:
            cur_epoch = self.ckpt_man['epoch']
        max_epoch = cur_epoch + epochs

        prev_loss = np.inf
        if save_best_only and exists(self.model_path):
            self.load_weights(self.model_path)
            prev_loss, _ = self.evaluate(valid_set)
            self.load_checkpoint(self.ckpt_path)

        while cur_epoch < max_epoch:
            # if cur_epoch % 20 == 0:
            #     self.optimizer.param_groups[0]["lr"] /= 10
            cur_epoch += 1
            loss_array = []
            metric_array = []
            isEnd = False
            while isEnd == False:
                lr, hr, isEnd = train_set.get_batch(batch_size)
                loss, metric = self.train_step(lr, hr)
                loss_array.append(loss.detach().numpy())
                metric_array.append(metric.detach().numpy())

            val_loss, val_metric = self.evaluate(valid_set)
            print(f"Epoch {cur_epoch}/{max_epoch}",
                  f"- loss: {np.mean(loss_array):.7f}",
                  f"- {self.metric.__name__}: {np.mean(metric_array):.3f}",
                  f"- val_loss: {val_loss:.7f}",
                  f"- val_{self.metric.__name__}: {val_metric:.3f}")

            torch.save({'epoch': cur_epoch,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                        }, self.ckpt_path)

            if save_best_only and val_loss > prev_loss:
                continue
            prev_loss = val_loss
            torch.save(self.model.state_dict(), self.model_path)
            print(f"Save model to {self.model_path}\n")

    def train_step(self, lr, hr):
        self.model.train()
        self.optimizer.zero_grad()

        lr, hr = lr.to(self.device), hr.to(self.device)
        sr = self.model(lr)

        loss = self.loss(hr, sr)
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.4 / self.optimizer.param_groups[0]["lr"]) 
        self.optimizer.step()

        metric = self.metric(hr, sr)
        loss = loss.cpu()
        metric = metric.cpu()
        return loss, metric
