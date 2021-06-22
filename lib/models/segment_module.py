import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
import os

import matplotlib


class SegmentsModule(pl.LightningModule):
    def __init__(self, backbone, criterion, label, metrics=[], is_siam=False, lr=1e-3):
        super().__init__()
        self.backbone = backbone
        self.train_metrics = [deepcopy(metric) for metric in metrics if metric.on_train]
        self.val_metrics = [deepcopy(metric) for metric in metrics if metric.on_val]
        self.criterion = criterion
        self.labels = label if type(label) == list else [label]
        self.is_siam = is_siam
        self.lr = lr

    def forward(self, batch_data):
        acc = batch_data['acc']
        ppg = batch_data['ppg']

        if len(self.labels) == 1 and self.labels[0] == 'hr_label':
            x = torch.cat((acc, ppg), -1)
        if (len(self.labels) == 1 and self.labels[0] == 'activity_label') \
                or self.is_siam:
            x = acc

        if len(x.shape) == 3:
            x = x.permute(0, 2, 1)
        elif len(x.shape) == 2:
            x = x.permute(1, 0).unsqueeze(0)
        elif len(x.shape) == 4:  # transform to shape (3, batch_size, cnt_of_input_channels, window_size)
            x = x.permute(1, 0, 3, 2)

        if self.is_siam:
            embeds = []
            for single_x in x:
                embeds.append(self.backbone.forward(single_x))
            prediction = torch.stack(embeds)
        else:
            prediction = self.backbone.forward(x)

        return prediction

    def _to_cpu(self, items):
        if type(items) == torch.Tensor:
            return items.cpu()
        if (type(items) == tuple) or (type(items) == list):
            return [self._to_cpu(item) for item in items]
        if type(items) == dict:
            return {key: self._to_cpu(val) for key, val in items.items()}
        return items

    def _get_target(self, batch_data):
        if len(self.labels) == 1:
            return batch_data[self.labels[0]]
        elif len(self.labels) > 1:
            return {label: batch_data[label] for label in self.labels}

    def training_step(self, batch_data, batch_idx):
        target = self._get_target(batch_data)
        prediction = self.forward(batch_data)

        loss = self.criterion(prediction, target)
        self.log('train_loss', loss, on_step=True)

        prediction = self._to_cpu(prediction)
        target = self._to_cpu(target)
        for metric in self.train_metrics:
            metric.update(prediction, target)

        return {'loss': loss}

    def validation_step(self, batch_data, batch_idx):
        target = self._get_target(batch_data)
        prediction = self.forward(batch_data)

        loss = self.criterion(prediction, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        prediction = self._to_cpu(prediction)
        target = self._to_cpu(target)
        for metric in self.val_metrics:
            metric.update(prediction, target)

        return {'loss': loss}

    def training_epoch_end(self, outs):
        for metric in self.train_metrics:
            metric_result = metric.compute()
        if type(self.logger) == pl.loggers.wandb.WandbLogger:
            for metric in self.train_metrics:
                metric_result = metric.compute()
                if type(metric_result) == list:
                    for result in metric_result:
                        self.logger.experiment.log({'train_' + result['name']: result['score']})
                else:
                    self.logger.experiment.log({'train_' + metric.name: metric_result})

    def validation_epoch_end(self, outs):
        for metric in self.val_metrics:
            metric_result = metric.compute()
        if type(self.logger) == pl.loggers.wandb.WandbLogger:
            for metric in self.val_metrics:
                metric_result = metric.compute()
                if type(metric_result) == list:
                    for result in metric_result:
                        self.logger.experiment.log({'val_' + result['name']: result['score']})
                else:
                    self.logger.experiment.log({'val_' + metric.name: metric_result})

    def evaluate(self, datamodule, on_experiments=True, prefix='best_model_'):
        scores = []
        mean_score = {}

        if on_experiments:
            all_pred = []
            all_target = []
            iterator = tqdm(enumerate(datamodule.valid_dataset.experiments), desc='Evaluating')
            for i, experiment in iterator:
                segments = datamodule.valid_dataset.get_processed_experiment(i)
                batch_item = {name: torch.stack([segment[name] for segment in segments])
                              for name in segments[0].keys()}

                target = self._get_target(batch_item)
                all_target.append(target)
                predictions_logits = self.forward(batch_item)
                all_pred.append(predictions_logits)

                for metric_ex in self.val_metrics:
                    metric = deepcopy(metric_ex)
                    score = metric(predictions_logits, target)
                    name = os.path.basename(experiment.path)
                    scores.append({name: {f'{metric.name}': score}})

            all_target = torch.cat(all_target)
            all_pred = torch.cat(all_pred)
            for metric_ex in self.val_metrics:
                metric = deepcopy(metric_ex)
                score = metric(all_pred, all_target)
                mean_score.update({metric.name: score})

        else:
            iterator = tqdm(datamodule.val_dataloader(), desc='Evaluating...')
            for batch_data in iterator:
                label = self._get_target(batch_data)
                prediction = self.forward(batch_data)

                prediction = self._to_cpu(prediction)
                label = self._to_cpu(label)

                for metric in self.val_metrics:
                    metric.update(prediction, label)

            for metric in self.val_metrics:
                score = metric.compute()
                mean_score.update({metric.name: score})

        if type(self.logger) == pl.loggers.wandb.WandbLogger:
            types_of_plots = [matplotlib.figure.Figure, matplotlib.axes.SubplotBase]
            for name, score in mean_score.items():
                if any([issubclass(type(score), plot_type) for plot_type in types_of_plots]):
                    self.logger.experiment.log({prefix + '_' + name: score})
                elif type(score) == list:
                    for score_item in score:
                        if any([issubclass(type(score_item['score']), plot_type) for plot_type in types_of_plots]):
                            self.logger.experiment.log({prefix + '_' + score_item['name']: score_item['score']})

        return scores, mean_score

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.backbone.parameters(), lr=self.lr)
        return optimizer
