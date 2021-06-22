"""
This script for train ResNet for Descriptor task
---------------------------------------------------------------------
Usage:
    python train_DE.py --epochs 1 --batch_size 512 --window_size 320 --step_size 32 --dropout_rate 0.5 --ignored_ids 1 2 --channels 4 64 64 128 256 512 32 --name "Descriptor ResNet" --nolog

"""

import sys
import argparse
from glob import glob

import torch
import torch.nn as nn
import pytorch_lightning as pl

sys.path.append('..')

from lib.data import SensorsDataModule
from lib.models import SegmentsModule
from lib.models import ResNet
from lib.training.metrics import DescriptorEvaluator, AccuracyScore
from lib.training.losses import triplet_loss_de

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from ptflops import get_model_complexity_info
import json

pl.seed_everything(42)


def main(args):
    if args.ignore_null:
        ignored_ids = [ignored_id - 1 for ignored_id in args.ignored_ids]
    else:
        ignored_ids = args.ignored_ids
    split_path = 'result_crossfit_data_split.pkl'
    data_module = SensorsDataModule(split_path=split_path,
                                    task='descriptor',
                                    ignore_train_ids=ignored_ids,
                                    processing_params={'sampling_rate': 100, 'slice_for_hr_label': (0.5, 1)},
                                    segment_params={'window_size': args.window_size, 'step_size': args.step_size},
                                    batch_size=args.batch_size, ignore_zero=args.ignore_null)

    desc_eval_metric = DescriptorEvaluator(special_classes=ignored_ids, n_spacial_examples=args.n_spacial_examples,
                                           on_train=False,
                                           metrics=['accuracy', 'f1', 'f2'],
                                           classify_models=['knn', 'random_forest', 'logreg'],
                                           classify_only_special=False,
                                           intersection_offset=10)
    criterion = triplet_loss_de(margin=1.0)

    resnet = ResNet(channels=args.channels, dropout_rate=args.dropout_rate, need_softmax=False)
    model = SegmentsModule(backbone=resnet,
                           criterion=criterion,
                           label=['activity_label', 'n_in_experiment', 'experiment_id', 'user_id'],
                           metrics=[desc_eval_metric],
                           is_siam=True,
                           lr=1e-5)

    if not args.nolog:
        wandb_logger = WandbLogger(name=args.name, project='polyn', log_model=True)
        wandb_logger.watch(model, log='parameters', log_freq=10)
        trainer = pl.Trainer(gpus=1, num_sanity_val_steps=0,
                             max_epochs=args.epochs, progress_bar_refresh_rate=20,
                             logger=wandb_logger, deterministic=True)
    else:
        trainer = pl.Trainer(gpus=1, num_sanity_val_steps=0,
                             max_epochs=args.epochs, progress_bar_refresh_rate=20,
                             deterministic=True)

    trainer.fit(model, data_module)

    # evaluate
    checkpoint_path = glob(f'{model.logger.save_dir}/{model.logger.name}/*/*/*')[-1]
    model.load_from_checkpoint(checkpoint_path=checkpoint_path, backbone=resnet,
                               criterion=criterion, label='hr_label')
    print(f'ckpt path: {checkpoint_path}')

    # only on spacial classes
    pl.seed_everything(42)
    desc_eval_metric = DescriptorEvaluator(special_classes=ignored_ids, n_spacial_examples=args.n_spacial_examples,
                                           on_train=False,
                                           metrics=['accuracy', 'confusion_matrix', 'tsne_visualize', 'f1', 'f2'],
                                           classify_models=['knn', 'random_forest', 'logreg'],
                                           classify_only_special=True,
                                           intersection_offset=10)  # 'confusion_matrix', 'tsne_visualize'
    model.val_metrics = [desc_eval_metric]
    model.backbone.eval()
    with torch.no_grad():
        scores, mean_scores = model.evaluate(data_module, on_experiments=False, prefix='best_model_new_classes')

    macs, params = get_model_complexity_info(model.backbone, (args.channels[0], args.window_size), as_strings=False,
                                             verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print(scores)
    print(mean_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='tmp', help='name of this experiment')
    parser.add_argument('-s', '--split_path', default='data/split_try.pkl', help='path to split_try.pkl')
    parser.add_argument('-d', '--device', type=int, default=0, help='id number of the cuda device to use')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='number of epochs to train the model')

    parser.add_argument('-ws', '--window_size', type=int, default=320, help='size of window')
    parser.add_argument('-ss', '--step_size', type=int, default=32, help='size of spep between windows')
    parser.add_argument('-dr', '--dropout_rate', type=float, default=0.1, help='dropout rate in ResNet')
    parser.add_argument('-ch', '--channels', nargs='+', type=int, help='count of channels in layers')
    parser.add_argument('-id', '--ignored_ids', nargs='+', type=int,
                        help='indexes on that will be tested descriptor evaluator on valid epoch')
    parser.add_argument('-se', '--n_spacial_examples', type=int, default=7, help='count of spatial examples for valid')
    parser.add_argument('-in', '--ignore_null', action='store_true', help='need to ignore null in dataset')

    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size for train and val')
    parser.add_argument('--nolog', action='store_true', help='turn off wandb logging')
    args = parser.parse_args()
    main(args)
