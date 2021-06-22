import torch
from pytorch_lightning.metrics import Metric
from sklearn.metrics import fbeta_score
from pytorch_lightning.metrics.classification import Fbeta, Accuracy
from pytorch_lightning.metrics import MeanAbsoluteError

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import random
import numpy as np


class AccuracyScore(Accuracy):
    def __init__(self, name = 'accuracy', on_train=True, on_val=True):
        super().__init__()
        self.name = name
        self.on_train = on_train
        self.on_val = on_val

class MAELoss(MeanAbsoluteError):
    def __init__(self, name = 'mae', on_train=True, on_val=True):
        super().__init__()
        self.name = name
        self.on_train = on_train
        self.on_val = on_val

class F2Score(Fbeta):
    def __init__(self, num_classes=1, dist_sync_on_step=False, name='f2score', on_train=True, on_val=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, num_classes=num_classes, beta=2.0, average='macro')
        self.name = name
        self.on_train = on_train
        self.on_val = on_val

class AccuracyWithThresholdScore(Metric):
    def __init__(self, dist_sync_on_step=False, name = 'accuracy', threshold=0., on_train=True, on_val=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = name
        self.threshold = threshold
        self.on_train = on_train
        self.on_val = on_val
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _input_format(self, pred, target):

        if not (pred.shape == target.shape):
            # try to reshape predict (if taget is labels, but predict is class-probability)
            pred = torch.argmax(pred, -1).float()

        return pred, target

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = self._input_format(pred, target) 
        assert pred.shape == target.shape, f'prediction shape is ({pred.shape}), but target shape is ({target.shape})'

        is_in_threshold = ((pred - self.threshold) <= target) * (target <= (pred + self.threshold))
        self.correct += torch.sum(is_in_threshold)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


class DescriptorEvaluator(Metric):
    def __init__(self, dist_sync_on_step=False, name = 'desc_eval',
                 classify_models='knn', metrics=['accuracy', 'f1', 'f2'], 
                 on_train=False, on_val=True, 
                 special_classes=[1,2], n_spacial_examples=10, 
                 classify_only_special=False, intersection_offset=4):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        assert all([cls in ['knn', 'svc', 'logreg', 'random_forest', 'gauss'] for cls in classify_models]), 'this classify_model not supported'
        assert all([metric in ['accuracy', 'f1', 'f2', 'confusion_matrix', 'tsne_visualize'] for metric in metrics]), 'one of the metrics not supported'

        self.name = name
        self.classify_models = classify_models
        self.metrics = metrics
        self.on_train = on_train
        self.on_val = on_val
        self.special_classes = special_classes
        self.n_spacial_examples = n_spacial_examples
        self.k_usual_examples = 0.8
        self.classify_only_special = classify_only_special
        self.intersection_offset = intersection_offset
        self.window_size = 10

        self._cnt_epoch = 0

        self.add_state("embeds", default=[], dist_reduce_fx="sum")
        self.add_state("activity_labels", default=[], dist_reduce_fx="sum")
        self.add_state("n_in_experiments", default=[], dist_reduce_fx="sum")
        self.add_state("user_ids", default=[], dist_reduce_fx="sum")
        self.add_state("experiment_ids", default=[], dist_reduce_fx="sum")
        self.shuffle_ids = None

    def _input_format(self, pred, label):
        # first element of tensor is the embedding of item
        embed = pred[0].cpu()

        # if we got triplet of embeddings and labels
        activity_label = label['activity_label'][:, 0]
        n_in_experiment = label['n_in_experiment'][:, 0]
        experiment_id = label['experiment_id'][:, 0]
        user_id = label['user_id'][:, 0]
        
        return embed, activity_label, n_in_experiment, experiment_id, user_id

    def update(self, pred: torch.Tensor, label: torch.Tensor):
        embed, activity_label, n_in_experiment, experiment_id, user_id = self._input_format(pred, label)

        self.embeds += embed
        self.activity_labels += activity_label
        self.n_in_experiments += n_in_experiment
        self.user_ids += user_id
        self.experiment_ids += experiment_id

    def _get_datasets_for_classes(self):
        embeds = torch.stack(self.embeds)
        labels = torch.stack(self.activity_labels)
        n_in_experiments = torch.stack(self.n_in_experiments)
        experiment_ids = torch.stack(self.experiment_ids)
        user_ids = torch.stack(self.user_ids)

        experiment_masks = [experiment_ids == ex_id for ex_id in torch.unique(experiment_ids)]
        class_has_datasets = {}
        for experiment_mask in experiment_masks:

            experiment_embeds = embeds[experiment_mask]
            experiment_labels = labels[experiment_mask]
            n_in_experiment = n_in_experiments[experiment_mask]

            now_collect = 'train'
            activity_masks = [experiment_labels == label_id for label_id in torch.unique(experiment_labels)]
            for activity_mask in activity_masks:
                activity_embeds = experiment_embeds[activity_mask]
                activity_labels = experiment_labels[activity_mask]
                activity_label = int(activity_labels[0])

                split_train_test_ind = len(activity_labels) // 2
                if now_collect == 'train':
                    train_class_embeds = activity_embeds[:split_train_test_ind - (self.window_size // 2)]
                    test_class_embeds = activity_embeds[split_train_test_ind + (self.window_size // 2):]
                    now_collect = 'test'
                elif now_collect == 'test':
                    test_class_embeds = activity_embeds[:split_train_test_ind - (self.window_size // 2)]
                    train_class_embeds = activity_embeds[split_train_test_ind + (self.window_size // 2):]
                    now_collect = 'train'

                if activity_label not in class_has_datasets:
                    class_has_datasets[activity_label] = {'train_embeds': [], 'test_embeds': []}
                class_has_datasets[activity_label]['train_embeds'].append(train_class_embeds)
                class_has_datasets[activity_label]['test_embeds'].append(test_class_embeds)

        for label_id in class_has_datasets.keys():
            class_has_datasets[label_id]['train_embeds'] = torch.cat(class_has_datasets[label_id]['train_embeds'])
            class_has_datasets[label_id]['test_embeds'] = torch.cat(class_has_datasets[label_id]['test_embeds'])

        return class_has_datasets

    def _create_dataset(self):
        # split usual data to train/test
        class_has_datasets = self._get_datasets_for_classes()

        train_embeds = []
        train_labels = []
        test_embeds = []
        test_labels = []
        for class_ind in class_has_datasets.keys():
            if (class_ind in self.special_classes) or not self.classify_only_special:
                label = class_ind
            else:
                label = 0
                
            class_train_embeds = class_has_datasets[class_ind]['train_embeds']
            class_test_embeds = class_has_datasets[class_ind]['test_embeds']
            class_train_labels = torch.full([len(class_has_datasets[class_ind]['train_embeds'])], label)
            class_test_labels = torch.full([len(class_has_datasets[class_ind]['test_embeds'])], label)

            if (class_ind != 0) and (self.n_spacial_examples < len(class_train_embeds)):
                class_train_embeds, _, class_train_labels, _ = train_test_split(class_train_embeds, 
                                                                                class_train_labels,
                                                                                train_size=self.n_spacial_examples, 
                                                                                random_state=42)

            train_embeds.append(class_train_embeds)
            train_labels.append(class_train_labels)
            test_embeds.append(class_test_embeds)
            test_labels.append(class_test_labels)

        # stuck data to train/test dataset
        train_embeds = torch.cat(train_embeds).detach().numpy()
        train_labels = torch.cat(train_labels).detach().numpy()
        test_embeds = torch.cat(test_embeds).detach().numpy()
        test_labels = torch.cat(test_labels).detach().numpy()

        print('train_embeds.shape = ', train_embeds.shape)
        print('train_labels.shape = ', train_labels.shape)
        print('test_embeds.shape = ', test_embeds.shape)
        print('test_labels.shape = ', test_labels.shape)
        return train_embeds, train_labels, test_embeds, test_labels

    def _compute_scores_for_cls(self, classifier, embeds, labels):
        predictions = classifier['model'].predict(embeds)

        scores = []
        if 'accuracy' in self.metrics:
            score = accuracy_score(predictions, labels)
            scores.append({'score': score, 'name': f'{self.name}_accuracy'})

        if 'f1' in self.metrics:
            if self.classify_only_special:
                visible_labels = list(range(1, len(np.unique(labels))))
                score = fbeta_score(predictions, labels, average='macro', labels=visible_labels, beta=1) # weighted macro
            else:
                score = fbeta_score(predictions, labels, average='macro', beta=1)
            scores.append({'score': score, 'name': f'{self.name}_f1'})

        if 'f2' in self.metrics:
            score = fbeta_score(predictions, labels, average='macro', beta=2)
            scores.append({'score': score, 'name': f'{self.name}_f2'})

        if 'confusion_matrix' in self.metrics:
            fig, ax = plt.subplots()
            plot_confusion_matrix(classifier['model'], embeds, labels, ax=ax)
            scores.append({'score': fig, 'name': f'{self.name}_confusion_matrix_{self._cnt_epoch}_epoch'})

        return scores

    def _tsne_visualize(self, embeds, labels):
        ax, fig = plt.subplots()

        tsne_vecs = TSNE(n_components=2).fit_transform(embeds)
        tsne_vecs /= tsne_vecs.max()
        
        axes = plt.gca()
        axes.set_xlim([-1.1,1.1])
        axes.set_ylim([-1.1,1.1])

        random.seed(0)
        colors = {int(label): "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) 
                    for label in np.unique(labels)}

        # add dots in plot
        for (x, y), label in zip(tsne_vecs, labels):
            plt.plot(x, y, color=colors[int(label)], marker='o', linestyle='dashed', linewidth=2, markersize=4)

        # add colorized legend
        handles = [mpatches.Patch(color=color, label=f"id_{label}") for label, color in colors.items()]
        legend = plt.legend(handles=handles, loc='upper right')
        for legend_text, color in zip(legend.get_texts(), colors.values()):
            plt.setp(legend_text, color=color)
        
        return fig

    def compute(self):
        self._cnt_epoch += 1
        train_embeds, train_labels, test_embeds, test_labels = self._create_dataset()
        
        classifiers = []
        for classify_model in self.classify_models:
            # train classifiers
            if classify_model == 'knn':
                cls_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
                classifiers.append({'name': 'knn', 'model': cls_model})
            if classify_model == 'logreg':
                cls_model = LogisticRegression()
                classifiers.append({'name': 'logreg', 'model': cls_model})
            if classify_model == 'gauss':
                cls_model = GaussianNB()
                classifiers.append({'name': 'gauss', 'model': cls_model})
            if classify_model == 'svc':
                cls_model = SVC(gamma='auto', C=1, kernel='rbf')
                classifiers.append({'name': 'svc', 'model': cls_model})
            if classify_model == 'random_forest':
                cls_model = RandomForestClassifier(random_state=0)
                classifiers.append({'name': 'random_forest', 'model': cls_model})

        models_scores = []

        if 'tsne_visualize' in self.metrics:
            tsne_plt = self._tsne_visualize(test_embeds, test_labels)
            models_scores.append({'name': f'test_{self.name}_tsne_{self._cnt_epoch}_epoch', 'score': tsne_plt})

            tsne_plt = self._tsne_visualize(train_embeds, train_labels)
            models_scores.append({'name': f'train_{self.name}_tsne_{self._cnt_epoch}_epoch', 'score': tsne_plt})

        for classifier in classifiers:
            # train model
            classifier['model'].fit(train_embeds, train_labels)

            # calculate test scores for this model
            scores = self._compute_scores_for_cls(classifier, train_embeds, train_labels)
            for score in scores:
                models_scores.append({'name': f"train_{score['name']}_{classifier['name']}", 'score': score['score']})

            # calculate test scores for this model
            scores = self._compute_scores_for_cls(classifier, test_embeds, test_labels)
            for score in scores:
                models_scores.append({'name': f"test_{score['name']}_{classifier['name']}", 'score': score['score']})

        return models_scores