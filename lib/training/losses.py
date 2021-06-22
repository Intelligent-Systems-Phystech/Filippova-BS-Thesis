import torch
import torch.nn as nn


def triplet_loss_de(margin=1.0):
    def calculate_triplet_loss(pred, target, margin=margin):
        anchor = pred[0]
        positive = pred[1]
        negative = pred[2]
        loss = nn.TripletMarginLoss(margin=margin)(anchor, positive, negative)
        return loss

    return calculate_triplet_loss


def cross_entropy_loss_de(ignore_ids=[1, 2, 3]):
    """
    Loss for training descriptor on definit classes

    Args
    ----------
    pred: torch.array
        prediction
    target: torch.array
        target class
    ignore_ids: list
        list of ignored indexes tor train
    """

    assert len(ignore_ids) != 0, 'ignore_ids should be not empty'

    def calculate_cross_entropy_loss_de(pred, target, ignore_ids=ignore_ids):
        # pred is turple (pred_embed, pred_class)
        pred = pred[1]

        masks = torch.stack([target != label for label in ignore_ids])
        mask = sum(masks).bool()

        pred = pred[mask]
        target = target[mask]

        # if the windows consist of only ignore_ids in labels
        if len(pred) > 0:
            loss = nn.CrossEntropyLoss()(pred, target)
        else:
            loss = None
        return loss

    return calculate_cross_entropy_loss_de