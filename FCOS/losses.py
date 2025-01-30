import torch
import torch.nn.functional as F
import boxops


def sigmoid_cross_entropy_with_logits(logits, labels):
    zeros = torch.zeros_like(logits, dtype=torch.float32)
    condition = logits >= zeros
    neg_abs_logits = torch.where(condition, -logits, logits)
    return F.relu(logits) - logits * labels + torch.log1p(torch.exp(neg_abs_logits))


class Focal(object):

    def __init__(self, gamma=2.0, alpha=0.25):
        self._gamma = gamma
        self._alpha = alpha

    def __call__(self, preds, targets):
        '''
        Args:
            preds: [N, M, C]
            targets: [N, M, C]
        return:
            loss: [N, M, C]
        '''

        p = torch.sigmoid(preds)
        p_t = targets * p + (1 - targets) * (1 - p)

        alpha_factor = targets * self._alpha + (1 - targets) * (1 - self._alpha)
        focal_weight = (1.0 - p_t) ** self._gamma

        ce = sigmoid_cross_entropy_with_logits(preds, targets)

        return alpha_factor * focal_weight * ce


class L1(object):

    @property
    def encoded(self):
        return True

    @property
    def decoded(self):
        return False

    def __call__(self, preds, targets):
        '''
        Args:
            preds: [N, M, 4]
            targets: [N, M, 4]
        return:
            loss: [N, M]
        '''

        loss = torch.abs(preds - targets)
        return torch.sum(loss, dim=-1)


class CenternessLoss(object):

    def __call__(self, preds, targets):
        return F.binary_cross_entropy_with_logits(preds, targets, reduction='none').sum(dim=-1)


class IOULoss(object):

    def __init__(self, name='ciou'):
        assert name in ['iou', 'giou', 'diou', 'ciou']

        self._fn = getattr(boxops, name)

    @property
    def encoded(self):
        return False

    @property
    def decoded(self):
        return True

    def __call__(self, preds, targets):
        '''
        Args:
            preds: [N, M, 4],
            targets: [N, M, 4],
        return:
            loss: [N, M]
        '''

        return 1.0 - self._fn(preds, targets)


