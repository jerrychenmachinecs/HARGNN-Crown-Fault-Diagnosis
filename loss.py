import torch
import torch.nn as nn
import torch.nn.functional as F

class ENSCostSensitiveLoss(nn.Module):
    def __init__(self, samples_per_cls, num_classes=3, beta=0.9999, gamma=1.0):
        super(ENSCostSensitiveLoss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.weights = self.compute_ens_weights()

    def compute_ens_weights(self):
        effective_num = 1.0 - torch.pow(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / torch.sum(weights) * self.num_classes
        return weights

    def forward(self, logits, labels):
        device = logits.device
        self.weights = self.weights.to(device)
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        weights = self.weights.unsqueeze(0).repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.num_classes)
        log_probs = F.log_softmax(logits, dim=1)
        loss = - (weights * labels_one_hot * log_probs).sum(dim=1)
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()
