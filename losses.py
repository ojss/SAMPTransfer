import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


# cross entropy and center loss
class CrossEntropyDistill(torch.nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self):
        super(CrossEntropyDistill, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size, num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class KLDivWithLogSM(torch.nn.Module):
    def __init__(self):
        super(KLDivWithLogSM, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.KLDiv = torch.nn.KLDivLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size, num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        loss = self.KLDiv(log_probs, targets)
        return loss


class DistanceLoss(torch.nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()
        # self.l2_teacher = torch.nn.MSELoss(reduce=False)
        # self.l2_student = torch.nn.MSELoss(reduce=False)
        # print('mean')
        self.l2_teacher = torch.nn.SmoothL1Loss(reduce=False)
        self.l2_student = torch.nn.SmoothL1Loss(reduce=False)

    def forward(self, teacher, student):
        num_samps = teacher.shape[0]
        i1 = torch.tensor([i for i in range(num_samps) for j in range(num_samps)]).cuda(student.get_device())
        i2 = torch.tensor([j for i in range(num_samps) for j in range(num_samps)]).cuda(student.get_device())

        teacher_l2s = torch.sum(self.l2_teacher(teacher.index_select(0, i1), teacher.index_select(0, i2)), dim=1)
        teacher_l2s = teacher_l2s / teacher_l2s.mean()
        # print(teacher_l2s.cpu().tolist()[:num_samps])
        student_l2s = torch.sum(self.l2_student(student.index_select(0, i1), student.index_select(0, i2)), dim=1)
        student_l2s = student_l2s / student_l2s.mean()
        # print(student_l2s.cpu().tolist()[:num_samps])
        # quit()
        return torch.sum(torch.abs(teacher_l2s - student_l2s))


# cross entropy and center loss
class CrossEntropyLabelSmooth(torch.nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, dev=0, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.dev = dev
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).type_as(log_probs).scatter_(1, targets.unsqueeze(1).data, 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # print(targets, log_probs)
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=7, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        print("Focal Loss: Gamma = {}".format(gamma))
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class CenterLoss(torch.nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=512, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = torch.nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = torch.nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        # dist = []
        # for i in range(batch_size):
        #    value = distmat[i][mask[i]]
        #    value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
        #    dist.append(value)
        # dist = torch.cat(dist)
        # loss = dist.mean()
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        # Compute pairwise distance
        m, n = inputs.size(0), inputs.size(0)
        x = inputs.view(m, -1)
        y = inputs.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()

        dist.addmm_(1, -2, x, y.t())

        mask = targets.type(torch.ByteTensor).cuda()

        # for numerical stability
        # For each anchor, find the hardest positive and negative
        # mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            mask = targets == targets[i]
            dist_ap.append(dist[i][mask].max().unsqueeze(dim=0))  # hp
            dist_an.append(dist[i][mask == 0].min().unsqueeze(dim=0))  # hn
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()

        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec


class HLoss(nn.Module):
    def __init__(self, temperature_t: float, temperature_s: float):
        super(HLoss, self).__init__()
        self.temperature_t = temperature_t
        self.temperature_s = temperature_s

    def __call__(self, t: torch.Tensor, s: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        t = F.softmax((t.detach() - center) / self.temperature_t, dim=1)
        log_s = F.log_softmax(s / self.temperature_s, dim=1)

        return -(t * log_s).sum(dim=-1).mean()


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
