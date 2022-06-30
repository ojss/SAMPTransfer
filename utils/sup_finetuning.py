import copy

import einops
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses
from torch import nn
from torch.autograd import Variable
from torchmetrics.functional import accuracy
from tqdm.auto import tqdm

from optimal_transport.sot import SOT
from utils.proto_utils import get_prototypes, prototypical_loss


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()

        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

    def _set_params(self, weight, bias):
        state_dict = dict(weight=weight, bias=bias)
        self.fc.load_state_dict(state_dict)

    def init_params_from_prototypes(self, z_support, n_way, n_support, z_proto=None):
        z_support = z_support.contiguous()
        z_proto = z_support.view(n_way, n_support, -1).mean(
            1) if z_proto is None else z_proto  # the shape of z is [n_data, n_dim]
        # Interpretation of ProtoNet as linear layer (see Snell et al. (2017))
        self._set_params(weight=2 * z_proto, bias=-torch.norm(z_proto, dim=-1) ** 2)


def std_proto_form(self, batch, batch_idx):
    x_support = batch["train"][0]
    y_support = batch["train"][1]
    x_support = x_support
    y_support = y_support

    x_query = batch["test"][0]
    y_query = batch["test"][1]
    x_query = x_query
    y_query = y_query

    # Extract shots
    shots = int(x_support.size(1) / self.eval_ways)
    test_shots = int(x_query.size(1) / self.eval_ways)

    # Extract features (first dim is batch dim)
    x = torch.cat([x_support, x_query], 1)
    x = einops.rearrange(x, "1 b c h w -> b c h w")
    if not self.mpnn_opts["_use"]:
        z = self.model.backbone(x)
        z = einops.rearrange(z, "b c h w -> b (c h w)")
    elif self.mpnn_opts["_use"] and self.mpnn_opts["adapt"] == "ot":
        sot = SOT(distance_metric=self.distance)
        z = self.forward(x)
        z = torch.cat(self.re_represent(z, x_support.shape[1], self.alpha1, self.alpha2, self.re_rep_temp))
        z = sot.forward(z, n_samples=shots + test_shots, y_support=y_support)
    elif self.mpnn_opts["_use"] and self.mpnn_opts["adapt"] == "re_rep":
        _, z = self.mpnn_forward(x)
        z = torch.cat(self.re_represent(z, x_support.shape[1], self.alpha1, self.alpha2, self.re_rep_temp))
    else:
        # adapt instances by default
        _, z = self.mpnn_forward(x)
    z = einops.rearrange(z, "b e -> 1 b e")
    z_support = z[:, :self.eval_ways * shots]
    z_query = z[:, self.eval_ways * shots:]

    # Calucalte prototypes
    z_proto = get_prototypes(z_support, y_support, self.eval_ways)
    # implementing GAT based adaptation:
    if self.mpnn_opts["_use"] and self.mpnn_opts["adapt"] == "task":
        z_proto, z_query = einops.rearrange(z_proto, "1 b e -> b e"), einops.rearrange(z_query, "1 b e -> b e")
        combined = torch.cat([z_proto, z_query])
        edge_attr, edge_index, combined = self.graph_generator.get_graph(combined, Y=None)
        _, (combined,) = self.gnn(combined, edge_index, edge_attr, self.mpnn_opts["output_train_gnn"])
        z_proto, z_query = combined.split([self.eval_ways, len(z_query)])  # split based on number of prototypes
        z_proto, z_query = einops.rearrange(z_proto, "b e -> 1 b e"), einops.rearrange(z_query, "b e -> 1 b e")
    elif self.mpnn_opts["_use"] and self.mpnn_opts["adapt"] == "proto_only":
        # adapt only the prototypes? like FEAT
        z_proto = einops.rearrange(z_proto, "1 b e -> b e")
        edge_attr, edge_index, z_proto = self.graph_generator.get_graph(z_proto)
        _, (z_proto,) = self.gnn(z_proto, edge_index, edge_attr, self.mpnn_opts["output_train_gnn"])
        z_proto = einops.rearrange(z_proto, "b e -> 1 b e")

    # Calculate loss and accuracies
    loss, acc = prototypical_loss(z_proto, z_query, y_query,
                                  distance=self.distance)
    return loss, acc


@torch.enable_grad()
def supervised_finetuning(encoder, episode, device='cpu', proto_init=True,
                          freeze_backbone=False, finetune_batch_norm=False,
                          inner_lr=0.001, total_epoch=15, n_way=5):
    x_support = episode['train'][0][0]  # only take data & only first batch
    x_support = x_support.to(device)
    x_support_var = Variable(x_support)
    x_query = episode['test'][0][0]  # only take data & only first batch
    x_query = x_query.to(device)
    x_query_var = Variable(x_query)
    n_support = x_support.shape[0] // n_way
    n_query = x_query.shape[0] // n_way

    batch_size = n_way
    support_size = n_way * n_support

    y_a_i = Variable(torch.from_numpy(np.repeat(range(n_way), n_support))).to(
        device)  # (25,)
    x_b_i = x_query_var
    x_a_i = x_support_var

    encoder.eval()
    z_a_i = encoder(x_a_i).flatten(1)
    encoder.train()

    # Define linear classifier
    input_dim = z_a_i.shape[1]
    classifier = Classifier(input_dim, n_way=n_way)
    classifier.to(device)
    classifier.train()
    ###############################################################################################
    loss_fn = nn.CrossEntropyLoss().to(device)
    # Initialise as distance classifer (distance to prototypes)
    if proto_init:
        classifier.init_params_from_prototypes(z_a_i, n_way, n_support)
    classifier_opt = torch.optim.Adam(classifier.parameters(), lr=inner_lr)
    if freeze_backbone is False:
        delta_opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, encoder.parameters()), lr=inner_lr)
        # TODO: add provisions for EdgeConv layer here until then freeze_backbone=False
    # Finetuning
    if freeze_backbone is False:
        encoder.train()
    else:
        encoder.eval()
    classifier.train()
    if not finetune_batch_norm:
        for module in encoder.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()

    for epoch in tqdm(range(total_epoch), total=total_epoch, leave=False):
        rand_id = np.random.permutation(support_size)

        for j in range(0, support_size, batch_size):
            classifier_opt.zero_grad()
            if freeze_backbone is False:
                delta_opt.zero_grad()

            #####################################
            selected_id = torch.from_numpy(
                rand_id[j: min(j + batch_size, support_size)]).to(device)

            z_batch = x_a_i[selected_id]
            y_batch = y_a_i[selected_id]
            #####################################
            output = encoder(z_batch).flatten(1)

            output = classifier(output)
            loss = loss_fn(output, y_batch)

            #####################################
            loss.backward()

            classifier_opt.step()

            if freeze_backbone is False:
                delta_opt.step()
    classifier.eval()
    encoder.eval()

    y_query = torch.tensor(np.repeat(range(n_way), n_query)).to(device)
    output = encoder(x_b_i).flatten(1)

    scores = classifier(output)

    loss = F.cross_entropy(scores, y_query, reduction='mean')
    _, predictions = torch.max(scores, dim=1)
    # acc = torch.mean(predictions.eq(y_query).float())
    acc = accuracy(predictions, y_query)
    return loss, acc.item()


@torch.enable_grad()
def proto_maml(self, batch, batch_idx):
    x_support = batch['train'][0][0]  # only take data & only first batch
    x_support = x_support.to(self.device)
    x_support_var = Variable(x_support)
    x_query = batch['test'][0][0]  # only take data & only first batch
    x_query = x_query.to(self.device)
    x_query_var = Variable(x_query)
    n_support = x_support.shape[0] // self.eval_ways
    n_query = x_query.shape[0] // self.eval_ways

    batch_size = self.eval_ways
    support_size = self.eval_ways * n_support
    y_supp = Variable(torch.from_numpy(np.repeat(range(self.eval_ways), n_support))).to(self.device)
    y_query = torch.tensor(np.repeat(range(self.eval_ways), n_query)).to(self.device)

    self.eval()
    _, z = self.mpnn_forward(torch.cat([x_support_var, x_query_var]))
    z_supp, _ = z.split([len(x_support), len(x_query)])
    classifier = Classifier(z_supp.shape[-1], self.eval_ways)
    classifier.init_params_from_prototypes(z_support=z_supp, n_way=self.eval_ways, n_support=n_support)
    classifier.to(self.device)
    ce_loss = nn.CrossEntropyLoss().to(self.device)
    sup_con_loss = losses.SupConLoss()
    local_model = copy.deepcopy(self.model)
    local_model.train()
    classifier.train()
    if not self.ft_freeze_backbone:
        # Only freeze the CNN backbone
        local_model.backbone.requires_grad_(False)
        local_model.gnn.requires_grad_(True)
    # TODO: should I use another projector layer here instead of touching the GAT?
    backbone_parameters = list(filter(lambda p: p.requires_grad, local_model.parameters()))
    classifier_params = list(classifier.parameters())
    delta_opt = torch.optim.Adam(backbone_parameters, lr=self.lr, weight_decay=self.weight_decay)
    classifier_opt = torch.optim.Adam(classifier_params, lr=self.sup_finetune_lr, weight_decay=self.weight_decay)

    for _ in range(self.sup_finetune_epochs):
        # MAML inner loop
        delta_opt.zero_grad()
        classifier_opt.zero_grad()
        _, outputs = local_model(torch.cat([x_support_var, x_query_var]))
        outputs, _ = outputs.split([len(x_support), len(x_query)])
        preds = classifier(outputs)
        loss1 = ce_loss(preds, y_supp)
        loss2 = sup_con_loss(outputs, y_supp)
        loss = loss1 + loss2
        loss.backward()
        delta_opt.step()
        classifier_opt.step()
    _, outputs = local_model(x_query_var)
    scores = classifier(outputs)
    loss = F.cross_entropy(scores, y_query, reduction="mean") + sup_con_loss(outputs, y_query, )
    _, predictions = torch.max(scores, dim=1)
    # run local_model on query points
    acc = accuracy(predictions, y_query)

    return loss, acc


class BaselineFinetune(nn.Module):
    """
    training the classifier for a single testing run
    """

    def __init__(self, n_ways=5, n_shots=5, n_aug_support_samples=5, n_queries=15, feat_dim=640):
        super(BaselineFinetune, self).__init__()
        self.n_way = n_ways
        self.n_support = n_aug_support_samples
        self.n_query = n_queries
        self.feat_dim = feat_dim
        self.alpha = 0.05 if self.n_way == 1 else 0.01
        self.lr = 0.5 if self.n_way == 1 else 1.0

    def forward(self, z_support, y_support, z_query, spatial=False, weight_inprint=False):
        torch.manual_seed(0)
        kernel_size = (z_support.size(2), z_support.size(3)) if spatial else None
        y_support = y_support.view(-1)

        if spatial:
            assert z_support.dim() == 4
            assert z_query.dim() == 4
        else:
            if z_support.dim() == 4:
                z_support = z_support.view(z_support.size(0), z_support.size(1), -1).mean(-1)
                z_query = z_query.view(z_query.size(0), z_query.size(1), -1).mean(-1)

        z_support = F.normalize(z_support, dim=1, p=2)
        z_query = F.normalize(z_query, dim=1, p=2)

        if spatial:
            linear_clf = nn.Sequential(nn.AdaptiveMaxPool2d(2),
                                       nn.Conv2d(self.feat_dim, self.n_way, kernel_size=(2, 2), padding=0))
        else:
            linear_clf = nn.Linear(self.feat_dim, self.n_way)

        linear_clf = linear_clf.cuda()

        if weight_inprint:
            if spatial:
                z_support_pooled = nn.AdaptiveMaxPool2d(2)(z_support)
                prototypes = [z_support_pooled[y_support == l].mean(0) for l in y_support.unique()]
            else:
                prototypes = [z_support[y_support == l].mean(0) for l in y_support.unique()]
            prototypes = F.normalize(torch.stack(prototypes), dim=1, p=2)

            if spatial:
                linear_clf[1].weight.data.copy_(prototypes.data)
            else:
                linear_clf.weight.data.copy_(prototypes.data)

        set_optimizer = torch.optim.LBFGS(linear_clf.parameters(), lr=self.lr)
        loss_function = nn.CrossEntropyLoss().cuda()

        def closure():
            set_optimizer.zero_grad()
            scores = linear_clf(z_support).squeeze()
            loss = loss_function(scores, y_support)

            l2_penalty = 0
            for param in linear_clf.parameters():
                l2_penalty = l2_penalty + 0.5 * (param ** 2).sum()
            loss = loss + self.alpha * l2_penalty

            loss.backward(retain_graph=True)

            return loss

        set_optimizer.step(closure)

        scores = linear_clf(z_query).squeeze().softmax(1).detach()
        return scores
