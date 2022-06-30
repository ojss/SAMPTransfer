import einops
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import SupConLoss
from torch import nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torchmetrics.functional import accuracy
from tqdm.auto import tqdm

from optimal_transport.sinkhorn import Sinkhorn2
from utils.sup_finetuning import Classifier


def euclidean_distance(x, y):
    """
    x, y have shapes (batch_size, num_examples, embedding_size).
    x is prototypes, y are embeddings in most cases
    """
    return torch.sum((x.unsqueeze(2) - y.unsqueeze(1)) ** 2, dim=-1)


@torch.enable_grad()
def sinkhorned_finetuning(encoder, episode, device='cpu', proto_init=True,
                          freeze_backbone=False, finetune_batch_norm=False,
                          inner_lr=0.001, total_epoch=15, n_way=5):
    sk = Sinkhorn2()
    sup_con_loss = SupConLoss()
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
    z_a_i = encoder(x_a_i)
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
        # adding the weight norm hook to access trained prototypes at the end
        m = weight_norm(classifier.fc)
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
            selected_id = torch.from_numpy(rand_id[j: min(j + batch_size, support_size)]).to(device)

            # z_batch = x_a_i[selected_id]
            # y_batch = y_a_i[selected_id]
            z_batch = x_a_i
            y_batch = y_a_i
            #####################################
            output = encoder(z_batch)
            protos = m.weight_v
            protos = einops.rearrange(protos, "p e -> 1 p e")
            emb = einops.rearrange(output, "b e -> 1 b e")
            dists = euclidean_distance(protos, emb).squeeze()
            assignments = sk(dists.t())

            # output = classifier(output)
            loss = sup_con_loss(emb.squeeze(0), y_batch) + loss_fn(assignments, y_batch)
            # calculate distances to prototypes

            #####################################
            loss.backward()

            classifier_opt.step()

            if freeze_backbone is False:
                delta_opt.step()
    classifier.eval()
    encoder.eval()

    y_query = torch.tensor(np.repeat(range(n_way), n_query)).to(device)
    output = encoder(x_b_i)
    # calculate distances to prototypes
    protos = m.weight_v
    protos = einops.rearrange(protos, "p e -> 1 p e")
    emb = einops.rearrange(output, "b e -> 1 b e")
    # TODO: try with cosine
    dists = euclidean_distance(protos, emb).squeeze()
    assignments = sk(dists.t())
    # scores = classifier(output)

    loss = F.cross_entropy(assignments, y_query, reduction='mean')
    _, predictions = torch.max(assignments, dim=1)
    # acc = torch.mean(predictions.eq(y_query).float())
    acc = accuracy(predictions, y_query)
    return loss, acc.item()
