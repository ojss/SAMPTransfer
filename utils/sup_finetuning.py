import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchmetrics.functional import accuracy
from tqdm.auto import tqdm


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

    def init_params_from_prototypes(self, z_support, n_way, n_support):
        z_support = z_support.contiguous()
        z_proto = z_support.view(n_way, n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        # Interpretation of ProtoNet as linear layer (see Snell et al. (2017))
        self._set_params(weight=2 * z_proto, bias=-torch.norm(z_proto, dim=-1) ** 2)


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
