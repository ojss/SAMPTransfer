import numpy as np
import scipy as sp
import torch

from .label_cleansing_utils import remaining_labels, rank_per_class, label_denoising, compute_optimal_transport, \
    update_plabels, preprocess_e2e, dim_reduce


#

def iter_balanced(opt, support_features, support_ys, query_features, query_ys, labelled_samples, support_xs=None,
                  query_xs=None, model=None):
    query_ys_updated = query_ys
    total_f = support_ys.shape[0] + query_ys.shape[0]
    # iterations = query_ys.shape[0]# int(query_ys.shape[0] / (opt.n_ways*opt.best_samples))
    iterations = int(query_ys.shape[0] / (opt["n_ways"] * opt["best_samples"]))
    for j in range(iterations):
        query_ys_pred, probs, weights = update_plabels(opt, support_features, support_ys, query_features)
        # ------------------------probability pre-filtering for distractors-------------------------------------------------------------------
        rank = sp.stats.rankdata(weights, method='ordinal')
        # keep = (query_ys.shape[0] - 150)/opt.n_ways
        indices_pre, ys = rank_per_class(support_ys.max() + 1, rank, query_ys_pred, 30)
        # -----------------------------------------------------------------------------------------------------------

        P, query_ys_pred, indices = compute_optimal_transport(opt, torch.Tensor(probs))
        loss_statistics, _ = label_denoising(opt, support_features, support_ys, query_features[indices_pre],
                                             query_ys_pred[indices_pre])
        # loss_statistics, _ = label_denoising(opt, support_features, support_ys, query_features, query_ys_pred, weights=torch.tensor(weights))

        un_loss_statistics = loss_statistics[support_ys.shape[0]:].detach().numpy()  # np.amax(P, 1) #
        un_loss_statistics = un_loss_statistics  # *weights[support_ys.shape[0]:]
        rank = sp.stats.rankdata(un_loss_statistics, method='ordinal')
        indices, ys = rank_per_class(support_ys.max() + 1, rank, query_ys_pred[indices_pre], opt["best_samples"])
        # print(indices)
        indices = indices_pre[indices]
        # print(indices)
        # if len(indices)<5:
        #    break;
        pseudo_mask = np.in1d(np.arange(query_features.shape[0]), indices)
        pseudo_features, query_features = query_features[pseudo_mask], query_features[~pseudo_mask]
        pseudo_ys, query_ys_pred = query_ys_pred[pseudo_mask], query_ys_pred[~pseudo_mask]
        query_ys_concat, query_ys_updated = query_ys_updated[pseudo_mask], query_ys_updated[~pseudo_mask]
        support_features = np.concatenate((support_features, pseudo_features), axis=0)
        support_ys = np.concatenate((support_ys, pseudo_ys), axis=0)
        query_ys = np.concatenate((query_ys, query_ys_concat), axis=0)
        if support_features.shape[0] == total_f or len(indices_pre) == 0:
            # print(support_features.shape[0], total_f)
            break
    # if opt.distractor == False:
    support_ys = np.concatenate((support_ys, query_ys_pred), axis=0)
    support_features = np.concatenate((support_features, query_features), axis=0)
    # query_ys = np.concatenate((query_ys, query_ys_updated), axis=0)
    # query_ys_pred = support_ys[labelled_samples:]
    # query_ys = query_ys[query_ys_pred.shape[0]:]
    return support_ys, support_features


def iter_balanced_trans(opt, support_features, support_ys, query_features, query_ys, labelled_samples):
    query_ys_updated = query_ys
    total_f = support_ys.shape[0] + query_ys.shape[0]
    iterations = int(query_ys.shape[0])  # int(opt.n_ways*opt.best_samples) #query_ys.shape[0]#
    # clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
    # clf = LogisticRegression(C=10, multi_class='auto', solver='lbfgs', max_iter=1000)
    for j in range(iterations):
        # clf.fit(support_features, support_ys)
        # probs = clf.predict_proba(query_features)
        query_ys_pred, probs, weights = update_plabels(opt, support_features, support_ys, query_features)
        P, query_ys_pred, indices = compute_optimal_transport(opt, torch.Tensor(probs))

        loss_statistics, _ = label_denoising(opt, support_features, support_ys, query_features, query_ys_pred)
        un_loss_statistics = loss_statistics[support_ys.shape[0]:].detach().numpy()  # np.amax(P, 1) #
        rank = sp.stats.rankdata(un_loss_statistics, method='ordinal')

        # rank = sp.stats.rankdata(weights, method='ordinal')

        indices, ys = rank_per_class(support_ys.max() + 1, rank, query_ys_pred, opt["best_samples"])
        if len(indices) < 5:
            break
        pseudo_mask = np.in1d(np.arange(query_features.shape[0]), indices)
        pseudo_features, query_features = query_features[pseudo_mask], query_features[~pseudo_mask]
        pseudo_ys, query_ys_pred = query_ys_pred[pseudo_mask], query_ys_pred[~pseudo_mask]
        query_ys_concat, query_ys_updated = query_ys_updated[pseudo_mask], query_ys_updated[~pseudo_mask]
        support_features = np.concatenate((support_features, pseudo_features), axis=0)
        support_ys = np.concatenate((support_ys, pseudo_ys), axis=0)
        query_ys = np.concatenate((query_ys, query_ys_concat), axis=0)
        # if opt.unbalanced:
        remaining_labels(opt, pseudo_ys)
        # print(sum(opt.no_samples), query_features.shape[0])
        if support_features.shape[0] == total_f:
            # print(support_features.shape[0], total_f)
            break
    support_ys = np.concatenate((support_ys, query_ys_pred), axis=0)
    support_features = np.concatenate((support_features, query_features), axis=0)
    query_ys = np.concatenate((query_ys, query_ys_updated), axis=0)
    query_ys_pred = support_ys[labelled_samples:]
    query_ys = query_ys[query_ys_pred.shape[0]:]
    return query_ys, query_ys_pred


def label_finetuning(params: dict,
                     support_features: torch.Tensor,
                     y_support: torch.Tensor,
                     y_query: torch.Tensor,
                     query_features: torch.Tensor):
    with torch.no_grad():
        params["no_samples"] = np.array(np.repeat(float(y_query.shape[0] / params["n_ways"]), params["n_ways"]))

        if params["use_pt"]:
            combined = torch.cat([support_features, query_features])
            X = preprocess_e2e(combined, params["beta_pt"])
            support_features, query_features = X[:support_features.shape[0]], X[support_features.shape[0]:]
    support_features = support_features.detach().cpu().numpy()
    query_features = query_features.detach().cpu().numpy()

    if params["reduce"] is not None:
        support_features, query_features = dim_reduce(params, support_features, query_features)

    y_support = y_support.detach().cpu().numpy()
    y_query = y_query.detach().cpu().numpy()

    labelled_samples = y_support.shape[0]
    y_query, y_query_pred = iter_balanced_trans(params,
                                                support_features,
                                                y_support,
                                                query_features, y_query,
                                                labelled_samples)
    return y_query, y_query_pred
