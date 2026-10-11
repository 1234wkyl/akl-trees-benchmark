"""Experimental PointGroup loss for partially labelled training crops.

Unknown points remain network inputs but are absent from supervised targets.
This is weak supervision, not checked ground truth or a certified SAT trainer.
The pinned upstream source/checkpoint is never modified by this module.
"""
import torch
import torch.nn.functional as F


def known_embedding_loss(embeddings, instance_ids, semantics, batch):
    """Pinned three-head discriminative margins, restricted to known tree points."""
    mask = (semantics == 1) & (instance_ids > 0)
    zero = embeddings.sum() * 0
    if not mask.any():
        return {'ins_loss': zero, 'ins_var_loss': zero,
                'ins_dist_loss': zero, 'ins_reg_loss': zero}
    variances, distances, regularizers = [], [], []
    for sample in torch.unique(batch[mask]):
        centers, point_variances = [], []
        local = mask & (batch == sample)
        for ident in torch.unique(instance_ids[local]):
            values = embeddings[local & (instance_ids == ident)]
            center = values.mean(0); centers.append(center)
            point_variances.append((torch.norm(values-center, p=1, dim=1)-.5).clamp_min(0).square().mean())
        centers = torch.stack(centers)
        variances.append(torch.stack(point_variances).mean())
        if len(centers) > 1:
            pairwise = torch.norm(centers[:,None,:]-centers[None,:,:],p=1,dim=2)
            off_diagonal = ~torch.eye(len(centers),dtype=torch.bool,device=embeddings.device)
            distances.append((3.-pairwise[off_diagonal]).clamp_min(0).square().mean())
        else:
            distances.append(centers.sum()*0)
        regularizers.append(.001*torch.norm(centers,p=1,dim=1).mean())
    var, dist, reg = (torch.stack(v).mean() for v in (variances,distances,regularizers))
    return {'ins_loss':var+dist+reg,'ins_var_loss':var,'ins_dist_loss':dist,'ins_reg_loss':reg}


def known_instance_ious(clusters, instance_ids, semantics, batch, mask_probs=None):
    """Known-only proposal IoU against positive instance IDs within each sample.

    An unknown point is excluded from both intersection and proposal union.
    A completely unknown proposal contributes no score or mask supervision.
    GT extents and offset centers still reflect incomplete weak membership.
    """
    if not (instance_ids.shape == semantics.shape == batch.shape):
        raise ValueError('aligned point targets and batch IDs required')
    # The pinned CUDA image uses PyTorch 1.8, before torch.isin was added.
    if not ((semantics == -1) | (semantics == 0) | (semantics == 1)).all():
        raise ValueError('reader semantics must be -1 unknown, 0 non-tree, 1 tree')
    known = semantics != -1
    if (instance_ids[(semantics == 1)] <= 0).any() or (instance_ids[semantics != 1] != 0).any():
        raise ValueError('reader instance convention violated')
    tree_mask = known & (instance_ids > 0)
    pairs = torch.unique(torch.stack([batch[tree_mask], instance_ids[tree_mask]], dim=1), dim=0)
    ious = torch.zeros((len(clusters), len(pairs)), device=semantics.device, dtype=torch.float32)
    supervised = torch.zeros(len(clusters), device=semantics.device, dtype=torch.bool)
    offset = 0
    for i, indices in enumerate(clusters):
        if indices.numel() == 0:
            raise ValueError('empty proposal')
        if (indices < 0).any() or (indices >= len(semantics)).any():
            raise ValueError('proposal index outside point cloud')
        if torch.unique(batch[indices]).numel() != 1:
            raise ValueError('proposal crosses batch samples')
        selected = known[indices]
        supervised[i] = selected.any()
        if mask_probs is not None:
            selected = selected & (mask_probs[offset:offset+len(indices)] > .5)
        proposal_size = selected.sum()
        for j, (sample, ident) in enumerate(pairs):
            if batch[indices[0]] != sample:
                continue
            truth = tree_mask & (batch == sample) & (instance_ids == ident)
            overlap = (selected & truth[indices]).sum()
            denominator = proposal_size + truth.sum() - overlap
            ious[i, j] = overlap.float() / (denominator.float() + 1e-5)
        offset += len(indices)
    if mask_probs is not None and offset != len(mask_probs):
        raise ValueError('mask probabilities do not align to proposals')
    return ious, supervised, pairs


def weak_proposal_losses(clusters, scores, instance_ids, semantics, batch,
                         mask_probs=None, min_iou=.25, max_iou=.75,
                         iou_uses_mask=False):
    if not 0 <= min_iou < max_iou <= 1 or len(scores) != len(clusters):
        raise ValueError('valid score thresholds and one score per proposal required')
    ious, supervise, pairs = known_instance_ious(clusters, instance_ids, semantics, batch,
                                               mask_probs if iou_uses_mask else None)
    best, owner = (ious.max(dim=1) if len(pairs) else
                   (torch.zeros(len(clusters), device=scores.device),
                    torch.zeros(len(clusters), dtype=torch.long, device=scores.device)))
    targets = ((best - min_iou) / (max_iou - min_iou)).clamp(0, 1)
    score_loss = F.binary_cross_entropy(scores[supervise], targets[supervise]) if supervise.any() else scores.sum() * 0
    mask_loss = scores.sum() * 0
    if mask_probs is not None:
        point_targets, weights = [], []
        for i, indices in enumerate(clusters):
            valid = (semantics[indices] != -1) & supervise[i] & (best[i] > .5)
            target = torch.zeros(len(indices), device=scores.device)
            if len(pairs):
                sample, ident = pairs[owner[i]]
                target = ((batch[indices] == sample) & (instance_ids[indices] == ident)).float()
            point_targets.append(target); weights.append(valid.float())
        if clusters:
            targets_flat, weights_flat = torch.cat(point_targets), torch.cat(weights)
            if len(mask_probs) != len(targets_flat):
                raise ValueError('unaligned mask probabilities')
            element_loss = F.binary_cross_entropy(mask_probs, targets_flat, reduction='none')
            mask_loss = (element_loss * weights_flat).sum() / weights_flat.sum().clamp_min(1)
        else:
            mask_loss = mask_probs.sum() * 0
    return {'score_loss': score_loss, 'mask_loss': mask_loss,
            'ious': ious, 'score_supervision_mask': supervise}


def compute_weak_pointgroup_loss(model, epoch):
    """Method-compatible adapter for a disposable copy of the pinned model.

    Binding this method is a separate exploratory training implementation.
    It must pass a real model forward/backward smoke check before training.
    """
    y = model.labels.y.to(torch.int64)
    known = y != -1
    logits = model.output.semantic_logits
    model.semantic_loss = F.nll_loss(logits[known], y[known]) if known.any() else logits.sum() * 0
    mask = model.input.instance_mask.to(logits.device) & known
    offsets = model.output.offset_logits[mask]
    votes = model.input.vote_label.to(logits.device)[mask]
    model.offset_norm_loss = torch.abs(offsets-votes).sum() / (mask.sum()+1e-6)
    model.offset_dir_loss = -(F.normalize(offsets, dim=1, eps=1e-8) *
                              F.normalize(votes, dim=1, eps=1e-8)).sum() / (mask.sum()+1e-6)
    weights = model.opt.loss_weights
    model.loss = (weights['semantic'] * model.semantic_loss +
                  weights['offset_norm_loss'] * model.offset_norm_loss +
                  weights['offset_dir_loss'] * model.offset_dir_loss)
    if getattr(model.output, 'embed_logits', None) is not None:
        losses = known_embedding_loss(model.output.embed_logits,
            model.input.instance_labels.to(logits.device), y,
            model.input.batch.to(logits.device))
        for name, loss in losses.items():
            setattr(model, name, loss)
        model.loss += weights['embedding_loss'] * model.ins_loss
    if getattr(model.output, 'bi_semantic_logits', None) is not None:
        bi = model.output.bi_semantic_logits
        model.semantic_loss_bi = F.nll_loss(bi[known],y[known]) if known.any() else bi.sum()*0
        model.loss += weights['semantic'] * model.semantic_loss_bi
    model.score_loss, model.mask_loss = logits.sum()*0, logits.sum()*0
    if model.output.cluster_scores is not None and model._scorer_type:
        probs = (torch.sigmoid(model.output.mask_scores).reshape(-1)
                 if model.output.mask_scores is not None else None)
        losses = weak_proposal_losses(model.output.clusters, model.output.cluster_scores.reshape(-1),
            model.input.instance_labels.to(logits.device), y, model.input.batch.to(logits.device),
            probs, model.opt.min_iou_threshold, model.opt.max_iou_threshold,
            model.cal_iou_based_on_mask and epoch > model.cal_iou_based_on_mask_start_epoch)
        model.score_loss = losses['score_loss']
        model.loss += weights['score_loss'] * model.score_loss
        if probs is not None and model.mask_supervise:
            model.mask_loss = losses['mask_loss']
            model.loss += weights['mask_loss'] * model.mask_loss
