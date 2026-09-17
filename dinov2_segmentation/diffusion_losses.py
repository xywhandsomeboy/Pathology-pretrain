"""Boundary-weighted diffusion auxiliary losses and tolerant boundary metrics.

Label boundaries weight the objective only; they never condition generation.
The objective is a mean of per-patch normalized losses. DDP uses an autograd
SUM of loss numerators and sample count, matching a serial global batch.
"""
import torch
import torch.nn.functional as F

DIFFUSION_METRICS = ('segmentation_loss', 'diffusion_noise_loss',
                     'diffusion_reconstruction_loss', 'diffusion_edge_loss', 'diffusion_aux_loss')
BOUNDARY_COUNTS = ('boundary_predicted_pixels', 'boundary_true_pixels',
                   'boundary_matched_prediction_pixels', 'boundary_matched_truth_pixels')


def label_boundary(labels, valid):
    edge = torch.zeros_like(valid)
    horizontal = (labels[:,:,1:] != labels[:,:,:-1]) & valid[:,:,1:] & valid[:,:,:-1]
    vertical = (labels[:,1:,:] != labels[:,:-1,:]) & valid[:,1:,:] & valid[:,:-1,:]
    edge[:,:,1:] |= horizontal; edge[:,:,:-1] |= horizontal
    edge[:,1:,:] |= vertical; edge[:,:-1,:] |= vertical
    return edge & valid


def dilate(mask, radius):
    if radius < 0:
        raise ValueError('boundary radius must be non-negative')
    if radius == 0:
        return mask
    return F.max_pool2d(mask[:,None].float(), 2*radius+1, stride=1, padding=radius)[:,0] > 0


def boundary_weights(target, ignore_index=255, boost=4., radius=2):
    if boost < 0:
        raise ValueError('boundary boost must be non-negative')
    valid = target != ignore_index
    band = dilate(label_boundary(target, valid), radius) & valid
    return valid.float() * (1 + boost * band.float()), valid


def _per_patch_mean(error, weight):
    return (error * weight).flatten(1).sum(1) / weight.flatten(1).sum(1).clamp_min(1)


def diffusion_auxiliary_loss(output, target, *, ignore_index=255, boundary_boost=4.,
                             boundary_radius=2, reconstruction_weight=.1,
                             edge_weight=.5, distributed=False):
    if reconstruction_weight < 0 or edge_weight < 0:
        raise ValueError('auxiliary loss component weights must be non-negative')
    weight, valid = boundary_weights(target, ignore_index, boundary_boost, boundary_radius)
    noise = (output['predicted_noise'].float() - output['noise'].float()).square().mean(1)
    # sqrt(alpha_bar) makes x0 errors stable even at the noisiest timesteps.
    scale = output['sqrt_alpha'].float()
    prediction = output['reconstruction'].float() * scale
    clean = output['clean'].float() * scale
    noise_loss = _per_patch_mean(noise, weight)
    reconstruction = _per_patch_mean((prediction-clean).abs().mean(1), weight)
    edge_numerator = prediction.new_zeros(prediction.shape[0])
    edge_denominator = prediction.new_zeros(prediction.shape[0])
    for axis in (2,3):
        error = (prediction.diff(dim=axis) - clean.diff(dim=axis)).abs().mean(1)
        if axis == 2:
            pair_weight = (weight[:,1:,:] + weight[:,:-1,:]) / 2
            pair_valid = valid[:,1:,:] & valid[:,:-1,:]
        else:
            pair_weight = (weight[:,:,1:] + weight[:,:,:-1]) / 2
            pair_valid = valid[:,:,1:] & valid[:,:,:-1]
        pair_weight = pair_weight * pair_valid
        edge_numerator += (error * pair_weight).flatten(1).sum(1)
        edge_denominator += pair_weight.flatten(1).sum(1)
    edge = edge_numerator / edge_denominator.clamp_min(1)
    stats = torch.stack((noise_loss.sum(), reconstruction.sum(), edge.sum(),
                         noise_loss.new_tensor(float(target.shape[0]))))
    if distributed:
        import torch.distributed as dist
        from torch.distributed.nn.functional import all_reduce
        if dist.is_initialized() and dist.get_world_size() > 1:
            stats = all_reduce(stats, op=dist.ReduceOp.SUM)
    means = stats[:3] / stats[3].clamp_min(1)
    auxiliary = means[0] + reconstruction_weight * means[1] + edge_weight * means[2]
    return auxiliary, dict(zip(('diffusion_noise_loss', 'diffusion_reconstruction_loss',
                               'diffusion_edge_loss', 'diffusion_aux_loss'),
                              (*means.unbind(), auxiliary)))


@torch.no_grad()
def boundary_counts(logits, target, ignore_index=255, tolerance=2):
    valid = target != ignore_index
    # Exclude an ignore-neighborhood so matching cannot bridge unlabeled areas.
    valid = valid & ~dilate(~valid, tolerance)
    truth = label_boundary(target, valid)
    prediction = label_boundary(logits.argmax(1), valid)
    matched_prediction = prediction & dilate(truth, tolerance)
    matched_truth = truth & dilate(prediction, tolerance)
    return dict(zip(BOUNDARY_COUNTS, (prediction.sum(), truth.sum(),
                                     matched_prediction.sum(), matched_truth.sum())))


def boundary_scores(counts):
    p = float(counts['boundary_predicted_pixels'])
    t = float(counts['boundary_true_pixels'])
    precision = float(counts['boundary_matched_prediction_pixels']) / p if p else (1. if not t else 0.)
    recall = float(counts['boundary_matched_truth_pixels']) / t if t else 1.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {'boundary_precision': precision, 'boundary_recall': recall, 'boundary_f1': f1}
