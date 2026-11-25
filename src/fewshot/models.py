"""Normal-only patch memory and a compact reconstruction baseline."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


def handcrafted_grid(x):
    gx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0), mode="replicate")
    gy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1), mode="replicate")
    padded = F.pad(x, (2, 2, 2, 2), mode="reflect")
    mean = F.avg_pool2d(padded, 5, 1)
    std = (F.avg_pool2d(padded.square(), 5, 1) - mean.square()).clamp_min(0).sqrt()
    channels = torch.cat([x - mean, gx.abs(), gy.abs(), std], 1)
    patches = F.unfold(F.pad(channels, (2, 2, 2, 2), mode="reflect"), 5, stride=4)
    return patches.transpose(1, 2), ((x.shape[-2] + 3) // 4, (x.shape[-1] + 3) // 4)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.GELU())
        self.second = nn.Sequential(nn.Conv2d(16, 8, 3, 2, 1), nn.GELU())
        self.bottleneck = nn.Conv2d(8, 4, 3, 2, 1)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(4, 8, 4, 2, 1), nn.GELU(),
                                     nn.ConvTranspose2d(8, 16, 4, 2, 1), nn.GELU(),
                                     nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid())

    def encode_grid(self, x):
        return self.second(self.first(x))

    def forward(self, x):
        reconstructed = self.decoder(self.bottleneck(self.encode_grid(x)))
        return F.interpolate(reconstructed, size=x.shape[-2:], mode="bilinear", align_corners=False)

    @torch.inference_mode()
    def score(self, x):
        self.eval()
        return (self(x) - x).abs()


def greedy_coreset(features, budget):
    """Exact farthest-first traversal with O(ND + ND*K) compute, O(N) distances."""
    if budget < 1 or not len(features):
        raise ValueError("Need non-empty features and a positive budget")
    count = min(budget, len(features))
    first = int(features.square().sum(1).argmin())
    indices = [first]
    minimum = (features - features[first]).square().sum(1)
    minimum[first] = -1
    for _ in range(count - 1):
        index = int(minimum.argmax())
        indices.append(index)
        candidate = (features - features[index]).square().sum(1)
        minimum = torch.minimum(minimum, candidate)
        minimum[indices] = -1
    radius = float(minimum.clamp_min(0).max().sqrt())
    return torch.tensor(indices), radius


class PatchMemory:
    def __init__(self, budget=256, encoder=None, chunk_size=512):
        if budget < 1 or chunk_size < 1:
            raise ValueError("Require positive memory/query budgets")
        self.budget, self.encoder, self.chunk_size = budget, encoder, chunk_size

    @torch.inference_mode()
    def _features(self, x):
        if self.encoder is None:
            return handcrafted_grid(x)
        self.encoder.eval()
        features = self.encoder.encode_grid(x)
        return features.flatten(2).transpose(1, 2), features.shape[-2:]

    @torch.inference_mode()
    def fit(self, normal_images):
        patches, _ = self._features(normal_images)
        features = patches.reshape(-1, patches.shape[-1])
        self.mean = features.mean(0)
        self.std = features.std(0, unbiased=False).clamp_min(.01)
        standardized = (features - self.mean) / self.std
        self.indices, self.coverage_radius = greedy_coreset(standardized, self.budget)
        self.bank = standardized[self.indices].clone()
        self.fit_patch_count = len(features)
        return self

    @torch.inference_mode()
    def score(self, x):
        if not hasattr(self, "bank"):
            raise RuntimeError("Fit normal images before scoring")
        patches, (h, w) = self._features(x)
        z = (patches - self.mean) / self.std
        # Direct distances avoid cancellation near exact prototype matches and
        # keep results independent of cdist's automatic matrix-multiply switch.
        scores = [torch.cdist(chunk, self.bank, compute_mode="donot_use_mm_for_euclid_dist").amin(1)
                  for chunk in z.reshape(-1, z.shape[-1]).split(self.chunk_size)]
        coarse = torch.cat(scores).reshape(len(x), 1, h, w)
        if self.encoder is not None:
            return F.interpolate(coarse, size=x.shape[-2:], mode="bilinear", align_corners=False)
        # Handcrafted patch centers are at 0, 4, 8, ... in original pixels.
        # Explicit grid sampling preserves that correspondence near boundaries.
        yy = torch.arange(x.shape[-2], dtype=x.dtype, device=x.device) / max(1, 4 * (h - 1)) * 2 - 1
        xx = torch.arange(x.shape[-1], dtype=x.dtype, device=x.device) / max(1, 4 * (w - 1)) * 2 - 1
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        grid = torch.stack([gx, gy], -1)[None].expand(len(x), -1, -1, -1)
        return F.grid_sample(coarse, grid, padding_mode="border", align_corners=True)

    def storage_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.bank, self.mean, self.std))


def image_scores(maps, fraction=.01):
    if not 0 < fraction <= 1:
        raise ValueError("Fraction must lie in (0,1]")
    flat = maps.flatten(1)
    return flat.topk(max(1, math.ceil(flat.shape[1] * fraction)), 1).values.mean(1)


def conformal_threshold(normal_scores, alpha=.05):
    """Conservative split-conformal upper threshold for a strict 'score > t' rule.

    Exchangeable normal images are required for the image-level marginal bound.
    Insufficient calibration returns infinity instead of claiming false precision.
    """
    scores = torch.as_tensor(normal_scores).flatten()
    if not 0 < alpha < 1 or not len(scores) or not torch.isfinite(scores).all():
        raise ValueError("Invalid normal scores or false alarm target")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return float("inf") if rank > len(scores) else float(scores.sort().values[rank - 1])
