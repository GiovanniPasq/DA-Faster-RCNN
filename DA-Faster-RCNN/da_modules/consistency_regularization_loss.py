import torch
import torch.nn.functional as F
from typing import List

def consistency_regularization_loss(
    img_logits: torch.Tensor,               # shape: [B, 1, H, W]
    ins_logits_list: List[torch.Tensor]     # list of B tensors, each of shape [Nᵢ, 1]
) -> torch.Tensor:
    """Compute the consistency regularization loss between image-level and instance-level
    domain classifier outputs.
    Args:
        img_logits (Tensor): Logits from the image-level domain classifier,
                             shape [B, 1, H, W].
        ins_logits_list (List[Tensor]): List of Tensors from the instance-level
                                        domain classifier, one per image in the batch.
                                        Each tensor has shape [Nᵢ, 1], where Nᵢ is the
                                        number of ROI proposals for image i.

    Returns:
        Tensor: Scalar tensor representing the mean consistency loss over the batch.
    """
    B = img_logits.size(0)
    total_loss = 0.0
    valid_samples = 0

    for i in range(B):
        # Compute average image-level domain probability (after sigmoid)
        img_prob = torch.sigmoid(img_logits[i, 0]).mean()  # scalar ∈ [0,1]

        if i >= len(ins_logits_list):
            continue

        ins_logits = ins_logits_list[i]  # shape [Nᵢ, 1]
        if ins_logits.numel() == 0:
            continue

        ins_probs = torch.sigmoid(ins_logits.view(-1))  # shape [Nᵢ]

        # Compute L2 distance between image-level and each instance-level probability
        loss = F.mse_loss(ins_probs, img_prob.expand_as(ins_probs))  # sum_j (p̄_i - p_{i,j})²
        total_loss += loss
        valid_samples += 1

    if valid_samples == 0:
        return torch.tensor(0.0, requires_grad=True, device=img_logits.device)

    return total_loss / valid_samples