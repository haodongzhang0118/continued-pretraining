import torch
import stable_pretraining as spt

def patchify(imgs, patch_size):
    """Convert images to patches.

    Args:
        imgs: (B, 3, H, W)
        patch_size: int, size of each patch

    Returns:
        patches: (B, N, patch_size^2 * 3)
    """
    B, C, H, W = imgs.shape
    assert H == W and H % patch_size == 0, f"Image size {H} must be divisible by patch_size {patch_size}"
    
    h = w = H // patch_size
    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(B, h * w, patch_size ** 2 * C)
    return x


def mae_cp_forward(self, batch, stage):
    """MAE Continued Pretraining forward pass (Pure MAE with MaskedEncoder).
    
    Flow:
    1. MaskedEncoder applies random masking BEFORE encoding
    2. Encoder only processes visible patches (~25% of total)
    3. Decoder reconstructs all patches from visible token representations
    4. Loss computed only on masked patches
    
    This is the correct MAE implementation as described in the paper.
    """
    # 1. MaskedEncoder forward
    encoder_out = self.backbone(batch["image"])
    
    # Extract visible tokens (remove CLS token)
    tokens = encoder_out.encoded[:, 1:]  # [B, N_visible, D]
    mask = encoder_out.mask  # [B, N_patches]
    ids_keep = encoder_out.ids_keep  # [B, N_visible] - position indices in SHUFFLED order
    
    # CRITICAL FIX: Sort visible tokens by position
    # MaskedEncoder outputs tokens in RANDOM shuffle order (based on ids_keep)
    # But MAEDecoder expects tokens SORTED by position (for correct position embeddings)
    # Without this sort, position embeddings will be completely wrong!
    B, N_vis, D = tokens.shape
    N_patches = mask.shape[1]
    
    # Sort ids_keep to get the permutation
    ids_sorted_idx = torch.argsort(ids_keep, dim=1)  # [B, N_vis]
    # Reorder tokens to be sorted by position
    tokens_sorted = torch.gather(
        tokens, 
        dim=1, 
        index=ids_sorted_idx.unsqueeze(-1).expand(-1, -1, D)
    )  # [B, N_visible, D] - now sorted by position!
    
    # 2. Decoder reconstructs all patches from visible tokens
    pred = self.decoder(tokens_sorted, mask, output_masked_only=False)  # [B, N_patches, output_dim]
    
    # 3. Store embedding for evaluation callbacks (KNN, linear probe)
    # Use CLS token (global representation) instead of mean of visible tokens
    batch["embedding"] = encoder_out.encoded[:, 0]  # CLS token contains global info
    
    # 4. Compute reconstruction loss on masked patches only
    if self.training:
        # Patchify: convert image to patches
        target = patchify(batch["image"], self.patch_size)  # [B, N_patches, patch_size^2 * 3]
        
        # Loss: MSE only on mask=1 positions
        batch["loss"] = spt.losses.mae(
            target=target,
            pred=pred,
            mask=mask,
            norm_pix_loss=False
        )
        
        self.log(f"{stage}/loss", batch["loss"], on_step=True, on_epoch=True, sync_dist=True)
    
    return batch
