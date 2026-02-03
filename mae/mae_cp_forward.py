# MAE CP forward - reconstruct pixels from pretrained backbone latents
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
    """MAE Continued Pretraining forward pass.
    
    Correct MAE Pipeline:
    1. MaskedEncoder: Apply random masking → Encode ONLY visible patches
    2. Decoder: Reconstruct all patches from visible patch representations
    3. Loss: Compute reconstruction error only on masked patches
    
    Key: Masking happens BEFORE encoding (not after), which is the correct MAE approach.
    The encoder only sees and processes visible patches, making it learn better representations.
    """
    # 1. Masked Encoder forward
    encoder_out = self.backbone(batch["image"])
    
    encoded_tokens = encoder_out.encoded[:, 1:]
    mask = encoder_out.mask
    
    # 2. Decoder forward
    pred = self.decoder(encoded_tokens, mask, output_masked_only=False)  # [B, N_patches, output_dim]
    
    # 3. Store embedding for evaluation callbacks (KNN, linear probe)
    batch["embedding"] = encoded_tokens.mean(dim=1)
    
    # 4. Compute reconstruction loss on masked patches
    if self.training:
        # Patchify: convert image to patches
        target = patchify(batch["image"], self.patch_size)  # [B, N_patches, patch_size^2 * 3]
        
        # Loss: compute MSE only on mask=1 positions
        batch["loss"] = spt.losses.mae(
            target=target,
            pred=pred,
            mask=mask,
            norm_pix_loss=False
        )
        
        self.log(f"{stage}/loss", batch["loss"], on_step=True, on_epoch=True, sync_dist=True)
    
    return batch
