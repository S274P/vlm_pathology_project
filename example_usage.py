# example_usage.py 
from PIL import Image
import numpy as np
import torch
from eval_metrics import (
    perturb_image_noise, perturb_image_brightness, perturb_image_rotate,
    compute_ssim_continuity, compute_mae_continuity, compute_lipschitz_stability,
    faithfulness_masking
)
from saliency_map import gradcam_to_pli  
# assume model, processor, device are loaded as in your scripts

img_path = "data/pcam/sample_images/pcam_000000.png"
raw = Image.open(img_path).convert("RGB")
# original saliency (already computed in your pipeline) as saliency_np (HxW, [0,1])
# pli_map = gradcam_to_pli(saliency_np)  # already available

# Continuity: perturb images and recompute saliency 
perturbed = perturb_image_noise(np.array(raw).astype(np.float32) / 255.0, sigma=0.05)
pert_pil = Image.fromarray((perturbed * 255).astype(np.uint8))

# recompute saliency for perturbed image using your pipeline (example: call a helper that returns saliency_np_pert)
# For demonstration, if you have a function compute_saliency_for_pil(raw_pil) -> saliency_np:
# saliency_pert = compute_saliency_for_pil(pert_pil)
# For now example assumes you computed saliency_pert

# ssim_score = compute_ssim_continuity(saliency_np, saliency_pert)
# mae_score = compute_mae_continuity(saliency_np, saliency_pert)
# lips = compute_lipschitz_stability(saliency_np, saliency_pert, np.array(raw)/255., perturbed)

# Faithfulness (mask top 5% pixels)
# target_token_id should be chosen same as in your loop
# faith = faithfulness_masking(model, processor, raw, target_token_id, saliency_np, mask_fraction=0.05, mode="top", device=device)
# print(faith)
