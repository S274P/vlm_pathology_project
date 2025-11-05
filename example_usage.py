"""
example_usage.py
----------------
Demonstrates how to use eval_metrics.py functions for Pixel-level Interpretability (PLI)
evaluation on a BLIP model. 

Evaluates:
- Continuity (SSIM, MAE)
- Lipschitz Stability
- Faithfulness via masking
"""

import numpy as np
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from eval_metrics import (
    perturb_image_noise,
    perturb_image_brightness,
    perturb_image_rotate,
    compute_ssim_continuity,
    compute_mae_continuity,
    compute_lipschitz_stability,
    faithfulness_masking
)

# -------------------------
#   Load model + processor
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Salesforce/blip-image-captioning-base"
print(f"Loading BLIP model ({model_name})...")
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval()

# -------------------------
#   Load sample image
# -------------------------
img_path = "data/pcam/sample_images/pcam_000000.png"  
raw = Image.open(img_path).convert("RGB")
raw_np = np.array(raw).astype(np.float32) / 255.0

# -------------------------
#   Compute saliency (GradCAM or PLI)
# -------------------------
# Placeholder saliency generator 
def compute_saliency_for_pil(pil_img):
    """Returns a mock saliency map (replace with gradcam_to_pli or your actual saliency)."""
    np_img = np.array(pil_img).astype(np.float32) / 255.0
    # Creates a fake heatmap: center brighter than edges
    h, w, _ = np_img.shape
    y, x = np.ogrid[:h, :w]
    cx, cy = h / 2, w / 2
    sal = np.exp(-((x - cy) ** 2 + (y - cx) ** 2) / (2 * (0.25 * h) ** 2))
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal

saliency_np = compute_saliency_for_pil(raw)

# -------------------------
#   Apply perturbations
# -------------------------
perturbed = perturb_image_noise(raw_np, sigma=0.05)
pert_pil = Image.fromarray((perturbed * 255).astype(np.uint8))
saliency_pert = compute_saliency_for_pil(pert_pil)

# -------------------------
#   Compute Continuity metrics
# -------------------------
ssim_score = compute_ssim_continuity(saliency_np, saliency_pert)
mae_score = compute_mae_continuity(saliency_np, saliency_pert)
lip_score = compute_lipschitz_stability(saliency_np, saliency_pert, raw_np, perturbed)

print("\n=== Continuity Metrics ===")
print(f"SSIM: {ssim_score:.4f}")
print(f"MAE:  {mae_score:.4f}")
print(f"Lipschitz Stability: {lip_score:.4f}")

# -------------------------
#   Compute Faithfulness (using real BLIP output)
# -------------------------
# Generate caption to extract a target token
inputs = processor(raw, return_tensors="pt").to(device)
out = model.generate(**inputs, max_new_tokens=20)
caption = processor.decode(out[0], skip_special_tokens=True)
print(f"\nGenerated Caption: {caption}")

# Choose a target token for testing (e.g. 2nd word)
tokenized = processor.tokenizer(caption, return_tensors="pt")
target_token_id = int(tokenized["input_ids"][0][1]) if tokenized["input_ids"].size(1) > 1 else int(tokenized["input_ids"][0][0])

faith = faithfulness_masking(
    model=model,
    processor=processor,
    raw_image=raw,
    target_token_id=target_token_id,
    saliency_map=saliency_np,
    mask_fraction=0.05,
    mode="top",
    device=device
)

print("\n=== Faithfulness Metrics ===")
for k, v in faith.items():
    print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

print("\n✅ Example evaluation complete.")
