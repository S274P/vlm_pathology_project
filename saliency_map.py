from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model + processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

# Pick one image
img_path = "data/pcam/sample_images/pcam_000000.png"
raw_image = Image.open(img_path).convert("RGB")

# Preprocess
inputs = processor(raw_image, return_tensors="pt").to(device)
image_tensor = inputs["pixel_values"].clone().detach().to(device)
image_tensor.requires_grad_(True)

# Generate a caption
generated_ids = model.generate(**inputs, max_length=30)
caption = processor.decode(generated_ids[0], skip_special_tokens=True)
print("Generated caption:", caption)

# Run encoder
encoder_outputs = model.get_encoder()(image_tensor)

# Decoder: predict the first token
decoder_input_ids = torch.tensor(
    [[model.config.decoder_start_token_id]], device=device
)
decoder_outputs = model.text_decoder(
    input_ids=decoder_input_ids,
    encoder_hidden_states=encoder_outputs.last_hidden_state,
)

# Pick target token
target_token_id = generated_ids[0, 0]
score = decoder_outputs.logits[0, 0, target_token_id]

# Backprop to get gradients
score.backward()

# Compute saliency map
saliency = image_tensor.grad.abs().squeeze().mean(dim=0)
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)

# Show saliency map
plt.imshow(saliency.cpu(), cmap="hot")
plt.axis("off")
plt.show()

