import torch
import clip

# Load your fine-tuned model
device = torch.device("cpu")  # Export on CPU for simplicity
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.load_state_dict(torch.load("models/fine_tuned_clip.pt", map_location=device))
model.eval()

# Dummy inputs
dummy_image = torch.randn(1, 3, 224, 224)  # Single image: [1, 3, 224, 224]
dummy_text = clip.tokenize(["dummy text"] * 22)  # 22 labels like MEDICAL_LABELS: [22, 77]

# Export image encoder (Vision Transformer)
torch.onnx.export(
    model.visual,
    dummy_image,
    "models/clip_image_encoder.onnx",
    input_names=["image"],
    output_names=["image_features"],
    dynamic_axes={"image": {0: "batch_size"}, "image_features": {0: "batch_size"}},
    opset_version=14,
    do_constant_folding=True
)

# Define a wrapper for text encoder to include embeddings
class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, 77] -> [batch_size, 77, 512]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # [77, batch_size, 512]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [batch_size, 77, 512]
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

# Export text encoder with embeddings
text_encoder = TextEncoderWrapper(model)
torch.onnx.export(
    text_encoder,
    dummy_text,
    "models/clip_text_encoder.onnx",
    input_names=["text"],
    output_names=["text_features"],
    dynamic_axes={"text": {0: "batch_size"}, "text_features": {0: "batch_size"}},
    opset_version=14,
    do_constant_folding=True
)

print("Image and text encoders exported to ONNX successfully!")