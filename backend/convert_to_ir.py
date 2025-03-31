from openvino.tools.mo import convert_model
from openvino.runtime import serialize

# Convert and save image encoder
image_ir = convert_model(
    input_model="models/clip_image_encoder.onnx",
    compress_to_fp16=True
)
serialize(image_ir, "models/clip_image_encoder.xml")  # Saves .xml and .bin

# Convert and save text encoder
text_ir = convert_model(
    input_model="models/clip_text_encoder.onnx",
    compress_to_fp16=True
)
serialize(text_ir, "models/clip_text_encoder.xml")  # Saves .xml and .bin

print("Models converted to IR format successfully!")