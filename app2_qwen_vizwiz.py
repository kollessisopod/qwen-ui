import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import gradio as gr

# 1. Configuration - Update the base model path if necessary
# Since you used VizWiz, I'm assuming Qwen2-VL-2B or 7B as the base
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct" 
ADAPTER_PATH = "./qwen_vizwiz_model"

# 1. Load the base model with specific settings
# Setting device_map="balanced" or "auto" is fine, 
# but we add offload_folder if VRAM is tight.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True
)

# 2. Explicitly load the adapter
# Instead of PeftModel.from_pretrained(model, ...), use the .load_adapter method
# This often handles device dispatching better for Qwen2.5-VL
try:
    model = PeftModel.from_pretrained(
        model, 
        ADAPTER_PATH,
        device_map="auto" # Ensure adapter follows base model placement
    )
    # Important: Merge the weights if you want to avoid overhead/warnings during inference
    model = model.merge_and_unload()
    print("Adapters merged successfully!")
except Exception as e:
    print(f"Error loading adapter: {e}")

processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

def process_inference(image, text_query):
    if image is None or text_query == "":
        return "Please provide both an image and a question."

    # Prepare inputs for Qwen2-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_query},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cpu")

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

# 2. Custom UI Theme (Gray, Dark Blue, Yellow)
custom_theme = gr.themes.Soft(
    primary_hue="yellow",
    secondary_hue="blue",
    neutral_hue="slate", # Provides the gray background tones
).set(
    body_background_fill="*neutral_50",
    block_background_fill="*neutral_100",
    button_primary_background_fill="*secondary_800", # Dark Blue
    button_primary_text_color="*primary_400",       # Yellow text
)

# 3. Build Interface
with gr.Blocks(theme=custom_theme) as demo:
    gr.Markdown("# Qwen VizWiz Analyzer")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload, Drag & Drop, or Paste (Ctrl+V) Image", type="pil")
            text_input = gr.Textbox(label="Question", placeholder="What is in this image?")
            run_btn = gr.Button("Run", variant="primary")
        
        with gr.Column():
            output_display = gr.Textbox(label="Model Response", interactive=False)

    run_btn.click(
        fn=process_inference, 
        inputs=[img_input, text_input], 
        outputs=output_display
    )

if __name__ == "__main__":
    demo.launch()