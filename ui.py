import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import gradio as gr
import os

# 1. Setup Paths
ADAPTER_PATH = r"C:\Users\108616\Desktop\qwen_vizwiz_model-20260102T075323Z-1-001\qwen_vizwiz_model"
BASE_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

# Create a folder for offloading if it doesn't exist (helps with the KeyError)
offload_dir = "offload_temp"
if not os.path.exists(offload_dir):
    os.makedirs(offload_dir)

print(f"Loading Qwen2-VL-7B + Adapters...")

# 2. Optimized Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True,
)

# 3. Load Base Model with specific flags to prevent deep nesting
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    offload_folder=offload_dir,
)

# 4. Attach Adapters with 'is_trainable=False'
# This specifically helps resolve the 'meta' parameter and 'KeyError' issues
model = PeftModel.from_pretrained(
    base_model, 
    ADAPTER_PATH, 
    is_trainable=False
)
model.eval() # Set to evaluation mode

# 5. Load Processor
processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

def predict(img, question):
    if img is None or not question:
        return "Please upload an image and type a question."

    # Standard Qwen2-VL Message format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    # Preprocess using qwen_vl_utils
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Decode answer
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return answer

# --- Gradio UI (Gray, Dark Blue, Yellow Theme) ---
theme = gr.themes.Soft(
    primary_hue="yellow",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_100", # Gray
    button_primary_background_fill="*secondary_900", # Dark Blue
    button_primary_text_color="*primary_400",       # Yellow
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# 👁️ VizWiz Qwen2-VL (7B-LoRA)")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Drop image or Ctrl+V here", type="pil")
            question_input = gr.Textbox(label="Question", placeholder="Ask about the image...")
            run_btn = gr.Button("Run Inference", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="Model Response", interactive=False, lines=8)

    run_btn.click(fn=predict, inputs=[image_input, question_input], outputs=output_text)

if __name__ == "__main__":
    demo.launch()