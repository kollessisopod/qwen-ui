import gradio as gr
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering

# 1. Model Loading
model_path = "./blipModel" 
device = torch.device("cpu")

print(f"Loading local model to CPU...")
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForQuestionAnswering.from_pretrained(model_path).to(device)

def vqa_inference(input_image, question):
    if input_image is None or not question:
        return "Error: Please provide both an image and a question."
    try:
        inputs = processor(images=input_image, text=question, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs)
        return processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Inference Error: {str(e)}"

# 2. JavaScript for Shift + Enter Shortcut
# This looks for the 'Enter' key while 'Shift' is held down to click the button
shortcut_js = """
function() {
    document.addEventListener('keydown', (e) => {
        if (e.shiftKey && e.key === 'Enter') {
            e.preventDefault();
            document.getElementById('submit_btn').click();
        }
    });
}
"""

# 3. Custom CSS (Theme: Dark Blue & Yellow)
custom_css = """
.gradio-container { background-color: #001f3f !important; }
.gradio-container *, label, span, p, h1 { color: #FFD700 !important; }
input, textarea { 
    background-color: #002d5a !important; 
    color: #FFD700 !important; 
    border: 1px solid #FFD700 !important; 
}
button#submit_btn { 
    background-color: #FFD700 !important; 
    color: #001f3f !important; 
    font-weight: bold !important; 
    border: none !important;
}
"""

# 4. Building the Interface
with gr.Blocks(css=custom_css, title="Local BLIP VQA", js=shortcut_js) as demo:
    gr.Markdown("# 👁️ Local BLIP VQA")
    
    with gr.Row():
        # LEFT COLUMN: Image only
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Image Input", 
                type="pil", 
                sources=["upload", "clipboard"]
            )
            
        # RIGHT COLUMN: Question on top, Answer below
        with gr.Column(scale=1):
            question_input = gr.Textbox(
                label="Your Question", 
                placeholder="Type question here... (Shift + Enter to run)",
                lines=3
            )
            
            submit_btn = gr.Button("Analyze Image", variant="primary", elem_id="submit_btn")
            
            answer_output = gr.Textbox(
                label="Model Response", 
                placeholder="The answer will appear here...",
                interactive=False
            )

    # Link functionality
    submit_btn.click(
        fn=vqa_inference, 
        inputs=[image_input, question_input], 
        outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch()