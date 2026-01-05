# app_qwen_vizwiz.py
# Fix for PEFT KeyError with device_map="auto": load on a single device (no offload) before loading adapter.

import argparse
import traceback
from pathlib import Path
from typing import Optional, Tuple

import torch
import gradio as gr
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info


CSS = r"""
body, .gradio-container { background: #3b3f46 !important; color: #eef2f7 !important; }
.gr-block, .gr-box, .gr-panel, .gr-form, .wrap, .block {
  border-radius: 14px !important;
  border: 1px solid rgba(245, 197, 66, 0.22) !important;
  background: rgba(11, 31, 58, 0.35) !important;
}
button, .gr-button {
  background: #0b1f3a !important;
  color: #f5c542 !important;
  border: 1px solid rgba(245, 197, 66, 0.55) !important;
  border-radius: 12px !important;
}
button:hover, .gr-button:hover { filter: brightness(1.08); }
textarea, input, .gr-text-input, .gr-text-area {
  background: rgba(0,0,0,0.25) !important;
  color: #eef2f7 !important;
  border: 1px solid rgba(245, 197, 66, 0.25) !important;
  border-radius: 12px !important;
}
h1, h2, h3, .markdown h1, .markdown h2, .markdown h3 { color: #f5c542 !important; }
.gr-image, .image-frame {
  border-radius: 14px !important;
  overflow: hidden !important;
  border: 1px solid rgba(245, 197, 66, 0.22) !important;
}
"""


def pick_torch_dtype(dtype_str: str, device: str) -> torch.dtype:
    if (dtype_str or "").lower().strip() == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    m = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return m.get((dtype_str or "auto").lower().strip(), torch.float16 if device == "cuda" else torch.float32)


def load_model_and_processor(
    base_model: str,
    adapter_path: Path,
    device: str,
    dtype: torch.dtype,
    load_4bit: bool,
    merge_lora: bool,
) -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    # Force single-device placement to avoid PEFT offload-index rewrite errors.
    device_map = {"": device}

    quant_cfg = None
    if load_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=None if load_4bit else dtype,
        device_map=device_map,
        quantization_config=quant_cfg,
        low_cpu_mem_usage=False,
    )

    # Load LoRA adapter from ./qwen_vizwiz_model
    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        is_trainable=False,
        # Keep adapter load simple; avoid PEFT dispatch/offload logic.
        device_map=None,
        low_cpu_mem_usage=False,
    )

    if merge_lora:
        model = model.merge_and_unload()

    model.eval()
    processor = AutoProcessor.from_pretrained(str(adapter_path))
    return model, processor


class QwenVizWizApp:
    def __init__(
        self,
        base_model: str,
        adapter_path: Path,
        device: str,
        dtype_str: str,
        load_4bit: bool,
        merge_lora: bool,
        max_new_tokens: int,
    ):
        self.device = device
        self.dtype = pick_torch_dtype(dtype_str, device)
        self.max_new_tokens = int(max_new_tokens)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        self.model, self.processor = load_model_and_processor(
            base_model=base_model,
            adapter_path=adapter_path,
            device=self.device,
            dtype=self.dtype,
            load_4bit=load_4bit,
            merge_lora=merge_lora,
        )

    @torch.inference_mode()
    def answer(self, image: Optional[Image.Image], question: str) -> str:
        if image is None:
            return "Error: No image provided."
        question = (question or "").strip()
        if not question:
            return "Error: Question is empty."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Inputs go to the same device the model is on
        if self.device == "cuda":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to("cpu")

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return answer.strip()


def build_ui(app: QwenVizWizApp) -> gr.Blocks:
    with gr.Blocks(title="Qwen VizWiz VQA") as demo:
        gr.Markdown("# Qwen2-VL VizWiz VQA")

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(
                    label="Image (upload / drag-drop / clipboard)",
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                gr.Markdown("Paste with Ctrl+V into the image field, or drag-drop, or upload from file.")

            with gr.Column(scale=1):
                question = gr.Textbox(label="Question", placeholder="Type your question...", lines=2)
                run_btn = gr.Button("Run Inference")
                answer = gr.Textbox(label="Answer", lines=6, interactive=False)
                err = gr.Markdown("", visible=False)

        def _run(image_in, question_in):
            try:
                out = app.answer(image_in, question_in)
                return out, gr.update(visible=False, value="")
            except Exception:
                tb = traceback.format_exc()
                return "", gr.update(visible=True, value=f"**Runtime error**\n\n```\n{tb}\n```")

        run_btn.click(fn=_run, inputs=[image, question], outputs=[answer, err])

    return demo


def main():
    here = Path(__file__).resolve().parent
    default_adapter = (here / "qwen_vizwiz_model").resolve()

    p = argparse.ArgumentParser(description="Gradio app for Qwen2-VL + VizWiz LoRA adapter")
    p.add_argument("--adapter_path", type=str, default=str(default_adapter), help="Default: ./qwen_vizwiz_model")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="auto", help="auto | bf16 | fp16 | fp32")
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--merge_lora", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    adapter_path = Path(args.adapter_path).expanduser().resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    app = QwenVizWizApp(
        base_model=args.base_model,
        adapter_path=adapter_path,
        device=args.device,
        dtype_str=args.dtype,
        load_4bit=args.load_4bit,
        merge_lora=args.merge_lora,
        max_new_tokens=args.max_new_tokens,
    )

    ui = build_ui(app)
    ui.launch(server_name=args.host, server_port=args.port, share=bool(args.share), css=CSS)


if __name__ == "__main__":
    main()
