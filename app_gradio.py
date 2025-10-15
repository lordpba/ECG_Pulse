import gradio as gr
import torch
import warnings
import subprocess
import tempfile
import os
from PIL import Image
import gc

# Ignore non-critical warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
# Global variables to keep model and processor in memory
MODEL_ID = "PULSE-ECG/PULSE-7B"
device = "cuda" if torch.cuda.is_available() else "cpu"
llava_installed = False

# Memory optimization configurations
QUANTIZATION_CONFIG = {
    "load_in_8bit": True,  # Load in 8-bit to save memory
    "load_in_4bit": False,  # Alternative: 4-bit for even more memory savings
    "low_cpu_mem_usage": True,
    "torch_dtype": torch.float16,  # Use FP16 instead of FP32
}

def get_gpu_memory_info():
    """Gets information about available GPU memory."""
    if torch.cuda.is_available():
        total_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {total_gpus}")
        for i in range(total_gpus):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            free_memory = total_memory - allocated_memory
            print(f"GPU {i}: {free_memory / 1024**3:.1f}GB free of {total_memory / 1024**3:.1f}GB total")
        return total_gpus, total_memory
    return 0, 0

def clear_gpu_memory():
    """Clears GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def setup_llava_environment():
    """
    Sets up the LLaVA environment required for PULSE-7B.
    """
    global llava_installed
    if not llava_installed:
        # Add only local LLaVA folder to PYTHONPATH if it exists
        llava_path = os.path.abspath("LLaVA")
        if os.path.exists(llava_path):
            import sys
            if llava_path not in sys.path:
                sys.path.insert(0, llava_path)
        llava_installed = True

def analyze_ecg(image, custom_prompt):
    """
    Takes an image and custom prompt as input and returns PULSE-7B model analysis using LLaVA.
    """
    if not llava_installed:
        print("[LLaVA] Environment not configured.")
        return "Error: LLaVA environment has not been configured. Check the console."

    if image is None:
        print("[INPUT] No image uploaded.")
        return "Please upload an ECG image."

    if not custom_prompt or custom_prompt.strip() == "":
        print("[INPUT] No prompt provided.")
        return "Please enter a prompt for analysis."

    try:
        print("[STEP] GPU Memory Info Before Loading Model:")
        get_gpu_memory_info()

        clear_gpu_memory()

        pil_image = Image.fromarray(image)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            pil_image.save(tmp_file.name)
            temp_image_path = tmp_file.name
        print(f"[INPUT] Image saved as: {temp_image_path}")

        try:
            print("[STEP] Importing LLaVA modules...")
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
            from llava.utils import disable_torch_init
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            from llava.conversation import conv_templates
            import torch

            print("[STEP] Loading PULSE-ECG/PULSE-7B with LLaVA system...")
            disable_torch_init()
            model_name = get_model_name_from_path(MODEL_ID)

            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )

            print("[STEP] Loading model...")
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                MODEL_ID,
                None,
                model_name,
                load_8bit=False,
                load_4bit=False,
                device_map="auto",
                device="cuda",
                **{"quantization_config": quantization_config}
            )
            print(f"[MODEL] Loaded: {model_name}")
            print(f"[MODEL] Context length: {context_len}")
            print(f"[MODEL] Image processor: {type(image_processor)}")

            if image_processor is None:
                print("[WARNING] image_processor is None, loading manually...")
                from transformers import CLIPImageProcessor
                image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

            conv_mode = "llava_v1"
            conv = conv_templates[conv_mode].copy()

            query = custom_prompt.strip()
            print(f"[INPUT] Prompt: {query}")

            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + query)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            print("[STEP] GPU Memory Info After Loading Model:")
            get_gpu_memory_info()

            image_sizes = [pil_image.size]
            try:
                print("[STEP] Processing image for model input...")
                images_tensor = process_images(
                    [pil_image],
                    image_processor,
                    model.config
                ).to(model.device, dtype=torch.float16)
            except AttributeError as attr_error:
                print(f"[ERROR] Image processor: {attr_error}")
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((336, 336)),
                    transforms.ToTensor(),
                ])
                images_tensor = transform(pil_image).unsqueeze(0).to(model.device, dtype=torch.float16)

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )

            print("[STEP] Generating response...")
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    max_new_tokens=256,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(f"[OUTPUT] Raw model output: {outputs}")

            if prompt in outputs:
                response = outputs.replace(prompt, "").strip()
            else:
                response = outputs.strip()

            print(f"[OUTPUT] Final response: {response}")

            del model
            clear_gpu_memory()

            print("[STEP] Analysis complete. Memory cleared.")
            return response if response else "Could not generate an analysis."

        except torch.cuda.OutOfMemoryError as oom_error:
            clear_gpu_memory()
            print(f"[ERROR] Out of Memory: {oom_error}")
            return "Error: Insufficient GPU memory. Try restarting the application or use a smaller image."

        except Exception as model_error:
            clear_gpu_memory()
            print(f"[ERROR] Model error: {model_error}")
            return f"An error occurred during analysis: {model_error}"

        finally:
            if os.path.exists(temp_image_path):
                print(f"[STEP] Removing temp image: {temp_image_path}")
                os.unlink(temp_image_path)

    except Exception as e:
        clear_gpu_memory()
        print(f"[ERROR] Analysis error: {e}")
        return f"An error occurred during analysis: {e}"

# --- Gradio Interface Creation ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # 🩺 PULSE-7B: Multimodal ECG Analysis
        Upload an electrocardiogram (ECG) image to get a textual analysis generated by the {MODEL_ID} model.
        """
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="ECG Image")
            
            # Text field for customizable prompt
            prompt_input = gr.Textbox(
                label="Analysis Prompt", 
                placeholder="Enter your custom prompt here...",
                value="Please provide a detailed analysis of this ECG image. Focus on rhythm, rate, intervals, and any abnormalities you can identify.",
                lines=3,
                interactive=True
            )
            
            analyze_button = gr.Button("Analyze ECG", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(label="Analysis Result", lines=15, interactive=False)

    analyze_button.click(fn=analyze_ecg, inputs=[input_image, prompt_input], outputs=output_text)
    
    gr.Markdown(
        """
        ---
        *Disclaimer: This tool is a technological demo and should not be used for real medical diagnoses. 
        Results are generated by an artificial intelligence model and may contain errors.*
        """
    )

# Launch app and configure environment at startup
if __name__ == "__main__":
    setup_llava_environment()
    demo.launch(
        share=True,           # 🌐 Crea link pubblico gradio.live
        server_name="0.0.0.0",  # Accetta connessioni esterne
        server_port=7860,       # Porta standard Gradio
        debug=True              # Mostra info debug
    )

