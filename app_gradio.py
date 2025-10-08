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
    """Get information about available GPU memory."""
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
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def setup_llava_environment():
    """
    Configure the LLaVA environment required for PULSE-7B.
    """
    global llava_installed
    if not llava_installed:
        print("Setting up LLaVA environment...")
        try:
            # Clone LLaVA repository if it doesn't exist
            if not os.path.exists("LLaVA"):
                print("Cloning LLaVA repository...")
                result = subprocess.run([
                    "git", "clone", "https://github.com/haotian-liu/LLaVA.git"
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"Clone error: {result.stderr}")
            
            # Add LLaVA to PYTHONPATH
            import sys
            llava_path = os.path.abspath("LLaVA")
            if llava_path not in sys.path:
                sys.path.insert(0, llava_path)
                print(f"Added {llava_path} to PYTHONPATH")
            
            llava_installed = True
            print("LLaVA environment configured successfully.")
            
        except Exception as e:
            print(f"Configuration error: {e}")
            raise gr.Error(f"Unable to configure LLaVA environment. Error: {e}")

def analyze_ecg(image, custom_prompt):
    """
    Takes an image and custom prompt as input and returns PULSE-7B model analysis using LLaVA.
    """
    if not llava_installed:
        return "Error: LLaVA environment has not been configured. Check the console."
    
    if image is None:
        return "Please upload an ECG image."
    
    if not custom_prompt or custom_prompt.strip() == "":
        return "Please enter a prompt for analysis."
        
        try:
            # Show memory info before loading
            print("=== GPU memory information before loading ===")
            get_gpu_memory_info()
            
            # Clear memory before starting
            clear_gpu_memory()
            
            # Save image to temporary file
            pil_image = Image.fromarray(image)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                pil_image.save(tmp_file.name)
                temp_image_path = tmp_file.name
        
        try:
            # Usa il sistema di caricamento LLaVA specifico per PULSE-7B
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
            from llava.utils import disable_torch_init
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            from llava.conversation import conv_templates
            import torch
            
            print("Loading PULSE-ECG/PULSE-7B with LLaVA system...")
            
            # Disable torch initialization for efficiency
            disable_torch_init()
            
            # Get model name
            model_name = get_model_name_from_path(MODEL_ID)
            
            # Load model with quantization using BitsAndBytesConfig to avoid deprecation warning
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                MODEL_ID, 
                None,  # model_base
                model_name, 
                load_8bit=False,  # Disable here because we use quantization_config
                load_4bit=False,
                device_map="auto",  # Automatic distribution across multiple GPUs
                device="cuda",
                **{"quantization_config": quantization_config}
            )
            
            print(f"Model loaded: {model_name}")
            print(f"Context length: {context_len}")
            print(f"Image processor: {type(image_processor)}")
            
            # Verify that image processor is valid
            if image_processor is None:
                print("WARNING: image_processor is None, trying to load one manually...")
                from transformers import CLIPImageProcessor
                image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
            
            # Configure conversation for PULSE-7B (uses llava_v1 as indicated in repo)
            if "pulse" in model_name.lower():
                conv_mode = "llava_v1"
            else:
                conv_mode = "llava_v1"
            
            conv = conv_templates[conv_mode].copy()
            
            # Use custom prompt from user
            query = custom_prompt.strip()
            
            # Add messages to conversation
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + query)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            print("=== GPU memory information after loading ===")
            get_gpu_memory_info()
            
            # Prepare inputs using LLaVA system with error checking
            image_sizes = [pil_image.size]
            try:
                images_tensor = process_images(
                    [pil_image],
                    image_processor,
                    model.config
                ).to(model.device, dtype=torch.float16)
            except AttributeError as attr_error:
                print(f"Image processor error: {attr_error}")
                # Try alternative approach
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
            
            print("Generating response...")
            # Generate response with memory-optimized parameters
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    max_new_tokens=256,  # Reduce tokens to save memory
                    use_cache=True,
                )
            
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # Extract only the response (remove prompt from generation)
            if prompt in outputs:
                response = outputs.replace(prompt, "").strip()
            else:
                response = outputs.strip()
            
            # Clean model from memory after use
            del model
            clear_gpu_memory()
                
            return response if response else "Unable to generate analysis."
            
        except torch.cuda.OutOfMemoryError as oom_error:
            clear_gpu_memory()
            print(f"Out of Memory Error: {oom_error}")
            return "Error: Insufficient GPU memory. Try restarting the application or use a smaller image."
            
        except Exception as model_error:
            clear_gpu_memory()
            print(f"Model error: {model_error}")
            return f"An error occurred during analysis: {model_error}"
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
                
    except Exception as e:
        clear_gpu_memory()
        print(f"Analysis error: {e}")
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

# Launch the app and configure environment on startup
if __name__ == "__main__":
    setup_llava_environment()
    demo.launch()

