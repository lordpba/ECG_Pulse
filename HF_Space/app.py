import gradio as gr
import torch
import warnings
import tempfile
import os
from PIL import Image
import gc

# Ignore non-critical warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
MODEL_ID = "PULSE-ECG/PULSE-7B"
device = "cuda" if torch.cuda.is_available() else "cpu"

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

def analyze_ecg_simple(image, custom_prompt):
    """
    Simplified ECG analysis for HF Spaces.
    Uses a direct transformers approach without LLaVA dependencies.
    """
    if image is None:
        return "Please upload an ECG image."
    
    if not custom_prompt or custom_prompt.strip() == "":
        return "Please enter a prompt for analysis."
    
    try:
        print("=== Starting ECG Analysis ===")
        get_gpu_memory_info()
        clear_gpu_memory()
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Try different approaches for PULSE-7B loading
        try:
            print("Attempting to load PULSE-7B model...")
            
            # Method 1: Try with AutoProcessor and AutoModelForCausalLM
            from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
            
            # Configure quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            
            print("Loading processor...")
            processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
            
            print("Loading model with quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            print(f"Model loaded successfully on device: {model.device}")
            
            # Prepare inputs
            inputs = processor(
                text=custom_prompt,
                images=pil_image,
                return_tensors="pt"
            )
            
            # Move inputs to model device
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            print("Generating response...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode response
            response = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated text
            input_text = processor.decode(inputs['input_ids'][0], skip_special_tokens=True)
            if input_text in response:
                response = response.replace(input_text, "").strip()
            
            # Cleanup
            del model
            clear_gpu_memory()
            
            return response if response else "Analysis completed but no specific findings to report."
            
        except Exception as method1_error:
            print(f"Method 1 failed: {method1_error}")
            
            # Method 2: Try with different model loading approach
            try:
                print("Trying alternative loading method...")
                from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
                
                # Try LLaVA-Next as fallback
                processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    "llava-hf/llava-v1.6-mistral-7b-hf",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_8bit=True
                )
                
                # Format prompt for LLaVA-Next
                prompt = f"[INST] <image>\n{custom_prompt} [/INST]"
                
                inputs = processor(prompt, pil_image, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False
                    )
                
                response = processor.decode(outputs[0], skip_special_tokens=True)
                
                # Extract response
                if "[/INST]" in response:
                    response = response.split("[/INST]")[-1].strip()
                
                del model
                clear_gpu_memory()
                
                return f"Analysis (using fallback model): {response}"
                
            except Exception as method2_error:
                print(f"Method 2 also failed: {method2_error}")
                
                # Method 3: Provide informative error message
                return f"""
                **Model Loading Error**: Unable to load PULSE-7B model on this hardware configuration.
                
                **Error Details**: 
                - Primary error: {str(method1_error)[:200]}...
                - Fallback error: {str(method2_error)[:200]}...
                
                **Possible Solutions**:
                1. This Space may need more GPU memory (try T4 or A10G hardware)
                2. The model may require specific dependencies not available in this environment
                3. Consider running locally with proper LLaVA setup
                
                **For Production Use**: Please run this application locally with:
                - Proper LLaVA environment setup
                - Sufficient GPU memory (12GB+ recommended)
                - All required dependencies installed
                """
                
    except Exception as e:
        clear_gpu_memory()
        return f"Unexpected error during analysis: {str(e)}"

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="ECG PULSE Analyzer") as demo:
    gr.Markdown(
        """
        # 🩺 PULSE-7B: Multimodal ECG Analysis
        
        Upload an electrocardiogram (ECG) image to get a textual analysis generated by AI.
        
        **⚠️ Note**: This is a demo version. For best results, use the full local installation with proper GPU setup.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="numpy", 
                label="ECG Image", 
                height=400
            )
            
            prompt_input = gr.Textbox(
                label="Analysis Prompt",
                placeholder="Enter your analysis request...",
                value="Please provide a detailed analysis of this ECG image. Focus on rhythm, rate, intervals, and any abnormalities you can identify.",
                lines=4,
                interactive=True
            )
            
            analyze_button = gr.Button(
                "🔍 Analyze ECG", 
                variant="primary",
                size="lg"
            )
            
        with gr.Column():
            output_text = gr.Textbox(
                label="Analysis Result",
                lines=20,
                interactive=False,
                placeholder="Analysis results will appear here..."
            )
    
    # Button click event
    analyze_button.click(
        fn=analyze_ecg_simple,
        inputs=[input_image, prompt_input],
        outputs=output_text
    )
    
    # Examples section
    gr.Markdown(
        """
        ## 📝 Example Prompts:
        - "Analyze this ECG for any arrhythmias or conduction abnormalities"
        - "What is the heart rate and rhythm in this ECG?"
        - "Identify any ST-segment changes or abnormalities"
        - "Provide a comprehensive ECG interpretation including axis and intervals"
        """
    )
    
    gr.Markdown(
        """
        ---
        **⚠️ Medical Disclaimer**: This tool is for educational and research purposes only. 
        Results should never be used for actual medical diagnosis. Always consult qualified 
        medical professionals for medical decisions.
        
        **🔧 Technical Note**: This demo version may have limitations compared to the full 
        local installation. For production use, consider running locally with proper setup.
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )