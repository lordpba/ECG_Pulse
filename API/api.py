from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tempfile
import os
import shutil
from typing import Dict, Optional
import traceback
import torch
import warnings
import gc
import io

# Ignore non-critical warnings
warnings.filterwarnings("ignore")

app = FastAPI(
    title="ECG PULSE API",
    description="API for ECG image analysis using PULSE-7B model",
    version="1.0.0"
)

# CORS - allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
MODEL_ID = "PULSE-ECG/PULSE-7B"

# Force execution on GPU 1 only
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
llava_installed = False

# Model cache - keep model loaded in memory
model_cache = {
    "tokenizer": None,
    "model": None,
    "image_processor": None,
    "context_len": None
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
        # Check both current directory and parent directory (in case running from API/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        llava_paths = [
            os.path.join(parent_dir, "LLaVA"),  # ../LLaVA when in API/
            os.path.join(current_dir, "LLaVA"),  # ./LLaVA when in root
            "LLaVA"  # Fallback
        ]
        
        import sys
        for llava_path in llava_paths:
            if os.path.exists(llava_path):
                if llava_path not in sys.path:
                    sys.path.insert(0, llava_path)
                print(f"[LLAVA] Added to path: {llava_path}")
                break
        
        llava_installed = True

def load_model():
    """Load the PULSE-7B model into memory cache."""
    global model_cache
    
    if model_cache["model"] is not None:
        print("[MODEL] Model already loaded in cache")
        return
    
    try:
        print("[STEP] Importing LLaVA modules...")
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from llava.utils import disable_torch_init
        from transformers import BitsAndBytesConfig
        
        print("[STEP] Loading PULSE-ECG/PULSE-7B with LLaVA system...")
        disable_torch_init()
        model_name = get_model_name_from_path(MODEL_ID)
        
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
        
        # Cache the model
        model_cache["tokenizer"] = tokenizer
        model_cache["model"] = model
        model_cache["image_processor"] = image_processor
        model_cache["context_len"] = context_len
        
        print(f"[MODEL] Loaded: {model_name}")
        print(f"[MODEL] Context length: {context_len}")
        print(f"[MODEL] Image processor: {type(image_processor)}")
        
        # Verify image processor
        if image_processor is None:
            print("[WARNING] image_processor is None, loading manually...")
            from transformers import CLIPImageProcessor
            model_cache["image_processor"] = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        
        print("[MODEL] Model successfully cached and ready")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        traceback.print_exc()
        raise

@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts"""
    print("Initializing ECG PULSE API...")
    setup_llava_environment()
    print("[STEP] GPU Memory Info Before Loading Model:")
    get_gpu_memory_info()
    
    try:
        load_model()
        print("✓ ECG PULSE API ready!")
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")

# API Key Authentication
async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    Verify the API key from the request header.
    If API_KEY environment variable is set, authentication is required.
    If not set, the API remains open (backward compatible).
    """
    required_api_key = os.environ.get("API_KEY")

    # If no API key is configured, allow access (backward compatible)
    if not required_api_key:
        return True

    # If API key is configured but not provided in request
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Please provide X-API-Key header."
        )

    # Verify the API key
    if x_api_key != required_api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    return True

@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "ECG PULSE API",
        "version": "1.0.0",
        "model": MODEL_ID,
        "endpoints": {
            "/analyze": "POST - Analyze an ECG image",
            "/health": "GET - Check service status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check to verify the service is active"""
    if model_cache["model"] is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Model not loaded"}
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_id": MODEL_ID,
        "device": device,
        "llava_configured": llava_installed
    }

@app.post("/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_ecg(
    image: UploadFile = File(...),
    prompt: str = Form(default="Please provide a detailed analysis of this ECG image. Focus on rhythm, rate, intervals, and any abnormalities you can identify.")
) -> Dict:
    """
    Analyze an ECG image and return medical insights.

    Args:
        image: ECG image file (supported formats: .png, .jpg, .jpeg, .bmp, .tiff)
        prompt: Custom analysis prompt (optional)

    Returns:
        Dict with the analysis results
    """
    if model_cache["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Verify file extension
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIFF']
    file_ext = os.path.splitext(image.filename)[1]  # type: ignore
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File format not supported. Use one of the following: {', '.join(allowed_extensions)}"
        )
    
    temp_image_path = None
    
    try:
        # Read image from upload
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Save to temporary file for logging
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            pil_image.save(tmp_file.name)
            temp_image_path = tmp_file.name

        print(f"[INPUT] Image received: {image.filename}")
        print(f"[INPUT] Image size: {pil_image.size}")
        print(f"[INPUT] Prompt: {prompt}")

        # Import LLaVA utilities
        from llava.mm_utils import tokenizer_image_token, process_images
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        
        # Configure conversation
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        
        query = prompt.strip()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + query)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        # Prepare inputs
        image_sizes = [pil_image.size]
        try:
            print("[STEP] Processing image for model input...")
            images_tensor = process_images(
                [pil_image],
                model_cache["image_processor"],
                model_cache["model"].config
            ).to(model_cache["model"].device, dtype=torch.float16)
        except AttributeError as attr_error:
            print(f"[ERROR] Image processor: {attr_error}")
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((336, 336)),
                transforms.ToTensor(),
            ])
            images_tensor = transform(pil_image).unsqueeze(0).to(model_cache["model"].device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt_text, model_cache["tokenizer"], IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        
        print("[STEP] Generating response...")
        # Generate response
        with torch.inference_mode():
            output_ids = model_cache["model"].generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                max_new_tokens=512,
                use_cache=True,
            )
        
        outputs = model_cache["tokenizer"].batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(f"[OUTPUT] Raw model output: {outputs}")
        
        # Extract response
        if prompt_text in outputs:
            response = outputs.replace(prompt_text, "").strip()
        else:
            response = outputs.strip()
        
        print(f"[OUTPUT] Final response: {response}")
        print(f"[SUCCESS] Analysis completed for {image.filename}")

        return {
            "success": True,
            "filename": image.filename,
            "prompt": prompt,
            "analysis": response,
            "model": MODEL_ID
        }

    except Exception as e:
        print(f"[ERROR] Error during analysis: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

    finally:
        # Cleanup temporary file
        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
            print(f"[CLEANUP] Removed temp image: {temp_image_path}")

@app.post("/analyze-batch", dependencies=[Depends(verify_api_key)])
async def analyze_batch(
    images: list[UploadFile] = File(...),
    prompt: str = Form(default="Please provide a detailed analysis of this ECG image. Focus on rhythm, rate, intervals, and any abnormalities you can identify.")
) -> Dict:
    """
    Analyze multiple ECG images simultaneously.

    Args:
        images: List of ECG image files
        prompt: Custom analysis prompt (optional, same for all images)

    Returns:
        Dict with results for each image
    """
    if len(images) > 5:
        raise HTTPException(
            status_code=400,
            detail="Maximum 5 images per request"
        )
    
    results = []
    for idx, img in enumerate(images):
        print(f"\n[BATCH] Processing image {idx+1}/{len(images)}: {img.filename}")
        try:
            result = await analyze_ecg(img, prompt)
            results.append(result)
        except HTTPException as e:
            results.append({
                "success": False,
                "filename": img.filename,
                "error": e.detail
            })
    
    return {
        "total": len(images),
        "prompt": prompt,
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting ECG PULSE API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
