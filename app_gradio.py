import gradio as gr
import torch
import warnings
import subprocess
import tempfile
import os
from PIL import Image
import gc

# Ignora avvisi non critici
warnings.filterwarnings("ignore")

# --- CONFIGURAZIONE ---
# Variabili globali per mantenere il modello e il processore in memoria
MODEL_ID = "PULSE-ECG/PULSE-7B"
device = "cuda" if torch.cuda.is_available() else "cpu"
llava_installed = False

# Configurazioni per ottimizzazione memoria
QUANTIZATION_CONFIG = {
    "load_in_8bit": True,  # Carica in 8-bit per risparmiare memoria
    "load_in_4bit": False,  # Alternative: 4-bit per ancora più risparmio
    "low_cpu_mem_usage": True,
    "torch_dtype": torch.float16,  # Usa FP16 invece di FP32
}

def get_gpu_memory_info():
    """Ottiene informazioni sulla memoria GPU disponibile."""
    if torch.cuda.is_available():
        total_gpus = torch.cuda.device_count()
        print(f"GPU disponibili: {total_gpus}")
        for i in range(total_gpus):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            free_memory = total_memory - allocated_memory
            print(f"GPU {i}: {free_memory / 1024**3:.1f}GB liberi di {total_memory / 1024**3:.1f}GB totali")
        return total_gpus, total_memory
    return 0, 0

def clear_gpu_memory():
    """Pulisce la memoria GPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def setup_llava_environment():
    """
    Configura l'ambiente LLaVA necessario per PULSE-7B.
    """
    global llava_installed
    if not llava_installed:
        # Aggiungi solo la cartella locale LLaVA al PYTHONPATH se esiste
        llava_path = os.path.abspath("LLaVA")
        if os.path.exists(llava_path):
            import sys
            if llava_path not in sys.path:
                sys.path.insert(0, llava_path)
        llava_installed = True

def analyze_ecg(image, custom_prompt):
    """
    Prende in input un'immagine e un prompt personalizzato e restituisce l'analisi del modello PULSE-7B usando LLaVA.
    """
    if not llava_installed:
        return "Errore: l'ambiente LLaVA non è stato configurato. Controlla la console."
    
    if image is None:
        return "Per favore, carica un'immagine ECG."
    
    if not custom_prompt or custom_prompt.strip() == "":
        return "Per favore, inserisci un prompt per l'analisi."
        
    try:
        # Mostra info memoria prima del caricamento
        print("=== Informazioni memoria GPU prima del caricamento ===")
        get_gpu_memory_info()
        
        # Pulisce la memoria prima di iniziare
        clear_gpu_memory()
        
        # Salva l'immagine in un file temporaneo
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
            
            print("Caricamento PULSE-ECG/PULSE-7B con sistema LLaVA...")
            
            # Disabilita l'inizializzazione di torch per efficienza
            disable_torch_init()
            
            # Ottieni il nome del modello
            model_name = get_model_name_from_path(MODEL_ID)
            
            # Carica il modello con quantizzazione usando BitsAndBytesConfig per evitare deprecation warning
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
                load_8bit=False,  # Disabilitiamo qui perché usiamo quantization_config
                load_4bit=False,
                device_map="auto",  # Distribuzione automatica su GPU multiple
                device="cuda",
                **{"quantization_config": quantization_config}
            )
            
            print(f"Modello caricato: {model_name}")
            print(f"Lunghezza del contesto: {context_len}")
            print(f"Processore immagini: {type(image_processor)}")
            
            # Verifica che il processore di immagini sia valido
            if image_processor is None:
                print("ATTENZIONE: image_processor è None, provo a caricarne uno manualmente...")
                from transformers import CLIPImageProcessor
                image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
            
            # Configura la conversazione per PULSE-7B (usa llava_v1 come indicato nel repo)
            if "pulse" in model_name.lower():
                conv_mode = "llava_v1"
            else:
                conv_mode = "llava_v1"
            
            conv = conv_templates[conv_mode].copy()
            
            # Usa il prompt personalizzato dall'utente
            query = custom_prompt.strip()
            
            # Aggiungi i messaggi alla conversazione
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + query)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            print("=== Informazioni memoria GPU dopo il caricamento ===")
            get_gpu_memory_info()
            
            # Prepara gli input usando il sistema LLaVA con controllo errori
            image_sizes = [pil_image.size]
            try:
                images_tensor = process_images(
                    [pil_image],
                    image_processor,
                    model.config
                ).to(model.device, dtype=torch.float16)
            except AttributeError as attr_error:
                print(f"Errore nel processore di immagini: {attr_error}")
                # Prova un approccio alternativo
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
            
            print("Generazione della risposta...")
            # Genera la risposta con parametri ottimizzati per memoria
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    max_new_tokens=256,  # Riduciamo i token per risparmiare memoria
                    use_cache=True,
                )
            
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # Estrae solo la risposta (rimuove il prompt dalla generazione)
            if prompt in outputs:
                response = outputs.replace(prompt, "").strip()
            else:
                response = outputs.strip()
            
            # Pulisce il modello dalla memoria dopo l'uso
            del model
            clear_gpu_memory()
                
            return response if response else "Non è stato possibile generare un'analisi."
            
        except torch.cuda.OutOfMemoryError as oom_error:
            clear_gpu_memory()
            print(f"Out of Memory Error: {oom_error}")
            return "Errore: Memoria GPU insufficiente. Prova a riavviare l'applicazione o usa un'immagine più piccola."
            
        except Exception as model_error:
            clear_gpu_memory()
            print(f"Errore con il modello: {model_error}")
            return f"Si è verificato un errore durante l'analisi: {model_error}"
        
        finally:
            # Pulizia del file temporaneo
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
                
    except Exception as e:
        clear_gpu_memory()
        print(f"Errore durante l'analisi: {e}")
        return f"Si è verificato un errore durante l'analisi: {e}"

# --- Creazione dell'interfaccia Gradio ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # 🩺 PULSE-7B: Analisi ECG Multimodale
        Carica un'immagine di un elettrocardiogramma (ECG) per ottenere un'analisi testuale generata dal modello {MODEL_ID}.
        """
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Immagine ECG")
            
            # Campo di testo per il prompt personalizzabile
            prompt_input = gr.Textbox(
                label="Prompt di Analisi", 
                placeholder="Inserisci qui il tuo prompt personalizzato...",
                value="Please provide a detailed analysis of this ECG image. Focus on rhythm, rate, intervals, and any abnormalities you can identify.",
                lines=3,
                interactive=True
            )
            
            analyze_button = gr.Button("Analizza ECG", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(label="Risultato Analisi", lines=15, interactive=False)

    analyze_button.click(fn=analyze_ecg, inputs=[input_image, prompt_input], outputs=output_text)
    
    gr.Markdown(
        """
        ---
        *Disclaimer: Questo strumento è una demo tecnologica e non deve essere utilizzato per diagnosi mediche reali. 
        I risultati sono generati da un modello di intelligenza artificiale e potrebbero contenere errori.*
        """
    )

# Avvia l'app e configura l'ambiente al momento dell'avvio
if __name__ == "__main__":
    setup_llava_environment()
    demo.launch()

