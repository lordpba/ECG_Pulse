# 🔌 ECG PULSE API

FastAPI REST API per analizzare immagini ECG usando il modello PULSE-7B.

## 🚀 Avvio Rapido

### 1. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 2. Avvia il server API

```bash
python api_gradio.py
```

Il server sarà disponibile su `http://localhost:8000`

### 3. Accedi alla documentazione interattiva

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 Endpoints

### GET `/`
Informazioni sull'API

```bash
curl http://localhost:8000/
```

### GET `/health`
Verifica stato del servizio e del modello

```bash
curl http://localhost:8000/health
```

### POST `/analyze`
Analizza una singola immagine ECG

**Parametri:**
- `image` (file): Immagine ECG (.png, .jpg, .jpeg, .bmp, .tiff)
- `prompt` (text, opzionale): Prompt personalizzato per l'analisi

**Esempio con cURL:**

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "image=@ecg_sample.png" \
  -F "prompt=Analyze this ECG and identify any arrhythmias."
```

**Esempio con Python:**

```python
import requests

with open('ecg_sample.png', 'rb') as f:
    files = {'image': f}
    data = {'prompt': 'Provide detailed ECG analysis'}
    response = requests.post('http://localhost:8000/analyze', files=files, data=data)
    print(response.json())
```

**Risposta:**

```json
{
  "success": true,
  "filename": "ecg_sample.png",
  "prompt": "Analyze this ECG...",
  "analysis": "The ECG shows normal sinus rhythm with a rate of approximately 75 bpm...",
  "model": "PULSE-ECG/PULSE-7B"
}
```

### POST `/analyze-batch`
Analizza multiple immagini ECG contemporaneamente (max 5)

**Parametri:**
- `images` (files): Lista di immagini ECG
- `prompt` (text, opzionale): Prompt comune per tutte le immagini

**Esempio con cURL:**

```bash
curl -X POST "http://localhost:8000/analyze-batch" \
  -F "images=@ecg1.png" \
  -F "images=@ecg2.png" \
  -F "images=@ecg3.png" \
  -F "prompt=Brief ECG analysis"
```

**Esempio con Python:**

```python
import requests

files = [
    ('images', ('ecg1.png', open('ecg1.png', 'rb'), 'image/png')),
    ('images', ('ecg2.png', open('ecg2.png', 'rb'), 'image/png')),
]
data = {'prompt': 'Analyze these ECGs'}
response = requests.post('http://localhost:8000/analyze-batch', files=files, data=data)
print(response.json())
```

## 🔐 Autenticazione (Opzionale)

Per abilitare l'autenticazione API key:

### 1. Imposta la variabile d'ambiente

```bash
export API_KEY="your-secret-key-here"
python api_gradio.py
```

### 2. Usa l'API key nelle richieste

**cURL:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "X-API-Key: your-secret-key-here" \
  -F "image=@ecg_sample.png"
```

**Python:**
```python
headers = {'X-API-Key': 'your-secret-key-here'}
response = requests.post('http://localhost:8000/analyze', files=files, headers=headers)
```

## 🧪 Test dell'API

Usa lo script di test fornito:

```bash
# Test singola immagine
python test_api.py ecg_sample.png

# Test batch
python test_api.py ecg1.png ecg2.png ecg3.png
```

## ⚙️ Configurazione

### Modello
Puoi cambiare il modello modificando `MODEL_ID` in `api_gradio.py`:

```python
MODEL_ID = "PULSE-ECG/PULSE-7B"
```

### Porta del server
Modifica la porta alla fine di `api_gradio.py`:

```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # Cambia 8000 con la porta desiderata
```

### Memory Management
Il modello viene caricato una volta all'avvio e mantenuto in cache per prestazioni ottimali. La memoria GPU viene gestita automaticamente.

## 📊 Prestazioni

- **Tempo di avvio**: ~30-60 secondi (caricamento modello)
- **Tempo di analisi**: ~10-30 secondi per immagine
- **Memoria GPU**: ~6-8GB VRAM (quantizzazione 8-bit)
- **Max immagini batch**: 5 per richiesta

## 🐛 Risoluzione Problemi

### Modello non si carica
```bash
# Verifica che LLaVA sia installato
cd LLaVA && pip install -e . && cd ..

# Verifica GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory
- Riduci la dimensione delle immagini prima dell'upload
- Assicurati che nessun altro processo stia usando la GPU
- Riavvia il server API

### Errori di import LLaVA
```bash
# Aggiungi LLaVA al PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/LLaVA"
python api_gradio.py
```

## 📝 Esempi Completi

### Esempio JavaScript (Node.js)

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const form = new FormData();
form.append('image', fs.createReadStream('ecg_sample.png'));
form.append('prompt', 'Detailed ECG analysis');

axios.post('http://localhost:8000/analyze', form, {
  headers: form.getHeaders()
}).then(response => {
  console.log(response.data);
}).catch(error => {
  console.error(error);
});
```

### Esempio con Gradio Client

```python
from gradio_client import Client

# Nota: questo è per l'app Gradio originale, non per l'API FastAPI
# Per l'API FastAPI usa requests come mostrato sopra
```

## ⚠️ Disclaimer

Questo strumento è solo per scopi educativi e di ricerca. Non deve essere utilizzato per diagnosi mediche reali. Consultare sempre professionisti medici qualificati.

## 📄 Licenza

MIT License - vedi file LICENSE

## 🤝 Supporto

Per problemi o domande:
- Apri una issue su GitHub
- Controlla la documentazione interattiva su `/docs`
- Verifica i log del server per errori dettagliati
