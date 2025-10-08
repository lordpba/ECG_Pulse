# 🩺 PULSE-7B ECG Analysis Interface

A powerful Gradio-based web interface for analyzing electrocardiogram (ECG) images using the PULSE-7B multimodal large language model specialized in ECG interpretation.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **State-of-the-art ECG Analysis**: Powered by PULSE-7B, a specialized 7-billion parameter model trained specifically for ECG interpretation
- **Interactive Web Interface**: User-friendly Gradio interface for easy ECG image upload and analysis
- **Customizable Prompts**: Modify analysis prompts to focus on specific aspects (rhythm, intervals, abnormalities, etc.)
- **Memory Optimization**: 8-bit quantization and multi-GPU support for efficient inference
- **Real-time Processing**: Fast analysis with automatic GPU memory management

## 🔬 About PULSE-7B

PULSE-7B is a multimodal large language model specifically designed for ECG image interpretation. Built on the LLaVA architecture, it has been trained on comprehensive ECG datasets and can:

- Analyze ECG rhythm and rate
- Identify intervals (PR, QRS, QT)
- Detect cardiac abnormalities
- Provide detailed clinical interpretations
- Support both printed and digital ECG formats

**Paper**: [Teach Multimodal LLMs to Comprehend Electrocardiographic Images](https://arxiv.org/abs/2410.19008)
**Model**: [PULSE-ECG/PULSE-7B on Hugging Face](https://huggingface.co/PULSE-ECG/PULSE-7B)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended: 12GB+ VRAM)
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/pulse-ecg-interface.git
   cd pulse-ecg-interface
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app_gradio.py
   ```

5. **Open your browser** and navigate to `http://127.0.0.1:7860`

## 💻 Usage

1. **Upload ECG Image**: Click on the "ECG Image" area to upload your electrocardiogram image
2. **Customize Prompt** (Optional): Modify the analysis prompt to focus on specific aspects:
   - General analysis: `"Provide a comprehensive ECG analysis including rhythm, rate, axis, intervals, and any abnormalities."`
   - Specific focus: `"Focus on detecting arrhythmias and conduction abnormalities in this ECG."`
   - Educational: `"Explain this ECG as if teaching a medical student, highlighting key features."`
3. **Analyze**: Click the "Analyze ECG" button to get your results
4. **Review Results**: The AI-generated analysis will appear in the results panel

## 🔧 Configuration

### GPU Memory Requirements

- **Minimum**: 8GB VRAM (with 8-bit quantization)
- **Recommended**: 12GB+ VRAM for optimal performance
- **Multi-GPU**: Automatic distribution across available GPUs

### Memory Optimization Features

- **8-bit Quantization**: Reduces memory usage by ~50%
- **Automatic GPU Management**: Smart memory allocation and cleanup
- **Multi-GPU Support**: Distributes model across multiple GPUs automatically

## 📁 Project Structure

```
pulse-ecg-interface/
├── app_gradio.py          # Main Gradio application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore           # Git ignore rules
└── LLaVA/               # Auto-downloaded LLaVA repository
```

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: LLaVA (Large Language and Vision Assistant)
- **Parameters**: 7 billion
- **Quantization**: 8-bit with BitsAndBytesConfig
- **Vision Encoder**: CLIP ViT-Large
- **Language Model**: LLaMA-based architecture

### Performance Optimizations
- Memory-efficient loading with `low_cpu_mem_usage=True`
- Automatic device mapping for multi-GPU setups
- Torch inference mode for reduced memory overhead
- Dynamic memory cleanup after each inference

## 🔬 Example Prompts

### General Analysis
```
Please provide a detailed analysis of this ECG image. Focus on rhythm, rate, intervals, and any abnormalities you can identify.
```

### Focused Analysis
```
Analyze this ECG for arrhythmias and conduction abnormalities. Specify the type and clinical significance.
```

### Educational Format
```
Explain this ECG step by step as if teaching a medical student, highlighting normal and abnormal findings.
```

### Multi-language Support
```
Fornisci un'analisi dettagliata di questo ECG in italiano, concentrandoti su ritmo, frequenza e anomalie.
```

## ⚠️ Important Disclaimers

- **For Research and Educational Use Only**: This tool is designed for research, educational, and demonstration purposes
- **Not for Clinical Diagnosis**: Do not use for actual medical diagnosis or patient care decisions
- **AI Limitations**: Results are generated by an AI model and may contain errors or inaccuracies
- **Professional Consultation**: Always consult qualified healthcare professionals for medical interpretations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PULSE-ECG Team** for developing the PULSE-7B model
- **LLaVA Team** for the base multimodal architecture
- **Hugging Face** for model hosting and transformers library
- **Gradio Team** for the excellent web interface framework

## 📚 Citation

If you use this interface in your research, please cite the original PULSE paper:

```bibtex
@article{liu2024pulse,
  title={Teach Multimodal LLMs to Comprehend Electrocardiographic Images},
  author={Ruoqi Liu, Yuelin Bai, Xiang Yue, Ping Zhang},
  journal={arXiv preprint arXiv:2410.19008},
  year={2024}
}
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/pulse-ecg-interface/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pulse-ecg-interface/discussions)
- **Original Paper**: [arXiv:2410.19008](https://arxiv.org/abs/2410.19008)

---

**⚡ Powered by PULSE-7B | Built with ❤️ for the medical AI community**