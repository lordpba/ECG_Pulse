# 🩺 ECG PULSE - Multimodal ECG Analysis with PULSE-7B

A powerful web application for analyzing electrocardiogram (ECG) images using the PULSE-7B multimodal language model. This tool provides detailed medical insights from ECG images through an intuitive Gradio interface.

## ✨ Features

- **Advanced ECG Analysis**: Utilizes the specialized PULSE-7B model for accurate ECG interpretation
- **Customizable Prompts**: Tailor analysis requests with custom prompts for specific medical insights
- **GPU Optimization**: 8-bit quantization and multi-GPU support for efficient model inference
- **Memory Management**: Automatic memory cleanup and optimization for stable performance
- **User-Friendly Interface**: Clean, responsive web interface built with Gradio
- **Real-time Processing**: Fast analysis with optimized inference pipeline

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended: 12GB+ VRAM)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/lordpba/ECG_Pulse.git
   cd ECG_Pulse
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up LLaVA environment** (if not already present)
   ```bash
   git clone https://github.com/haotian-liu/LLaVA.git
   cd LLaVA
   pip install -e .
   cd ..
   ```

5. **Launch the application**
   ```bash
   python app_gradio.py
   ```

6. **Open your browser** and navigate to `http://127.0.0.1:7860`

## 💻 Usage

1. **Upload ECG Image**: Click on the "ECG Image" area to upload your electrocardiogram image
2. **Customize Prompt** (Optional): Modify the analysis prompt to focus on specific aspects:
   - General analysis: "Provide a comprehensive ECG analysis including rhythm, rate, and abnormalities"
   - Specific focus: "Focus on arrhythmias and conduction abnormalities"
   - Educational: "Explain this ECG for medical students with key teaching points"
3. **Analyze**: Click the "Analyze ECG" button to get AI-powered insights
4. **Review Results**: Read the detailed analysis in the results panel

## 🔧 Configuration

### GPU Memory Optimization

The application automatically optimizes for available GPU memory:

- **8-bit Quantization**: Reduces memory usage by ~50%
- **Multi-GPU Support**: Automatically distributes model across available GPUs
- **Memory Cleanup**: Automatic cleanup after each analysis

### Model Configuration

Edit `MODEL_ID` in `app_gradio.py` to use different models:

```python
MODEL_ID = "PULSE-ECG/PULSE-7B"  # Default specialized ECG model
```

## 📋 Requirements

### System Requirements

- **GPU**: NVIDIA GPU with 12GB+ VRAM (dual RTX 3060 12GB tested)
- **RAM**: 16GB+ system RAM recommended
- **Storage**: 20GB+ free space for model downloads

### Python Dependencies

- gradio
- torch
- transformers==4.37.2
- Pillow
- bitsandbytes
- accelerate
- sentencepiece
- protobuf

See `requirements.txt` for complete list.

## 🏗️ Architecture

```
ECG_Pulse/
├── app_gradio.py          # Main application with Gradio interface
├── requirements.txt       # Python dependencies
├── setup.sh              # Setup script
├── README.md             # This file
├── LICENSE               # MIT License
└── LLaVA/                # LLaVA framework (local dependency)
```

## 🔬 Model Details

- **Base Model**: PULSE-ECG/PULSE-7B
- **Architecture**: LLaVA-based multimodal transformer
- **Specialization**: ECG image analysis and medical interpretation
- **Context Length**: 2048 tokens
- **Quantization**: 8-bit for memory efficiency

## ⚠️ Important Disclaimers

- **Medical Use**: This tool is for **educational and research purposes only**
- **Not for Diagnosis**: Results should **never be used for actual medical diagnosis**
- **Professional Consultation**: Always consult qualified medical professionals for medical decisions
- **AI Limitations**: AI-generated analyses may contain errors or inaccuracies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PULSE-ECG Team** for the specialized ECG analysis model
- **LLaVA Project** for the multimodal framework
- **Hugging Face** for model hosting and transformers library
- **Gradio** for the intuitive web interface

## 📊 Performance

- **Analysis Time**: ~10-30 seconds per ECG (depending on GPU)
- **Memory Usage**: ~6-8GB VRAM with 8-bit quantization
- **Supported Formats**: PNG, JPG, JPEG, BMP, TIFF
- **Max Image Size**: 2048x2048 pixels recommended

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce image size
   - Ensure no other GPU-intensive processes are running
   - Restart the application

2. **Model Loading Issues**
   - Check internet connection for initial model download
   - Verify CUDA installation
   - Ensure sufficient disk space

3. **LLaVA Import Errors**
   - Ensure LLaVA folder is present
   - Check Python path configuration
   - Reinstall LLaVA dependencies

### Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Provide system specifications and error logs

---

**Made with ❤️ for the medical AI community**
