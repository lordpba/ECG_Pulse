---
title: ECG PULSE Analyzer
emoji: 🩺
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: mit
short_description: AI-powered ECG analysis using PULSE-7B model
---

# 🩺 ECG PULSE Analyzer

A multimodal ECG analysis application powered by the PULSE-ECG/PULSE-7B model for analyzing electrocardiogram images.

## Features

- **🔍 Advanced ECG Analysis**: Utilizes specialized AI models for ECG interpretation
- **📝 Custom Prompts**: Tailor your analysis requests with custom prompts
- **⚡ GPU Accelerated**: Optimized for fast inference on available hardware
- **🎯 Medical Focus**: Specialized for cardiac rhythm and abnormality detection

## How to Use

1. **Upload Image**: Click on "ECG Image" area and upload your ECG image (PNG, JPG, etc.)
2. **Customize Prompt**: Modify the analysis prompt to focus on specific aspects
3. **Analyze**: Click "🔍 Analyze ECG" to get AI-powered insights
4. **Review Results**: Read the detailed analysis in the results panel

## Example Prompts

- "Analyze this ECG for any arrhythmias or conduction abnormalities"
- "What is the heart rate and rhythm in this ECG?"
- "Identify any ST-segment changes or abnormalities"
- "Provide a comprehensive ECG interpretation including axis and intervals"

## Technical Details

- **Model**: PULSE-ECG/PULSE-7B (specialized for ECG analysis)
- **Framework**: Built with Gradio for easy interaction
- **Optimization**: 8-bit quantization for memory efficiency
- **Hardware**: Optimized for GPU acceleration when available

## Important Disclaimers

⚠️ **Medical Disclaimer**: This tool is for educational and research purposes only. Results should never be used for actual medical diagnosis. Always consult qualified medical professionals for medical decisions.

🔧 **Technical Note**: This is a demo version that may have limitations compared to a full local installation. For production medical research, consider running locally with proper setup.

## Local Installation

For the full-featured version with better performance:

```bash
git clone https://github.com/lordpba/ECG_Pulse.git
cd ECG_Pulse
pip install -r requirements.txt
python app_gradio.py
```

## Support

- **GitHub**: [ECG_Pulse Repository](https://github.com/lordpba/ECG_Pulse)
- **Issues**: Report issues on GitHub
- **Documentation**: See README.md in the repository

---

*Made with ❤️ for the medical AI community*