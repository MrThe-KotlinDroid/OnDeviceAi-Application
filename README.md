# AI Summarizer Android App

An Android application that uses on-device AI (T5 Model) to summarize long passages of text. This project demonstrates how to integrate ONNX Runtime for mobile inference with a modern Jetpack Compose UI.

## Features
- **On-Device Inference**: No internet connection required for summarization.
- **T5 Transformer Model**: Uses the popular T5 model optimized for mobile (INT8 quantization).
- **Custom C++ Tokenizer**: Includes a native SentencePiece-based tokenizer for fast and accurate text processing.
- **Jetpack Compose UI**: A clean, modern interface for input and result display.
- **Background Processing**: Inference is handled off the main thread to ensure a smooth user experience.

## Tech Stack
- **Kotlin**: Primary language for Android development.
- **Jetpack Compose**: Modern UI toolkit.
- **ONNX Runtime (Mobile)**: For running the optimized transformer models.
- **C++/JNI**: For high-performance native tokenization.
- **Coroutines**: For asynchronous processing.

## Getting Started
1. Clone the repository.
2. Ensure you have the required ONNX models (`encoder_model_int8.onnx` and `decoder_model_merged_int8.onnx`) in the `assets` folder.
3. Ensure `spiece.model` is present in the `assets` folder for the tokenizer.
4. Build and run the app on an Android device (API 24+).

## License
Apache License 2.0
