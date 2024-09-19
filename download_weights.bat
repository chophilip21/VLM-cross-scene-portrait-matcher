@echo off
:: Get the installation directory from the first argument
set app_dir=%1
echo Downloading model weights...

:: Download yoloworld.onnx to the app's assets/weights directory
gdown https://drive.google.com/uc?id=1IOCoXCeIF2kT9kRs0UZ4Y7C8UKnN7Ubc -O "%app_dir%\assets\weights\yoloworld.onnx"

:: Download grad_embedding.onnx to the app's assets/weights directory
gdown https://drive.google.com/uc?id=1lLRnJfzakiD2UF9lrsswe4oEEBtFsER9 -O "%app_dir%\assets\weights\grad_embedding.onnx"

pause