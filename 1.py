import google.generativeai as genai

# Initialize your GenerativeModel (you might need your API key here)
genai.configure(api_key="AIzaSyC59TU3GZtrZqpU1y7sGZJBXaErrpOzszo")

# List all available models
models = genai.list_models()
for model in models:
    print(f"Model: {model.name}")
    for method in model.supported_generation_methods:
        print(f"- Supports: {method}")