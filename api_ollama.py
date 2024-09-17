import ollama

def get_models():
    models = ollama.list()
    model_names = [model['name'] for model in models['models']]
    return model_names

model_names = get_models()

print(model_names)

