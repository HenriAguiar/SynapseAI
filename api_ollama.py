import ollama

def get_models():
    try:
        # Tenta listar os modelos disponíveis
        models = ollama.list()
        model_names = [model['name'] for model in models['models']]
        return model_names
    except Exception as e:
        print(f"Erro ao conectar ou listar modelos do Ollama: {e}")
        # Retorna modelos manualmente configurados como fallback
        return ["llama3.1:8b-instruct-q8_0"]

model_names = get_models()

if model_names:
    print("Modelos disponíveis:", model_names)
else:
    print("Nenhum modelo encontrado ou servidor indisponível.")
