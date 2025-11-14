from transformers import pipeline
import os

def load_trained_model():
    model_path = "modelo_finetunado"  # Remove './'
    
    # Verifica se o modelo existe localmente
    if os.path.exists(model_path):
        qa_pipeline = pipeline(
            "question-answering", 
            model=model_path, 
            tokenizer=model_path,
            local_files_only=True  # Força carregamento local
        )
    else:
        # Se não existir, usa o modelo padrão
        qa_pipeline = pipeline(
            "question-answering", 
            model="pierreguillou/bert-large-cased-squad-v1.1-portuguese"
        )
    
    return qa_pipeline

def get_treatment(qa_pipeline, symptom):
    question = "Qual é o tratamento indicado?"
    result = qa_pipeline(question=question, context=symptom)
    return result["answer"]