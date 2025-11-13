from transformers import pipeline

def load_trained_model(model_path="./modelo_finetunado"):
    qa_pipeline = pipeline("question-answering", model=model_path, tokenizer=model_path)
    return qa_pipeline

def get_treatment(qa_pipeline, symptom):
    question = "Qual é o tratamento indicado?"
    result = qa_pipeline(question=question, context=symptom)
    return result["answer"]