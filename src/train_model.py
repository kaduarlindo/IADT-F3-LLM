from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
import os

def train_model(dataset, model_name, output_dir="modelo_treinado"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"DEBUG: dataset original tem {len(dataset)} samples")
    if len(dataset) > 0:
        print(f"DEBUG: primeiro sample: {dataset[0]}")
        print(f"DEBUG: colunas: {dataset.column_names}")
    
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(examples):
        tokenized = tokenizer(
            examples["context"], 
            examples["question"],
            truncation=True, 
            padding="max_length", 
            max_length=512
        )
        
        starts, ends = [], []
        for ctx, ans in zip(examples["context"], examples["answer"]):
            start = ctx.find(ans)
            if start == -1:
                starts.append(0)
                ends.append(0)
            else:
                starts.append(start)
                ends.append(start + len(ans))
        
        tokenized["start_positions"] = starts
        tokenized["end_positions"] = ends
        return tokenized

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    
    print(f"DEBUG: dataset tokenizado tem {len(tokenized)} samples")
    if len(tokenized) > 0:
        print(f"DEBUG: primeiro sample tokenizado: {tokenized[0]}")
    else:
        print("❌ ERRO: Dataset tokenizado está vazio!")
        return

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        learning_rate=3e-5,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)