from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
import torch
import os

def train_model(dataset, model_name, output_dir="modelo_treinado"):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def preprocess(example):
        inputs = tokenizer(
            example["context"], example["question"],
            truncation=True, padding="max_length", max_length=512
        )
        start = example["context"].find(example["answer"])
        if start == -1:
            start, end = 0, 0
        else:
            end = start + len(example["answer"])
        inputs["start_positions"] = start
        inputs["end_positions"] = end
        return inputs

    tokenized = dataset.map(preprocess)
    training_args = TrainingArguments(
                    output_dir=output_dir,
                    evaluation_strategy="no",
                    per_device_train_batch_size=2,
                    num_train_epochs=2,
                    save_strategy="epoch",
                    logging_dir=f"{output_dir}/logs",
                    learning_rate=3e-5
                    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)