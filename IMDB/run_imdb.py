
# import packages
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "6" # use the gpu number 1
device ="cuda"
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer,AutoTokenizer
from transformers import DataCollatorWithPadding
from datasets import load_dataset
import evaluate
import argparse



# Function to compute the eval metric of the model
def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    return acc


def main():
    # Download dataset
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        required=True,
        help="Path to the teacher model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the final model will be saved.",
    )
    parser.add_argument(
        "--output_dir_intermed",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="The training and testing batch size."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")

    args = parser.parse_args()
    imdb = load_dataset("imdb")
    model = args.model  # ""
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Preprocess dataset
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    # Define the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model, num_labels=2, id2label=id2label, label2id=label2id
    )

    # Define the training args
    training_args = TrainingArguments(
        output_dir= args.output_dir_intermed,
        learning_rate= args.learning_rate,
        per_device_train_batch_size= args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0,
        save_strategy="no",
        do_eval=False,
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train and save the model
    trainer.train()
    trainer.save_model(args.output_dir)

    # Eval
    model_checkpoint = args.output_dir
    model_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model_finetuned = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2,
                                                                         id2label=id2label, label2id=label2id)


    test_args = TrainingArguments(
        output_dir=args.output_dir_intermed,
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=args.batch_size,
        dataloader_drop_last=False)

    trainer2 = Trainer(
        model=model_finetuned,
        args=test_args,
        tokenizer=model_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics)

    print(trainer2.evaluate(tokenized_imdb["test"]))


if __name__ == "__main__":
    main()