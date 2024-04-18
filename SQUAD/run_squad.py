

# Import packages
import os 
os.environ["CUDA_VISIBLE_DEVICES"]= "1" # use the gpu number 1 
device ="cuda"
from pprint import pprint
import torch
import pandas as pd
import datasets
import transformers
datasets.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()
from datasets import load_dataset , load_metric
from transformers import (AutoTokenizer, AutoModelForQuestionAnswering, 
                          default_data_collator, QuestionAnsweringPipeline)
from  question_answering import *
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using transformers v{transformers.__version__} and datasets v{datasets.__version__}")
print(f"Running on device: {device}")


if __name__ == "__main__":

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
        parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
        args = parser.parse_args()

        # Process the data
        squad_ds = load_dataset("squad")
        pprint(squad_ds['train'][0])
        pprint(squad_ds['validation'][666])
        answers_ds = squad_ds.map(lambda x : {'num_possible_answers' : pd.Series(x['answers']['answer_start']).nunique()})
        answers_ds.set_format('pandas')
        answers_df = answers_ds['validation'][:]
        print(answers_df['num_possible_answers'].value_counts())

        # Fine-tune BERT-base
        model_checkpoint = args.model
        teacher_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        num_train_examples = len(squad_ds['train'])
        num_eval_examples = len(squad_ds['validation'])
        train_ds, eval_ds, eval_examples = convert_examples_to_features(squad_ds, teacher_tokenizer, num_train_examples, num_eval_examples)

        # Configure and initialise the trainer
        logging_steps = len(train_ds) // args.batch_size

        teacher_args = QuestionAnsweringTrainingArguments(
            output_dir=args.output_dir_intermed,
            evaluation_strategy = "epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=0.01,
            logging_steps=logging_steps)

        print(f"Number of training examples: {train_ds.num_rows}")
        print(f"Number of validation examples: {eval_ds.num_rows}")
        print(f"Number of raw validation examples: {eval_examples.num_rows}")
        print(f"Logging steps: {logging_steps}")


        # Train model
        def teacher_init():
            return AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

        teacher_trainer = QuestionAnsweringTrainer(
            model_init = teacher_init,
            args=teacher_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            eval_examples=eval_examples,
            tokenizer=teacher_tokenizer)

        teacher_trainer.train()
        teacher_trainer.save_model(args.output_dir)

        # Evaluate fine-tuned model
        trained_model_checkpoint = args.output_dir
        model_tokenizer = AutoTokenizer.from_pretrained(trained_model_checkpoint)
        model_finetuned = AutoModelForQuestionAnswering.from_pretrained(trained_model_checkpoint)
        teacher_trainer = QuestionAnsweringTrainer(
            model=model_finetuned,
            args=teacher_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            eval_examples=eval_examples,
            tokenizer=teacher_tokenizer)

        print(teacher_trainer.evaluate())
