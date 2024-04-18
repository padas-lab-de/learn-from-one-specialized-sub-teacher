


# Import packages
import os 
os.environ["CUDA_VISIBLE_DEVICES"]= "6" # choose the GPU
device ="cuda"
import torch
import numpy as np
import evaluate
import datasets
import torch.nn as nn
import argparse
import transformers
import torch.nn.functional as F
datasets.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                         TrainingArguments, Trainer, DataCollatorWithPadding)


#Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using transformers v{transformers.__version__} and datasets v{datasets.__version__}")
print(f"Running on device: {device}")


# Function to compute the evaluation metric
def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Function to be used later in the loss function
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()




# Distillation Trainer
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, alpha_imdb,alpha_ce, alpha_mse, alpha_cos, alpha_corr, temperature,**kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.alpha_imdb = alpha_imdb
        self.alpha_ce = alpha_ce
        self.alpha_mse = alpha_mse
        self.alpha_cos = alpha_cos
        self.alpha_corr = alpha_corr
        self.temperature = temperature
        self.teacher.eval() # teacher is in the eval mode
        self.train_dataset.set_format(
            type=self.train_dataset.format["type"], columns=list(self.train_dataset.features.keys()))
        

    # Function to compute the distillation loss function
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs_stu = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
            
            }
        outputs_stu = model(**inputs_stu, labels=inputs["labels"].unsqueeze(0), output_hidden_states=True) # model takes the input and provide output
        loss = outputs_stu.loss  
        with torch.no_grad():
            outputs_tea = self.teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"], output_hidden_states=True) # , labels=inputs["labels"]


        assert self.alpha_ce >= 0.0
        assert self.alpha_corr >= 0.0
        assert self.alpha_mse >= 0.0
        assert self.alpha_cos >= 0.0
        assert self.alpha_imdb >= 0.0
        assert self.alpha_ce + self.alpha_corr + self.alpha_imdb + self.alpha_mse + self.alpha_cos > 0.0
        loss = self.alpha_imdb * loss
        if self.alpha_ce > 0.0:
            logits_stu = outputs_stu.logits
            logits_tea = outputs_tea.logits
            loss_fct = nn.KLDivLoss(reduction="batchmean")
            loss_logits = (loss_fct(
                F.log_softmax(logits_stu / self.temperature, dim=-1),
                F.softmax(logits_tea / self.temperature, dim=-1)) * (self.temperature ** 2))
            loss = loss + self.alpha_ce * loss_logits

        outputs_stu_hidden_states = outputs_stu.hidden_states
        outputs_tea_hidden_states = outputs_tea.hidden_states
        attention_mask = inputs['attention_mask']
        s_hidden_states = outputs_stu_hidden_states[-1]  # (bs, seq_length, dim)
        t_hidden_states = outputs_tea_hidden_states[-1]  # (bs, seq_length, dim)
        mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states).float()  # (bs, seq_length, dim)
        mask = mask.type(torch.ByteTensor).to(device)
        assert s_hidden_states.size() == t_hidden_states.size()
        dim = s_hidden_states.size(-1)
        s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
        z1 = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
        t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
        z2 = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

        if self.alpha_corr>0.0:
            z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
            z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)
            cross_corr = torch.matmul(z1_norm.T, z2_norm) / t_hidden_states_slct.size(0)
            on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(cross_corr).pow_(2).sum()
            loss_corr = on_diag + (5e-3 * off_diag)
            loss = loss + self.alpha_corr * loss_corr

        if self.alpha_cos > 0.0:
            target = z1.new(s_hidden_states_slct.size(0)).fill_(1)
            cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")
            loss_cos = cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
            loss = loss + self.alpha_cos * loss_cos

        if self.alpha_mse > 0.0:
            mse_loss_fct = nn.MSELoss(reduction='mean')
            loss_mse = mse_loss_fct(z1, z2)
            loss = loss + self.alpha_mse * loss_mse

        return  (loss, outputs_stu) if return_outputs else loss
   



def main():
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help= "Path to the student model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--teacher_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help= "Path to the teacher model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help= "The output directory where the final model will be saved.",
    )
    parser.add_argument(
        "--output_dir_intermed",
        default=None,
        type=str,
        required=True,
        help= "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="The training and testing batch size."
    )
    parser.add_argument(
        "--alpha_imdb", default=0.5, type=float, help="True imdb cross entropy loss."
    )
    parser.add_argument(
        "--temperature", default=2.0, type=float, help="Distillation temperature. Only for distillation."
    )
    parser.add_argument(
        "--alpha_ce", default=0.0, type=float, help="the weight of the logit distillation loss."
    )
    parser.add_argument(
        "--alpha_corr", default=0.005, type=float, help="The weight of the our proposed correlation loss for feature distillation."
    )
    parser.add_argument(
        "--alpha_mse", default=0.0, type=float, help="The mean square error loss for feature distillation."
    )
    parser.add_argument(
        "--alpha_cos", default=0.0, type=float, help="The cosine distance loss for feature distillation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    args = parser.parse_args()

    # Process the data
    # Download the data and the student model
    imdb_ds = load_dataset("imdb")
    student_model = args.student_model_name_or_path  #"distilbert-base-uncased"
    student_tokenizer = AutoTokenizer.from_pretrained(student_model)
    data_collator = DataCollatorWithPadding(tokenizer=student_tokenizer)

    # Functions to process the data
    def preprocess_function(examples):
        return student_tokenizer(examples["text"], truncation=True)

    def convert_examples_to_features(imdb, distilbert_tokenizer, num_train_examples, num_eval_examples):
        train_dataset = (imdb['train']
                         .select(range(num_train_examples))
                         .map(preprocess_function, batched=True))
        eval_dataset = (imdb['test']
                        .select(range(num_eval_examples))
                        .map(preprocess_function, batched=True))
        train_labels = torch.tensor(imdb["train"]["label"][:num_train_examples])
        test_labels = torch.tensor(imdb["test"]["label"][:num_eval_examples])
        eval_examples = imdb['test'].select(range(num_eval_examples))
        return train_dataset, eval_dataset, eval_examples, train_labels, test_labels

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    num_train_examples = len(imdb_ds['train'])
    num_eval_examples = len(imdb_ds['test'])
    train_ds, eval_ds, eval_examples, train_labels, test_labels = convert_examples_to_features(imdb_ds, student_tokenizer, num_train_examples, num_eval_examples)

    # Train
    logging_steps = len(train_ds) // args.batch_size
    student_training_args = TrainingArguments(
        output_dir=args.output_dir_intermed,
        save_strategy="no",
        overwrite_output_dir = True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0,
        logging_steps=logging_steps,
        do_eval=False,
    )
    print(f"Number of training examples: {train_ds.num_rows}")
    print(f"Number of validation examples: {eval_ds.num_rows}")
    print(f"Number of raw validation examples: {eval_examples.num_rows}")
    print(f"Logging steps: {logging_steps}")
    # Set the teacher model
    teacher_checkpoint = args.teacher_model_name_or_path # "./bert_model_fine_tuned_on_imdb"
    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_checkpoint).to(device)
    student_model = AutoModelForSequenceClassification.from_pretrained(student_model).to(device)
    distil_trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        alpha_imdb=args.alpha_imdb,
        alpha_ce=args.alpha_ce,
        alpha_mse=args.alpha_mse,
        alpha_cos=args.alpha_cos,
        alpha_corr=args.alpha_corr,
        temperature=args.temperature,
        args=student_training_args,
        train_dataset=train_ds,
        tokenizer=student_tokenizer,
    )
    distil_trainer.train()  # train
    distil_trainer.save_model(args.output_dir)  # save the mode to the specified path

    # Eval
    student_model = args.output_dir  # use the save student model
    student_tokenizer = AutoTokenizer.from_pretrained(student_model)
    student_trained = AutoModelForSequenceClassification.from_pretrained(student_model, num_labels=2,
                                                                           id2label=id2label, label2id=label2id)
    test_args = TrainingArguments(
        output_dir=args.output_dir_intermed,
        do_train=False,
        do_eval=True,
        overwrite_output_dir=True,
        per_device_eval_batch_size=16,
        dataloader_drop_last=False)
    student_trainer = Trainer(
        model=student_trained,
        args=test_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=student_tokenizer)
    # Print the results
    print(student_trainer.evaluate(eval_ds))


if __name__ == "__main__":
    main()