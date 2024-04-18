


# Import Packages
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # use the gpu number 1
device = "cuda"
import logging
import random
from dataclasses import dataclass, field
from typing import Optional
import evaluate
import numpy as np
from datasets import load_dataset
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,

)
import torch.nn.functional as F
import torch.nn as nn
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.40.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


# Data prepration
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass # Prepare the models arguments
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    student_model_name_or_path: str = field(
        metadata={"help": "Path to the student model or model identifier from huggingface.co/models"}
    )
    teacher_model_name_or_path: str = field(
        metadata={"help": "Path to the teacher model or model identifier from huggingface.co/models"}
    )
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    overwrite_output_dir: bool = field(
        default=False, metadata={"help": "Overwrite the output dir or not."}
    )

    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training or not."},
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation or not."},
    )
    do_mask: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation or not."},
    )
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})

    batch_size: int = field(
        default=16,
        metadata={
            "help": (
                "The training and testing batch size "
            )
        },
    )
    alpha_ce: float = field(
        default=0.5,
        metadata={
            "help": (
                "the weight of the logit distillation loss "
            )
        },
    )
    alpha_glue: float = field(
        default=0.5,
        metadata={
            "help": (
                "The weight of the usual cross entropy loss "
            )
        },
    )
    alpha_corr: float = field(
        default=0.005,
        metadata={
            "help": (
                "The weight of the our proposed correlation loss for feature distillation "
            )
        },
    )
    alpha_mse: float = field(
        default=0.5,
        metadata={
            "help": (
                "The mean square error loss for feature distillation"
            )
        },
    )
    alpha_cos: float = field(
        default=0.5,
        metadata={
            "help": (
                "The cosine distance loss for feature distillation"
            )
        },
    )
    temperature: float = field(
        default=2.0,
        metadata={
            "help": (
                "The temperature in the logit distillation"
            )
        },
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={
            "help": (
                "The initial learning rate for Adam."
            )
        },
    )



parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
model_args, data_args = parser.parse_args_into_dataclasses()


# Function to be used later in computing the newly proposed loss function
class MyDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(MyDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
    def forward(self, X1, X2):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
            d = X1.size(dim=1)
            Y = (binomial.sample((d,)) * (1.0 / (1 - self.p))).to(device)
            x1 = X1[:, Y.bool()]
            x2 = X2[:, Y.bool()]
            return x1, x2


# Function to be used later in computing the loss function
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# Set the distillation trainer function
class DistillationTrainer(Trainer):

    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        self.train_dataset.set_format(
            type=self.train_dataset.format["type"], columns=list(self.train_dataset.features.keys()))

    def compute_loss(self, model, inputs, return_outputs=False):

        inputs_stu = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],
        }
        outputs_stu = model(**inputs_stu, labels=inputs["labels"].unsqueeze(0),
                            output_hidden_states=True)  # model takes the input and provide output
        loss = outputs_stu.loss
        with torch.no_grad():
            outputs_tea = self.teacher(
                input_ids=inputs["input_ids"],
                # token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"], output_hidden_states=True)  # , labels=inputs["labels"]
        loss = model_args.alpha_glue*loss

        assert model_args.alpha_ce >= 0.0
        assert model_args.alpha_corr >= 0.0
        assert model_args.alpha_mse >= 0.0
        assert model_args.alpha_cos >= 0.0
        assert model_args.alpha_glue >= 0.0
        assert model_args.alpha_ce + model_args.alpha_corr + model_args.alpha_glue +model_args.alpha_mse + model_args.alpha_cos > 0.0

        if model_args.alpha_ce > 0.0:
            logits_stu = outputs_stu.logits
            logits_tea = outputs_tea.logits
            loss_fct = nn.KLDivLoss(reduction="batchmean")
            loss_logits = (loss_fct(
                F.log_softmax(logits_stu / model_args.temperature, dim=-1),
                F.softmax(logits_tea / model_args.temperature, dim=-1)) * (model_args.temperature ** 2))
            loss = loss + model_args.alpha_ce * loss_logits

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

        if model_args.alpha_corr>0.0:
            z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
            z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)
            cross_corr = torch.matmul(z1_norm.T, z2_norm) / t_hidden_states_slct.size(0)
            on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(cross_corr).pow_(2).sum()
            loss_corr = on_diag + (5e-3 * off_diag)
            loss = loss + model_args.alpha_corr * loss_corr

        if model_args.alpha_corr>0.0 and model_args.do_mask==True :
            cont = 0.5  # You can specify the dropout probability here.
            z1, z2 = MyDropout(cont).forward(z1, z2)
            z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
            z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)
            cross_corr = torch.matmul(z1_norm.T, z2_norm) / t_hidden_states_slct.size(0)
            on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(cross_corr).pow_(2).sum()
            loss_corr = on_diag + (5e-3 * off_diag)
            loss = loss + model_args.alpha_corr * loss_corr


        if model_args.alpha_cos > 0.0:
            target = z1.new(s_hidden_states_slct.size(0)).fill_(1)
            cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")
            loss_cos = cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
            loss = loss + model_args.alpha_cos * loss_cos

        if model_args.alpha_mse > 0.0:
            mse_loss_fct = nn.MSELoss(reduction='mean')
            loss_mse = mse_loss_fct(z1, z2)
            loss = loss + model_args.alpha_mse * loss_mse

        return (loss, outputs_stu) if return_outputs else loss



def main():

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(model_args.output_dir) and model_args.do_train and not model_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(model_args.output_dir)
        if last_checkpoint is None and len(os.listdir(model_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({model_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and model_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Download the data
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "nyu-mll/glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.student_model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.student_model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    student_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.student_model_name_or_path,
        from_tf=bool(".ckpt" in model_args.student_model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    ).to(device)

    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.teacher_model_name_or_path
    ).to(device)

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            student_model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in student_model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        student_model.config.label2id = label_to_id
        student_model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        student_model.config.label2id = {l: i for i, l in enumerate(label_list)}
        student_model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result



    raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if model_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if model_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Log a few random samples from the training set:
    if model_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name)
    else:
        metric = evaluate.load("accuracy")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

# Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
# we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif model_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Set the trainer arguments
    training_args = TrainingArguments(
        output_dir=model_args.output_dir,
        per_device_train_batch_size=model_args.batch_size,
        num_train_epochs=model_args.num_train_epochs,
        do_train=model_args.do_train,
        do_eval=model_args.do_eval,
    )

    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset if model_args.do_train else None,
        eval_dataset=eval_dataset if model_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if model_args.do_train:
        checkpoint = None
        if model_args.resume_from_checkpoint is not None:
            checkpoint = model_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if model_args.do_eval:
        logger.info("*** Evaluate ***")
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}
        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)


if __name__ == "__main__":
    main()