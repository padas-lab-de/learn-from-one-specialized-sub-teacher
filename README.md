## Learn From One Specialized Sub-Teacher: One-to-One Mapping for Feature-Based Knowledge Distillation
#### This repository contains the official implementation of the paper [Learn From One Specialized Sub-Teacher: One-to-One Mapping for Feature-Based Knowledge Distillation](https://aclanthology.org/2023.findings-emnlp.882.pdf) (EMNLP-Findings 2023). A short version of this paper was at first accepted at ICML-Neural-Compression workshop 2023.

![KD_camer_ready_version drawio](https://github.com/Khsaadi/learn-from-one-specialized-sub-teacher/assets/58224339/55cd0c29-f027-4580-90bb-7cefa60068ee)


#### For more details about the technical details, please refer to our paper.

**Installation**

Run command below to install the environment (using python3.9):

```
pip install -r requirements.txt
```

**Data Preparation**



Run command below to get SQUAD-V1 data:

```
download the data from: https://drive.google.com/drive/folders/1IHRfVG0VBE22-0AAiWODNy80T8ckA-yK?usp=sharing
```
Run command below to get GLUE data:

```
python download_glue_data.py --data_dir glue_data --task all
```

**Student Initialization**

The 6-layer DistilBERT is the initial student model.

**Fine-tuning**

Run command below to get fine-tuned teacher model for every task of GLUE, SQUAD-V1, and IMDB:

```
# GLUE (e.g., RTE)
  python run_glue.py   --model_type bert   --model_name_or_path bert-base-uncased   --task_name rte   --data_dir ./glue_data/RTE/   --do_lower_case   --max_seq_length 128   --do_train   --per_gpu_train_batch_size 32   --per_gpu_eval_batch_size 32   --learning_rate 5e-5   --num_train_epochs 3.0   --output_dir ./model/RTE/teacher/
# SQUAD
 python run_squad.py --model bert-base-uncased  --output_dir  ./output   --output_dir_intermed ./output_intermed

# IMDB
 python run_imdb.py --model bert-base-uncased  --output_dir  ./output   --output_dir_intermed ./output_intermed
```

**Distillation**

Run command below to get distilled student model for every task of GLUE, SQUAD-V1, and IMDB:

```
# GLUE (e.g., RTE)
python run_glue_w_distillation.py  --teacher_model_name_or_path ./model/RTE/teacher     --student_model_name_or_path distilbert-base-cased   --task_name rte  --max_seq_length 128 --output_dir ./output  --alpha_ce 0 --alpha_glue 0.5 --alpha_corr 0.005 --alpha_mse 0 --alpha_cos 0 --do_mask True --num_train_epochs 3.0

# SQUAD
python run_squad_w_distillation.py --model_type distilbert --model_name_or_path distilbert-base-uncased --output_dir ./run_squad_w_distillation_CrossCorrLoss --teacher_type bert --teacher_name_or_path ./bert-base-uncased-finetuned-squad-v1 --data_dir data  --train_file train.json --predict_file dev.json  --do_train --do_eval  --do_lower_case --save_steps 5541 --logging_steps 5541 --alpha_cos 0.0 --alpha_corr 0.5 --alpha_ce 0.0 --alpha_squad 0.5 --num_train_epochs 3

# IMDB
 python run_imdb_w_distillation.py  --student_model_name_or_path  distilbert-base-uncased  --teacher_model_name_or_path ./bert_model_fine_tuned_on_imdb --output_dir ./output --output_dir_intermed ./output_intermed
```

**Results**

On the GLUE development set:
![imgres](https://github.com/Khsaadi/learn-from-one-specialized-sub-teacher/assets/58224339/e5a83ecd-0948-4840-b1c2-b89598ce5a6a)



**Cite**
```
@inproceedings{saadi-etal-2023-learn,
    title = "Learn From One Specialized Sub-Teacher: One-to-One Mapping for Feature-Based Knowledge Distillation",
    author = "Saadi, Khouloud  and Mitrovi{\'c}, Jelena  and Granitzer, Michael",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    year = "2023",
    doi = "10.18653/v1/2023.findings-emnlp.882",
    pages = "13235--13245"
 }
```


