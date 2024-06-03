# Slurm Job Request Scripts

## EHRMamba - Pretrain
```
#!/bin/bash
#SBATCH --job-name=mamba_pretrain
#SBATCH --gres=gpu:4
#SBATCH --qos a100_amritk
#SBATCH -p a100
#SBATCH -c 24
#SBATCH --time=23:00:00
#SBATCH --mem=200G
#SBATCH --output=/h/afallah/odyssey/mamba_pretrain_a100-%j.out
#SBATCH --error=/h/afallah/odyssey/mamba_pretrain_a100-%j.err
#SBATCH --no-requeue

source /h/afallah/light/bin/activate

cd /h/afallah/odyssey/odyssey

export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

stdbuf -oL -eL srun python3 pretrain.py  \
                --model-type ehr_mamba \
                --is-decoder True \
                --exp-name mamba_pretrain_with_embeddings \
                --config-dir odyssey/models/configs \
                --data-dir odyssey/data/bigbird_data \
                --sequence-file patient_sequences/patient_sequences_2048.parquet \
                --id-file patient_id_dict/dataset_2048_multi_v2.pkl \
                --vocab-dir odyssey/data/vocab \
                --val-size 0.1 \
                --checkpoint-dir checkpoints
```

## MultiBird - Pretrain
```
#!/bin/bash
#SBATCH --job-name=multibird_pretrain
#SBATCH --gres=gpu:4
#SBATCH --qos a100_amritk
#SBATCH -p a100
#SBATCH -c 24
#SBATCH --time=23:00:00
#SBATCH --mem=200G
#SBATCH --output=/h/afallah/odyssey/multibird_a100-%j.out
#SBATCH --error=/h/afallah/odyssey/multibird_a100-%j.err
#SBATCH --no-requeue

source /h/afallah/light/bin/activate

cd /h/afallah/odyssey/odyssey

export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

stdbuf -oL -eL srun python3 pretrain.py  \
                --model-type cehr_bigbird \
                --exp-name multibird_pretrain \
                --config-dir odyssey/models/configs \
                --data-dir odyssey/data/bigbird_dat \
                --sequence-file patient_sequences_2048.parquet \
                --id-file dataset_2048_multi.pkl \
                --vocab-dir odyssey/data/vocab \
                --val-size 0.1 \
                --checkpoint-dir checkpoints
```


## MultiBird - Finetune
```
#!/bin/bash
#SBATCH --job-name=multibird_finetune
#SBATCH --gres=gpu:4
#SBATCH --qos a100_amritk
#SBATCH -p a100
#SBATCH -c 24
#SBATCH --time=23:59:00
#SBATCH --mem=200G
#SBATCH --output=/h/afallah/odyssey/multibird_finetune-%j.out
#SBATCH --error=/h/afallah/odyssey/multibird_finetune-%j.err
#SBATCH --no-requeue

source /h/afallah/light/bin/activate

cd /h/afallah/odyssey/odyssey

export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

stdbuf -oL -eL srun python3 finetune.py  \
                --model-type cehr_bigbird \
                --is-multi-model True \
                --exp-name multibird_finetune \
                --pretrained-path checkpoints/multibird_pretrain/multibird_pretrain/best.ckpt \
                --config-dir odyssey/models/configs \
                --data-dir odyssey/data/bigbird_data \
                --sequence-file patient_sequences_2048_multi.parquet \
                --id-file dataset_2048_multi.pkl \
                --vocab-dir odyssey/data/vocab \
                --val-size 0.15 \
                --valid_scheme few_shot \
                --num_finetune_patients all \
                --problem_type single_label_classification \
                --num_labels 2 \
                --checkpoint-dir checkpoints \
                --test_output_dir test_outputs \
                --tasks "mortality_1month los_1week readmission_1month c0 c1 c2" \
                --balance_guide "mortality_1month=0.5, los_1week=0.5, readmission_1month=0.5, c0=0.5, c1=0.5, c2=0.5"
```


## BigBird - Pretrain
```
#!/bin/bash
#SBATCH --job-name=bigbird_pretrain
#SBATCH --gres=gpu:4
#SBATCH --qos a100_amritk
#SBATCH -p a100
#SBATCH -c 24
#SBATCH --time=23:00:00
#SBATCH --mem=200G
#SBATCH --output=/h/afallah/odyssey/multibird_a100-%j.out
#SBATCH --error=/h/afallah/odyssey/multibird_a100-%j.err
#SBATCH --no-requeue

source /h/afallah/light/bin/activate

cd /h/afallah/odyssey/odyssey

export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

stdbuf -oL -eL srun python3 pretrain.py  \
                --model-type cehr_bigbird \
                --exp-name bigbird_pretrain \
                --config-dir models/configs \
                --data-dir data/bigbird_data \
                --sequence-file patient_sequences/patient_sequences_2048.parquet \
                --id-file patient_id_dict/dataset_2048_pretrain.pkl \
                --vocab-dir data/vocab \
                --val-size 0.1 \
                --checkpoint-dir checkpoints/bigbird_pretrain
```


## EHRMamba - Finetune Multidataset
```
#!/bin/bash
#SBATCH --job-name=mamba_finetune
#SBATCH --gres=gpu:4
#SBATCH --qos a100_amritk
#SBATCH -p a100
#SBATCH -c 24
#SBATCH --time=23:59:00
#SBATCH --mem=200G
#SBATCH --output=/h/afallah/odyssey/mamba_finetune-%j.out
#SBATCH --error=/h/afallah/odyssey/mamba_finetune-%j.err
#SBATCH --no-requeue

source /h/afallah/light/bin/activate

cd /h/afallah/odyssey/odyssey

export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

stdbuf -oL -eL srun python3 finetune.py  \
		        --seed 23 \
                --model-type ehr_mamba \
                --is-multi-model True \
                --is-decoder True \
                --exp-name mamba_finetune_with_embeddings_multihead \
                --pretrained-path checkpoints/mamba_pretrain_with_embeddings/best-v1.ckpt \
                --config-dir odyssey/models/configs \
                --data-dir odyssey/data/bigbird_data \
                --sequence-file patient_sequences_2048_multi_v2.parquet \
                --id-file dataset_2048_multi_v2.pkl \
                --vocab-dir odyssey/data/vocab \
                --val-size 0.1 \
                --valid_scheme few_shot \
                --num_finetune_patients all \
                --problem_type single_label_classification \
                --num_labels 2 \
                --checkpoint-dir checkpoints \
                --test_output_dir test_outputs \
                --tasks "mortality_1month readmission_1month los_1week c0 c1 c2" \
                --resume_checkpoint checkpoints/mamba_finetune_with_embeddings_multihead/best-v1.ckpt
```


## BigBird - Finetune Mortality
```
#!/bin/bash
#SBATCH --job-name=bigbird_finetune_mortality
#SBATCH --gres=gpu:2
#SBATCH --qos a100_amritk
#SBATCH -p a100
#SBATCH -c 6
#SBATCH --time=15:00:00
#SBATCH --mem=32G
#SBATCH --output=/h/afallah/odyssey/bigbird_finetune_mortality-%j.out
#SBATCH --error=/h/afallah/odyssey/bigbird_finetune_mortality-%j.err
#SBATCH --no-requeue

source /h/afallah/light/bin/activate

cd /h/afallah/odyssey/odyssey

export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

stdbuf -oL -eL srun python3 finetune.py  \
                --model-type cehr_bigbird \
                --exp-name mortality_1month_20000_patients \
                --pretrained-path checkpoints/bigbird_pretrain_with_conditions/pretrain_with_conditions/best-v1.ckpt \
                --label-name label_mortality_1month \
                --config-dir models/configs \
                --data-dir data/bigbird_data \
                --sequence-file patient_sequences/patient_sequences_2048_mortality.parquet \
                --id-file patient_id_dict/dataset_2048_mortality.pkl \
                --vocab-dir data/vocab \
                --val-size 0.1 \
                --valid_scheme few_shot \
                --num_finetune_patients '20000' \
                --problem_type 'single_label_classification' \
                --num_labels 2 \
                --checkpoint-dir checkpoints/bigbird_finetune_with_condition \
                --resume_checkpoint checkpoints/bigbird_finetune_with_condition/mortality_1month_20000_patients/best.ckpt
```


## BigBird - Finetune Condition
```
#!/bin/bash
#SBATCH --job-name=bigbird_finetune_condition
#SBATCH --gres=gpu:1
#SBATCH --qos a100_amritk
#SBATCH -p a100
#SBATCH -c 6
#SBATCH --time=15:00:00
#SBATCH --mem=32G
#SBATCH --output=/h/afallah/odyssey/bigbird_finetune_condition-%j.out
#SBATCH --error=/h/afallah/odyssey/bigbird_finetune_condition-%j.err
#SBATCH --no-requeue

source /h/afallah/light/bin/activate

cd /h/afallah/odyssey/odyssey

export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

stdbuf -oL -eL srun python3 finetune.py  \
                --model-type cehr_bigbird \
                --exp-name condition_50000_patients \
                --pretrained-path checkpoints/bigbird_pretrain_with_conditions/pretrain_with_conditions/best-v1.ckpt \
                --label-name all_conditions \
                --config-dir models/configs \
                --data-dir data/bigbird_data \
                --sequence-file patient_sequences/patient_sequences_2048_condition.parquet \
                --id-file patient_id_dict/dataset_2048_condition.pkl \
                --vocab-dir data/vocab \
                --val-size 0.1 \
                --valid_scheme few_shot \
                --num_finetune_patients '50000' \
                --problem_type 'multi_label_classification' \
                --num_labels 20 \
                --checkpoint-dir checkpoints/bigbird_finetune_with_condition \
                --resume_checkpoint checkpoints/bigbird_finetune_with_condition/condition_50000_patients/best.ckpt
```


## BigBird - Finetune Readmission
```
#!/bin/bash
#SBATCH --job-name=bigbird_finetune_readmission
#SBATCH --gres=gpu:2
#SBATCH --qos a100_amritk
#SBATCH -p a100
#SBATCH -c 6
#SBATCH --time=15:00:00
#SBATCH --mem=32G
#SBATCH --output=/h/afallah/odyssey/bigbird_finetune_readmission-%j.out
#SBATCH --error=/h/afallah/odyssey/bigbird_finetune_readmission-%j.err
#SBATCH --no-requeue

source /h/afallah/light/bin/activate

cd /h/afallah/odyssey/odyssey

export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

stdbuf -oL -eL srun python3 finetune.py  \
                --model-type cehr_bigbird \
                --exp-name readmission_1month_60000_patients \
                --pretrained-path checkpoints/bigbird_pretrain_with_conditions/pretrain_with_conditions/best-v1.ckpt \
                --label-name label_readmission_1month \
                --config-dir models/configs \
                --data-dir data/bigbird_data \
                --sequence-file patient_sequences/patient_sequences_2048_readmission.parquet \
                --id-file patient_id_dict/dataset_2048_readmission.pkl \
                --vocab-dir data/vocab \
                --val-size 0.1 \
                --valid_scheme few_shot \
                --num_finetune_patients '60000' \
                --problem_type 'single_label_classification' \
                --num_labels 2 \
                --checkpoint-dir checkpoints/bigbird_finetune_with_condition
```


## Bi-LSTM
```
#!/bin/bash
#SBATCH --job-name=baseline_lstm
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --time=6:00:00
#SBATCH -c 30
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --output=/h/afallah/odyssey/slurm/baseline_lstm-%j.out
#SBATCH --error=/h/afallah/odyssey/slurm/baseline_lstm-%j.err

#module --ignore_cache load cuda-11.8
#module load anaconda/3.10
#source activate light

source /h/afallah/light/bin/activate

cd /h/afallah/odyssey/slurm

export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

stdbuf -oL -eL srun python3 Bi-LSTM.py
```
