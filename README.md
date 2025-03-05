<p align="center">
    <img src="https://github.com/VectorInstitute/odyssey/assets/90617686/34ecf262-e455-4866-a870-300433d09bfe" width="50%">
</p>
<h1 style="text-align: center;">Odyssey</h1>
<p style="text-align: center;">A library for developing foundation models using Electronic Health Records (EHR) data.</p>

<p align="center">
    <a href="https://vectorinstitute.github.io/EHRMamba/">Visit our recent EHRMamba paper</a>
</p>

## Introduction

Odyssey is a comprehensive library designed to facilitate the development, training, and deployment of foundation models for Electronic Health Records (EHR). Recently, we used this toolkit to develop EHRMamba, a cutting-edge EHR foundation model that leverages the Mamba architecture and Multitask Prompted Finetuning (MPF) to overcome the limitations of existing transformer-based models. EHRMamba excels in processing long temporal sequences, simultaneously learning multiple clinical tasks, and performing EHR forecasting, significantly advancing the state of the art in EHR modeling.
<br><br>

## Key Features

The toolkit is structured into four main modules to streamline the development process:

1. **data**:
   - Tokenizes data and creates data splits for model training.
   - Provides a dataset class for model training.

2. **models**:
   - Implements models including XGBoost, LSTM, CEHR-BERT, BigBird, MultiBird, and EHRMamba.
   - Offers various embedding classes necessary for the models.

3. **evals**:
   - Includes tools for testing models on clinical prediction tasks and forecasting.
   - Provides evaluation metrics for thorough assessment of model performance.

4. **interp**:
   - Contains methods for interpreting model decisions.
   - Features interactive visualization of attention matrices for Transformer-based models.
   - Includes novel interpretability techniques for EHRMamba and gradient attribution methods.
<br><br>

## Data Preprocessing
The data extraction and preprocessing pipeline requires running the repository located at [MEDS_transforms](https://github.com/VectorInstitute/meds/tree/odyssey). The pipeline extracts and preprocesses the MIMIC-IV dataset to generate a patients' sequence of events.

### Installation

Clone and install the required repository locally:
```bash
git clone --branch odyssey https://github.com/VectorInstitute/meds.git
cd meds/MIMIC-IV_Example
pip install .
```

As mentioned in the [MEDS repository](https://github.com/VectorInstitute/meds/tree/odyssey) two (optional) hydra multirun job launchers for parallelizing extraction and pre-processing pipeline steps: [`joblib`](https://hydra.cc/docs/plugins/joblib_launcher/) (for local parallelism) and [`submitit`](https://hydra.cc/docs/plugins/submitit_launcher/) to launch things with slurm for cluster parallelism.

To use either of these, you need to install additional optional dependencies:

1. `pip install -e .[local_parallelism]` for joblib local parallelism support, or
2. `pip install -e .[slurm_parallelism]` for submitit cluster parallelism support.


### Running the Extract Pipeline

The `run_extract.sh` script performs the following steps:

- Unzips the MIMIC data files if necessary.
- Batches hospital lab events and chart events into multiple Parquet files to prevent memory issues during processing.
- Runs the `pre_MEDS` pipeline.
- Executes the `extract` pipeline, which:
  - Converts raw data.
  - Shards events.
  - Splits subjects into train, test, and holdout sets (note: in our case, we process all data as train and perform the split later in the Odyssey pipeline).
  - Converts data to sharded events.
  - Merges data into a MEDS cohort.

Note that the events that are extracted and included in the MEDS cohort are defined based on the [event configs files](https://github.com/VectorInstitute/meds/tree/odyssey/MIMIC-IV_Example/configs/event_configs_seq.yaml).

Run the extract pipeline using:
```bash
./run_extract.sh path_to_raw_data_dir path_to_preMEDS_dir path_to_MEDS_dir
```

#### Options:
- `do_unzip=true|false` (Optional) Unzip CSV files before processing (default: false).
- `batch_files` Run `batch_files.py` before processing (requires extra arguments):
  - `--lab_input=<path>` (Required if `batch_files` is set) Path to `labevents` CSV.
  - `--chart_input=<path>` (Required if `batch_files` is set) Path to `chartevents` CSV.

To use a specific stage runner file (e.g., to set different parallelism options), you can specify it as an additional argument

```bash
export N_WORKERS=5
./run_extract.sh path_to_raw_data_dir path_to_preMEDS_dir path_to_Extract_dir \
    stage_runner_fp=slurm_runner.yaml
```

The `N_WORKERS` environment variable set before the command controls how many parallel workers should be used
at maximum.

### Running the Preprocess Pipeline

The `run_preprocess` script executes the following steps:

- **filter_codes**: Remove specific codes from patient records if necessary.
- **filter_subjects**: Exclude patients with fewer than a minimum number of events.
- **filter_labs**: Remove lab records without numerical values.
- **filter_meds**: Exclude medications with a specific code (`0` in our case).
- **update_transfers**: Rename certain transfer codes.
- **add_age**: Compute and add patient age as the first event in records.
- **add_cls_token**: Add CLS token as the first event.
- **quantize_labs**: Quantize lab values based on binning strategy.
- **add_time_tokens**: Apply binning strategy to account for time intervals between events using predefined bins ([`hour2bin`](https://github.com/VectorInstitute/meds/tree/odyssey/MIMIC-IV_Example/time_bins/hour2bin.json) and [`minute2bin`](https://github.com/VectorInstitute/meds/tree/odyssey/MIMIC-IV_Example/time_bins/minute2bin.json)).
- **generate_sequence**: Concatenate all patient records to form the final sequence.

Run the preprocess pipeline using:
```bash
./run_preprocess.sh path_to_Extract_dir path_to_Processed_DIR
```

### Modifying Default Configuration

To customize the default parameters for each pipeline step, modify the following configuration files:
- `extract_MIMIC_seq.yaml`
- `preprocess_MIMIC_seq.yaml`

## Contributing

We welcome contributions from the community! Please open an issue. <br><br>

## Citation

If you use EHRMamba or Odyssey in your research, please cite our paper:
```
@misc{fallahpour2024ehrmamba,
      title={EHRMamba: Towards Generalizable and Scalable Foundation Models for Electronic Health Records},
      author={Adibvafa Fallahpour and Mahshid Alinoori and Arash Afkanpour and Amrit Krishnan},
      year={2024},
      eprint={2405.14567},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

