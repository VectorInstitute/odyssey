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
   - Gathers EHR datasets from HL7 FHIR resources.
   - Processes patient sequences for clinical tasks.
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
