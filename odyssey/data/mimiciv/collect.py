"""Collect data from the FHIR database and save to csv files."""

import json
import logging
import os
from ast import literal_eval
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fhir.resources.condition import Condition
from fhir.resources.encounter import Encounter
from fhir.resources.medication import Medication
from fhir.resources.medicationrequest import MedicationRequest
from fhir.resources.observation import Observation
from fhir.resources.patient import Patient
from fhir.resources.procedure import Procedure
from sqlalchemy import MetaData, Table, create_engine, select
from tqdm import tqdm

from odyssey.utils.log import setup_logging


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


PATIENT = "patient"
ENCOUNTER = "encounter"
PROCEDURE = "procedure"
MEDICATION = "medication"
LAB = "lab"
CONDITION = "condition"
DATA_COLLECTION_CONFIG = {
    PATIENT: {
        "table_name": PATIENT,
        "columns": [
            "patient_id",
            "birthDate",
            "gender",
            "deceasedBoolean",
            "deceasedDateTime",
        ],
        "save_path": "patients.csv",
    },
    ENCOUNTER: {
        "table_name": ENCOUNTER,
        "columns": ["patient_id", "length", "encounter_ids", "starts", "ends"],
        "save_path": "encounters.csv",
    },
    PROCEDURE: {
        "table_name": PROCEDURE,
        "columns": [
            "patient_id",
            "length",
            "proc_codes",
            "proc_dates",
            "encounter_ids",
        ],
        "save_path": "procedures.csv",
    },
    MEDICATION: {
        "table_name": "medication_request",
        "columns": [
            "patient_id",
            "length",
            "med_codes",
            "med_dates",
            "encounter_ids",
        ],
        "save_path": "med_requests.csv",
    },
    LAB: {
        "table_name": "observation_labevents",
        "columns": [
            "patient_id",
            "length",
            "lab_codes",
            "lab_values",
            "lab_units",
            "lab_dates",
            "encounter_ids",
        ],
        "filter_columns": [
            "lab_codes",
            "lab_values",
            "lab_units",
            "lab_dates",
            "encounter_ids",
        ],
        "save_path": "labs.csv",
    },
    CONDITION: {
        "table_name": CONDITION,
        "columns": ["patient_id", "length", "encounter_conditions"],
        "save_path": "conditions.csv",
    },
}


def filter_lab_codes(row: pd.Series, vocab: List[str]) -> pd.Series:
    """Filter out lab codes that are not in the vocabulary.

    Parameters
    ----------
    row : pd.Series
        The row to filter.
    vocab : List[str]
        The vocabulary to filter by.

    Returns
    -------
    pd.Series
        The filtered row.

    """
    for col in DATA_COLLECTION_CONFIG[LAB]["filter_columns"]:
        row[col] = literal_eval(row[col])
    indices = [i for i, code in enumerate(row["lab_codes"]) if code in vocab]
    for col in DATA_COLLECTION_CONFIG[LAB]["filter_columns"]:
        row[col] = [row[col][i] for i in indices]
    row["length"] = len(row["lab_codes"])

    return row


class FHIRDataCollector:
    """Collect data from the FHIR database and save to csv files."""

    def __init__(
        self,
        db_path: str,
        schema: str = "mimic_fhir",
        save_dir: str = "data_files",
        buffer_size: int = 10000,
    ) -> None:
        self.engine = create_engine(db_path)
        self.metadata = MetaData()
        self.schema = schema
        self.save_dir = save_dir
        self.buffer_size = buffer_size

        self.vocab_dir = os.path.join(self.save_dir, "vocab")
        self.csv_dir = os.path.join(self.save_dir, "csv_files")

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.vocab_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

    def save_to_csv(
        self,
        buffer: List[Dict[str, Any]],
        columns: List[str],
        save_path: str,
        flush: bool = False,
    ) -> None:
        """Save the DataFrame to a csv file.

        Parameters
        ----------
        buffer : List[Dict[str, Any]]
            The data to save.
        columns : List[str]
            The column names of the data.
        save_path : str
            The path to save the data.
        flush : bool, optional
            Whether to flush the buffer, by default False

        """
        if len(buffer) >= self.buffer_size or (flush and buffer):
            dataframe = pd.DataFrame(buffer, columns=columns)
            dataframe.to_csv(
                save_path,
                mode="a",
                header=(not os.path.exists(save_path)),
                index=False,
            )
            buffer.clear()

    def execute_query(
        self,
        table_name: str,
        patient_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a query on the database.

        Parameters
        ----------
        table_name : str
            The name of the table to query.
        patient_id : Optional[str], optional
            The patient ID to query, by default None

        Returns
        -------
        List[Dict[str, Any]]
            The results of the query.

        """
        table = Table(
            table_name,
            self.metadata,
            autoload_with=self.engine,
            schema=self.schema,
        )
        query = select(table.c.fhir)
        if patient_id:
            query = query.where(table.c.patient_id == patient_id)
        with self.engine.connect() as connection:
            results = connection.execute(query).fetchall()
        return [result[0] for result in results]

    def get_patient_data(self) -> None:
        """Get patient data from the database and save to a csv file."""
        save_path = os.path.join(
            self.csv_dir,
            DATA_COLLECTION_CONFIG[PATIENT]["save_path"],
        )
        buffer = []
        results = self.execute_query(DATA_COLLECTION_CONFIG[PATIENT]["table_name"])
        LOGGER.info("Fetching patient data ...")
        for p in tqdm(results, desc="Processing patients", unit="patients"):
            patient = Patient(p)
            patient_data = {
                "patient_id": patient.id,
                "birthDate": patient.birthDate.isostring if patient.birthDate else None,
                "gender": patient.gender,
                "deceasedBoolean": patient.deceasedBoolean,
                "deceasedDateTime": patient.deceasedDateTime.isostring
                if patient.deceasedDateTime
                else None,
            }
            buffer.append(patient_data)
            self.save_to_csv(
                buffer,
                DATA_COLLECTION_CONFIG[PATIENT]["columns"],
                save_path,
            )
        self.save_to_csv(
            buffer,
            DATA_COLLECTION_CONFIG[PATIENT]["columns"],
            save_path,
            flush=True,
        )

    def get_encounter_data(self) -> None:
        """Get encounter data from the database and save to a csv file."""
        try:
            patients = pd.read_csv(os.path.join(self.csv_dir, "patients.csv"))
        except FileNotFoundError:
            print("Patients file not found. Please run get_patient_data() first.")
            return
        save_path = os.path.join(
            self.csv_dir,
            DATA_COLLECTION_CONFIG[ENCOUNTER]["save_path"],
        )
        buffer = []
        outpatient_ids = []
        LOGGER.info("Fetching encounter data ...")
        for _, patient_id in tqdm(
            patients["patient_id"].items(),
            desc="Processing patients",
            unit="patients",
        ):
            results = self.execute_query(
                DATA_COLLECTION_CONFIG[ENCOUNTER]["table_name"],
                patient_id,
            )
            if len(results) == 0:
                outpatient_ids.append(patient_id)
                continue
            starts = []
            ends = []
            ids = []
            for row in results:
                enc = Encounter(row)
                starts.append(enc.period.start.isostring)
                ends.append(enc.period.end.isostring)
                ids.append(enc.id)
            assert (
                len(starts)
                == len(
                    ends,
                )
            ), f"Length of starts and ends should be equal. {len(starts)} != {len(ends)}"
            e_data = {
                "patient_id": patient_id,
                "length": len(starts),
                "encounter_ids": ids,
                "starts": starts,
                "ends": ends,
            }
            buffer.append(e_data)
            self.save_to_csv(
                buffer,
                DATA_COLLECTION_CONFIG[ENCOUNTER]["columns"],
                save_path,
            )
        self.save_to_csv(
            buffer,
            DATA_COLLECTION_CONFIG[ENCOUNTER]["columns"],
            save_path,
            flush=True,
        )
        patients = patients[~patients["patient_id"].isin(outpatient_ids)]
        patients.to_csv(os.path.join(self.csv_dir, "inpatient.csv"), index=False)

    def get_procedure_data(self) -> None:
        """Get procedure data from the database and save to a csv file."""
        try:
            patients = pd.read_csv(os.path.join(self.csv_dir, "inpatient.csv"))
        except FileNotFoundError:
            print(
                "Encounters (inpatient) file not found. Please run get_encounter_data() first.",
            )
            return
        save_path = os.path.join(
            self.csv_dir,
            DATA_COLLECTION_CONFIG[PROCEDURE]["save_path"],
        )
        procedure_vocab = set()
        buffer = []
        LOGGER.info("Fetching procedure data ...")
        for _, patient_id in tqdm(
            patients["patient_id"].items(),
            desc="Processing patients",
            unit="patients",
        ):
            results = self.execute_query("procedure", patient_id)
            proc_codes = []
            proc_dates = []
            encounters = []
            for row in results:
                proc = Procedure(row)
                if proc.encounter is None or proc.code is None:
                    continue
                if proc.performedPeriod is None and proc.performedDateTime is None:
                    continue
                if proc.performedPeriod is None:
                    proc_date = proc.performedDateTime.isostring
                elif proc.performedDateTime is None:
                    proc_date = proc.performedPeriod.start.isostring
                proc_codes.append(proc.code.coding[0].code)
                proc_dates.append(proc_date)
                encounters.append(proc.encounter.reference.split("/")[-1])
                procedure_vocab.add(proc.code.coding[0].code)
            assert len(proc_codes) == len(
                proc_dates,
            ), f"Length of proc_codes and proc_dates should be equal. \
                    {len(proc_codes)} != {len(proc_dates)}"
            m_data = {
                "patient_id": patient_id,
                "length": len(proc_codes),
                "proc_codes": proc_codes,
                "proc_dates": proc_dates,
                "encounter_ids": encounters,
            }
            buffer.append(m_data)
            self.save_to_csv(
                buffer,
                DATA_COLLECTION_CONFIG[PROCEDURE]["columns"],
                save_path,
            )
        self.save_to_csv(
            buffer,
            DATA_COLLECTION_CONFIG[PROCEDURE]["columns"],
            save_path,
            flush=True,
        )
        with open(os.path.join(self.vocab_dir, "procedure_vocab.json"), "w") as f:
            json.dump(list(procedure_vocab), f)

    def get_medication_data(self) -> None:
        """Get medication data from the database and save to a csv file."""
        try:
            patients = pd.read_csv(os.path.join(self.csv_dir, "inpatient.csv"))
        except FileNotFoundError:
            print("Patients file not found. Please run get_encounter_data() first.")
            return
        medication_table = Table(
            "medication",
            self.metadata,
            autoload_with=self.engine,
            schema=self.schema,
        )
        save_path = os.path.join(self.csv_dir, "med_requests.csv")
        med_vocab = set()
        buffer = []
        LOGGER.info("Fetching medication data ...")
        with self.engine.connect() as connection:
            for _, patient_id in tqdm(
                patients["patient_id"].items(),
                desc="Processing patients",
                unit="patients",
            ):
                results = self.execute_query(
                    DATA_COLLECTION_CONFIG[MEDICATION]["table_name"],
                    patient_id,
                )
                med_codes = []
                med_dates = []
                encounters = []
                for row in results:
                    if "medicationCodeableConcept" in row:
                        # Includes messy text data, so we skip it
                        # should not be of type list
                        continue
                    med_req = MedicationRequest(row)
                    if med_req.authoredOn is None or med_req.encounter is None:
                        continue
                    med_query = select(medication_table.c.fhir).where(
                        medication_table.c.id
                        == med_req.medicationReference.reference.split("/")[-1],
                    )
                    med_result = connection.execute(med_query).fetchone()
                    med_result = Medication(med_result[0]) if med_result else None
                    if med_result is not None:
                        code = med_result.code.coding[0].code
                        if not code.isdigit():
                            continue
                        med_vocab.add(code)
                        med_codes.append(code)
                        med_dates.append(med_req.authoredOn.isostring)
                        encounters.append(
                            med_req.encounter.reference.split("/")[-1],
                        )
                assert len(med_codes) == len(
                    med_dates,
                ), f"Length of med_codes and med_dates should be equal. \
                        {len(med_codes)} != {len(med_dates)}"
                m_data = {
                    "patient_id": patient_id,
                    "length": len(med_codes),
                    "med_codes": med_codes,
                    "med_dates": med_dates,
                    "encounter_ids": encounters,
                }
                buffer.append(m_data)
                self.save_to_csv(
                    buffer,
                    DATA_COLLECTION_CONFIG[MEDICATION]["columns"],
                    save_path,
                )
        self.save_to_csv(
            buffer,
            DATA_COLLECTION_CONFIG[MEDICATION]["columns"],
            save_path,
            flush=True,
        )
        with open(os.path.join(self.vocab_dir, "med_vocab.json"), "w") as f:
            json.dump(list(med_vocab), f)

    def get_lab_data(self) -> None:
        """Get lab data from the database and save to a csv file."""
        try:
            patients = pd.read_csv(os.path.join(self.csv_dir, "inpatient.csv"))
        except FileNotFoundError:
            print("Patients file not found. Please run get_encounter_data() first.")
            return
        save_path = os.path.join(self.csv_dir, "labs.csv")
        lab_vocab = set()
        all_units = {}
        buffer = []
        LOGGER.info("Fetching lab data ...")
        for _, patient_id in tqdm(
            patients["patient_id"].items(),
            desc="Processing patients",
            unit="patients",
        ):
            results = self.execute_query(
                DATA_COLLECTION_CONFIG[LAB]["table_name"],
                patient_id,
            )
            lab_codes = []
            lab_values = []
            lab_units = []
            lab_dates = []
            encounters = []
            for row in results:
                event = Observation(row)
                if (
                    event.encounter is None
                    or event.effectiveDateTime is None
                    or event.code is None
                    or event.valueQuantity is None
                ):
                    continue
                code = event.code.coding[0].code
                lab_codes.append(code)
                lab_vocab.add(code)
                lab_dates.append(event.effectiveDateTime.isostring)
                lab_values.append(event.valueQuantity.value)
                lab_units.append(event.valueQuantity.unit)
                encounters.append(event.encounter.reference.split("/")[-1])
                if code not in all_units:
                    all_units[code] = {event.valueQuantity.unit}
                else:
                    all_units[code].add(event.valueQuantity.unit)
            assert (
                len(lab_codes) == len(lab_values) == len(lab_dates)
            ), f"Length of lab_codes, lab_values and lab_dates should be equal. \
                    {len(lab_codes)} != {len(lab_values)} != {len(lab_dates)}"
            m_data = {
                "patient_id": patient_id,
                "length": len(lab_codes),
                "lab_codes": lab_codes,
                "lab_values": lab_values,
                "lab_units": lab_units,
                "lab_dates": lab_dates,
                "encounter_ids": encounters,
            }
            buffer.append(m_data)
            self.save_to_csv(buffer, DATA_COLLECTION_CONFIG[LAB]["columns"], save_path)
        self.save_to_csv(
            buffer,
            DATA_COLLECTION_CONFIG[LAB]["columns"],
            save_path,
            flush=True,
        )
        with open(os.path.join(self.vocab_dir, "lab_vocab.json"), "w") as f:
            json.dump(list(lab_vocab), f)
        all_units = {k: list(v) for k, v in all_units.items()}
        with open(os.path.join(self.vocab_dir, "lab_units.json"), "w") as f:
            json.dump(all_units, f)

    def filter_lab_data(
        self,
    ) -> None:
        """Filter out lab codes that have more than one units."""
        try:
            labs = pd.read_csv(os.path.join(self.csv_dir, "labs.csv"))
            with open(os.path.join(self.vocab_dir, "lab_vocab.json"), "r") as f:
                lab_vocab = json.load(f)
            with open(os.path.join(self.vocab_dir, "lab_units.json"), "r") as f:
                lab_units = json.load(f)
        except FileNotFoundError:
            print("Labs file not found. Please run get_lab_data() first.")
            return
        LOGGER.info("Filtering lab data ...")
        for code, units in lab_units.items():
            if len(units) > 1:
                lab_vocab.remove(code)
        labs = labs.apply(lambda x: filter_lab_codes(x, lab_vocab), axis=1)
        labs.to_csv(os.path.join(self.csv_dir, "filtered_labs.csv"), index=False)
        with open(os.path.join(self.vocab_dir, "lab_vocab.json"), "w") as f:
            json.dump(list(lab_vocab), f)

    def process_lab_values(self, num_bins: int = 5) -> None:
        """Bin lab values into discrete values.

        Parameters
        ----------
        num_bins : int, optional
            number of bins, by default 5

        """
        try:
            labs = pd.read_csv(os.path.join(self.csv_dir, "filtered_labs.csv"))
            with open(os.path.join(self.vocab_dir, "lab_vocab.json"), "r") as f:
                lab_vocab = json.load(f)
        except FileNotFoundError:
            print("Labs file not found. Please run get_lab_data() first.")
            return

        def apply_eval(row: pd.Series) -> pd.Series:
            for col in ["lab_codes", "lab_values"]:
                row[col] = literal_eval(row[col])
            return row

        def assign_to_quantile_bins(row: pd.Series) -> pd.Series:
            if row["length"] == 0:
                row["binned_values"] = []
                return row
            binned_values = []
            for value, code in zip(row["lab_values"], row["lab_codes"]):
                bin_index = np.digitize(value, quantile_bins[code].right, right=False)
                binned_values.append(bin_index)
            row["binned_values"] = binned_values
            return row

        LOGGER.info("Processing lab values ...")
        labs = labs.apply(apply_eval, axis=1)
        quantile_bins = {}
        for code in lab_vocab:
            all_values = [
                value
                for sublist, sublist_codes in zip(labs["lab_values"], labs["lab_codes"])
                for value, sublist_code in zip(sublist, sublist_codes)
                if sublist_code == code
            ]

            quantile_bins[code] = pd.qcut(
                all_values,
                q=num_bins,
                duplicates="drop",
            ).categories

        labs = labs.apply(assign_to_quantile_bins, axis=1)
        labs.to_csv(os.path.join(self.csv_dir, "processed_labs.csv"), index=False)

        lab_vocab_binned = []
        lab_vocab_binned.extend(
            [f"{code}_{i}" for code in lab_vocab for i in range(num_bins)],
        )
        with open(os.path.join(self.vocab_dir + "lab_vocab.json"), "w") as f:
            json.dump(lab_vocab_binned, f)

    def get_condition_data(self) -> None:
        """Get condition data from the database and save to a csv file."""
        try:
            patients = pd.read_csv(os.path.join(self.csv_dir, "inpatient.csv"))
        except FileNotFoundError:
            print("Patients file not found. Please run get_encounter_data() first.")
            return
        save_path = os.path.join(self.csv_dir, "conditions.csv")
        condition_vocab = set()
        condition_counts = {}
        condition_systems = {}
        buffer = []
        LOGGER.info("Fetching condition data ...")
        for _, patient_id in tqdm(
            patients["patient_id"].items(),
            desc="Processing patients",
            unit="patients",
        ):
            patient_conditions_counted = set()
            results = self.execute_query(
                DATA_COLLECTION_CONFIG[CONDITION]["table_name"],
                patient_id,
            )
            encounter_conditions = {}
            for row in results:
                cond = Condition(row)
                if cond.encounter is None or cond.code is None:
                    continue
                encounter_id = cond.encounter.reference.split("/")[-1]
                code = cond.code.coding[0].code
                if encounter_id not in encounter_conditions:
                    encounter_conditions[encounter_id] = []
                encounter_conditions[encounter_id].append(code)
                display = cond.code.coding[0].display
                condition_vocab.add(code)
                if code not in condition_systems:
                    condition_systems[code] = cond.code.coding[0].system
                if code not in patient_conditions_counted:
                    if code in condition_counts:
                        condition_counts[code]["count"] += 1
                    else:
                        condition_counts[code] = {"count": 1, "display": display}
                    patient_conditions_counted.add(code)
            m_data = {
                "patient_id": patient_id,
                "length": sum(len(value) for value in encounter_conditions.values()),
                "encounter_conditions": encounter_conditions,
            }
            buffer.append(m_data)
            self.save_to_csv(
                buffer,
                DATA_COLLECTION_CONFIG[CONDITION]["columns"],
                save_path,
            )
        self.save_to_csv(
            buffer,
            DATA_COLLECTION_CONFIG[CONDITION]["columns"],
            save_path,
            flush=True,
        )
        with open(os.path.join(self.vocab_dir, "condition_vocab.json"), "w") as f:
            json.dump(list(condition_vocab), f)
        sorted_conditions = sorted(
            condition_counts.items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        )
        sorted_dict = dict(sorted_conditions)
        with open(os.path.join(self.vocab_dir, "condition_counts.json"), "w") as f:
            json.dump(sorted_dict, f)
        with open(os.path.join(self.vocab_dir, "condition_systems.json"), "w") as f:
            json.dump(condition_systems, f)

    def group_conditions(self) -> None:
        """Group conditions into categories."""
        with open(os.path.join(self.vocab_dir, "condition_counts.json"), "r") as file:
            data = json.load(file)
        with open(os.path.join(self.vocab_dir, "condition_systems.json"), "r") as file:
            systems = json.load(file)
        LOGGER.info("Grouping conditions ...")
        grouped_data = {}
        for code, info in data.items():
            prefix = code[:3]
            icd_version = systems[code][-1]
            group_key = f"{prefix}_10" if icd_version == "0" else f"{prefix}_9"
            if group_key not in grouped_data:
                grouped_data[group_key] = {"total_count": 0}
            grouped_data[group_key]["total_count"] += info["count"]
            grouped_data[group_key][f"{code}_count"] = info["count"]
        sorted_grouped_data = dict(
            sorted(
                grouped_data.items(),
                key=lambda x: x[1]["total_count"],
                reverse=True,
            ),
        )
        with open(
            os.path.join(self.vocab_dir, "condition_categories.json"),
            "w",
        ) as file:
            json.dump(sorted_grouped_data, file, indent=4)


if __name__ == "__main__":
    collector = FHIRDataCollector(
        db_path="postgresql://postgres:pwd@localhost:5432/mimiciv-2.0",
        schema="mimic_fhir",
        save_dir="/mnt/data/odyssey/mimiciv_fhir1",
        buffer_size=10000,
    )
    collector.get_patient_data()
    collector.get_encounter_data()
    collector.get_procedure_data()
    collector.get_medication_data()
    collector.get_lab_data()
    collector.filter_lab_data()
    collector.process_lab_values()
    collector.get_condition_data()
    collector.group_conditions()
