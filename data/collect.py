"""Collect data from the FHIR database and save to csv files."""
import json
import os
from typing import List

import numpy as np
import pandas as pd
from fhir.resources.encounter import Encounter
from fhir.resources.medication import Medication
from fhir.resources.medicationrequest import MedicationRequest
from fhir.resources.observation import Observation
from fhir.resources.patient import Patient
from fhir.resources.procedure import Procedure
from sqlalchemy import MetaData, Table, create_engine, select
from tqdm import tqdm


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

    def get_patient_data(self) -> None:
        """Get patient data from the database and save to a csv file."""
        patients_table = Table(
            "patient",
            self.metadata,
            autoload_with=self.engine,
            schema=self.schema,
        )

        patient_cols = [
            "patient_id",
            "birthDate",
            "gender",
            "deceasedBoolean",
            "deceasedDateTime",
        ]
        save_path = os.path.join(self.csv_dir, "patients.csv")
        buffer = []

        with self.engine.connect() as connection:
            query = select(patients_table.c.fhir)
            results = connection.execute(query).fetchall()
            for p in tqdm(results, desc="Processing patients", unit="patient"):
                patient = Patient(p[0])
                patient_data = {
                    "patient_id": patient.id,
                    "birthDate": patient.birthDate.isostring
                    if patient.birthDate
                    else None,
                    "gender": patient.gender,
                    "deceasedBoolean": patient.deceasedBoolean,
                    "deceasedDateTime": patient.deceasedDateTime.isostring
                    if patient.deceasedDateTime
                    else None,
                }
                buffer.append(patient_data)
                if len(buffer) >= self.buffer_size:
                    df_buffer = pd.DataFrame(buffer, columns=patient_cols)
                    buffer = []
                    df_buffer.to_csv(
                        save_path,
                        mode="a",
                        header=(not os.path.exists(save_path)),
                        index=False,
                    )
            if buffer:
                df_buffer = pd.DataFrame(buffer, columns=patient_cols)
                df_buffer.to_csv(
                    save_path,
                    mode="a",
                    header=(not os.path.exists(save_path)),
                    index=False,
                )

    def get_encounter_data(self) -> None:
        """Get encounter data from the database and save to a csv file."""
        try:
            patients = pd.read_csv(self.csv_dir + "/patients.csv")
        except FileNotFoundError:
            print("Patients file not found. Please run get_patient_data() first.")
            return

        encounters_table = Table(
            "encounter",
            self.metadata,
            autoload_with=self.engine,
            schema=self.schema,
        )

        encounter_cols = ["patient_id", "length", "encounter_ids", "starts", "ends"]
        save_path = os.path.join(self.csv_dir, "encounters.csv")
        buffer = []
        outpatient_ids = []

        with self.engine.connect() as connection:
            for _, patient_id in tqdm(
                patients["patient_id"].items(),
                desc="Processing patients",
                unit="patient",
            ):
                query = select(encounters_table.c.fhir).where(
                    encounters_table.c.patient_id == patient_id,
                )
                results = connection.execute(query).fetchall()
                if len(results) == 0:
                    outpatient_ids.append(patient_id)
                    continue

                starts = []
                ends = []
                ids = []
                for row in results:
                    enc = Encounter(row[0])
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
                if len(buffer) >= self.buffer_size:
                    df_buffer = pd.DataFrame(buffer, columns=encounter_cols)
                    buffer = []
                    df_buffer.to_csv(
                        save_path,
                        mode="a",
                        header=(not os.path.exists(save_path)),
                        index=False,
                    )

            if buffer:
                df_buffer = pd.DataFrame(buffer, columns=encounter_cols)
                df_buffer.to_csv(
                    save_path,
                    mode="a",
                    header=(not os.path.exists(save_path)),
                    index=False,
                )

        patients = patients[~patients["patient_id"].isin(outpatient_ids)]
        patients.to_csv(self.csv_dir + "/inpatient.csv", index=False)

    def get_procedure_data(self) -> None:
        """Get procedure data from the database and save to a csv file."""
        try:
            patients = pd.read_csv(self.csv_dir + "/inpatient.csv")
        except FileNotFoundError:
            print("Patients file not found. Please run get_encounter_data() first.")
            return

        procedure_table = Table(
            "procedure",
            self.metadata,
            autoload_with=self.engine,
            schema=self.schema,
        )

        procedure_cols = [
            "patient_id",
            "length",
            "proc_codes",
            "proc_dates",
            "encounter_ids",
        ]
        save_path = os.path.join(self.csv_dir, "procedures.csv")
        procedure_vocab = set()
        buffer = []

        with self.engine.connect() as connection:
            for _, patient_id in tqdm(
                patients["patient_id"].items(),
                desc="Processing patients",
                unit="patient",
            ):
                query = select(procedure_table.c.fhir).where(
                    procedure_table.c.patient_id == patient_id,
                )

                results = connection.execute(query).fetchall()
                proc_codes = []
                proc_dates = []
                encounters = []

                for row in results:
                    proc = Procedure(row[0])
                    if (
                        proc.encounter is None
                        or proc.code is None
                        or proc.performedDateTime is None
                    ):
                        continue
                    proc_codes.append(proc.code.coding[0].code)
                    proc_dates.append(proc.performedDateTime.isostring)
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
                if len(buffer) >= self.buffer_size:
                    df_buffer = pd.DataFrame(buffer, columns=procedure_cols)
                    buffer = []
                    df_buffer.to_csv(
                        save_path,
                        mode="a",
                        header=(not os.path.exists(save_path)),
                        index=False,
                    )

            if buffer:
                df_buffer = pd.DataFrame(buffer, columns=procedure_cols)
                df_buffer.to_csv(
                    save_path,
                    mode="a",
                    header=(not os.path.exists(save_path)),
                    index=False,
                )

        with open(self.vocab_dir + "/procedure_vocab.json", "w") as f:
            json.dump(list(procedure_vocab), f)

    def get_medication_data(self) -> None:
        """Get medication data from the database and save to a csv file."""
        try:
            patients = pd.read_csv(self.csv_dir + "/inpatient.csv")
        except FileNotFoundError:
            print("Patients file not found. Please run get_encounter_data() first.")
            return

        med_request_table = Table(
            "medication_request",
            self.metadata,
            autoload_with=self.engine,
            schema=self.schema,
        )
        medication_table = Table(
            "medication",
            self.metadata,
            autoload_with=self.engine,
            schema=self.schema,
        )

        medication_cols = [
            "patient_id",
            "length",
            "med_codes",
            "med_dates",
            "encounter_ids",
        ]
        save_path = os.path.join(self.csv_dir, "med_requests.csv")
        med_vocab = set()
        buffer = []

        with self.engine.connect() as connection:
            for _, patient_id in tqdm(
                patients["patient_id"].items(),
                desc="Processing patients",
                unit="patient",
            ):
                query = select(med_request_table.c.fhir).where(
                    med_request_table.c.patient_id == patient_id,
                )
                results = connection.execute(query).fetchall()
                med_codes = []
                med_dates = []
                encounters = []
                for row in results:
                    data = row[0]
                    if "medicationCodeableConcept" in data:
                        # includes messy text data, so we skip it
                        # should not be of type list
                        continue
                    med_req = MedicationRequest(data)
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
                if len(buffer) >= self.buffer_size:
                    df_buffer = pd.DataFrame(buffer, columns=medication_cols)
                    buffer = []
                    df_buffer.to_csv(
                        save_path,
                        mode="a",
                        header=(not os.path.exists(save_path)),
                        index=False,
                    )

            if buffer:
                df_buffer = pd.DataFrame(buffer, columns=medication_cols)
                df_buffer.to_csv(
                    save_path,
                    mode="a",
                    header=(not os.path.exists(save_path)),
                    index=False,
                )
        with open(self.vocab_dir + "/med_vocab.json", "w") as f:
            json.dump(list(med_vocab), f)

    def get_lab_data(self) -> None:
        """Get lab data from the database and save to a csv file."""
        try:
            patients = pd.read_csv(self.csv_dir + "/inpatient.csv")
        except FileNotFoundError:
            print("Patients file not found. Please run get_encounter_data() first.")
            return

        lab_table = Table(
            "observation_labevents",
            self.metadata,
            autoload_with=self.engine,
            schema=self.schema,
        )

        lab_cols = [
            "patient_id",
            "length",
            "lab_codes",
            "lab_values",
            "lab_units",
            "lab_dates",
            "encounter_ids",
        ]
        save_path = os.path.join(self.csv_dir, "labs.csv")
        lab_vocab = set()
        all_units = {}
        buffer = []

        with self.engine.connect() as connection:
            for _, patient_id in tqdm(
                patients["patient_id"].items(),
                desc="Processing patients",
                unit="patient",
            ):
                query = select(lab_table.c.fhir).where(
                    lab_table.c.patient_id == patient_id,
                )

                results = connection.execute(query).fetchall()
                lab_codes = []
                lab_values = []
                lab_units = []
                lab_dates = []
                encounters = []

                for row in results:
                    event = Observation(row[0])
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
                        all_units[code] = set(event.valueQuantity.unit)
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
                if len(buffer) >= self.buffer_size:
                    df_buffer = pd.DataFrame(buffer, columns=lab_cols)
                    buffer = []
                    df_buffer.to_csv(
                        save_path,
                        mode="a",
                        header=(not os.path.exists(save_path)),
                        index=False,
                    )

            if buffer:
                df_buffer = pd.DataFrame(buffer, columns=lab_cols)
                df_buffer.to_csv(
                    save_path,
                    mode="a",
                    header=(not os.path.exists(save_path)),
                    index=False,
                )

        with open(self.vocab_dir + "/lab_vocab.json", "w") as f:
            json.dump(list(lab_vocab), f)

        all_units = {k: list(v) for k, v in all_units.items()}
        with open(self.vocab_dir + "/lab_units.json", "w") as f:
            json.dump(all_units, f)

    def filter_lab_data(
        self,
    ) -> None:
        """Filter out lab codes that have more than one units."""
        try:
            labs = pd.read_csv(self.csv_dir + "/labs.csv")
            with open(self.vocab_dir + "/lab_vocab.json", "r") as f:
                lab_vocab = json.load(f)
            with open(self.vocab_dir + "/lab_units.json", "r") as f:
                lab_units = json.load(f)
        except FileNotFoundError:
            print("Labs file not found. Please run get_lab_data() first.")
            return

        for code, units in lab_units.items():
            if len(units) > 1:
                lab_vocab.remove(code)

        def filter_codes(row: pd.Series, vocab: List[str]) -> pd.Series:
            for col in [
                "lab_codes",
                "lab_values",
                "lab_units",
                "lab_dates",
                "encounter_ids",
            ]:
                row[col] = eval(row[col])

            indices = [i for i, code in enumerate(row["lab_codes"]) if code in vocab]
            for col in [
                "lab_codes",
                "lab_values",
                "lab_units",
                "lab_dates",
                "encounter_ids",
            ]:
                row[col] = [row[col][i] for i in indices]

            row["length"] = len(row["lab_codes"])
            return row

        labs = labs.apply(lambda x: filter_codes(x, lab_vocab), axis=1)

        labs.to_csv(self.csv_dir + "/filtered_labs.csv", index=False)
        with open(self.vocab_dir + "/lab_vocab.json", "w") as f:
            json.dump(list(lab_vocab), f)

    def process_lab_values(self, num_bins: int = 5) -> None:
        """Bin lab values into discrete values.

        Parameters
        ----------
        num_bins : int, optional
            number of bins, by default 5
        """
        try:
            labs = pd.read_csv(self.csv_dir + "/filtered_labs.csv")
            with open(self.vocab_dir + "/lab_vocab.json", "r") as f:
                lab_vocab = json.load(f)
        except FileNotFoundError:
            print("Labs file not found. Please run get_lab_data() first.")
            return

        def apply_eval(row: pd.Series) -> pd.Series:
            for col in ["lab_codes", "lab_values"]:
                row[col] = eval(row[col])
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
        labs.to_csv(self.csv_dir + "/processed_labs.csv", index=False)

        lab_vocab_binned = []
        lab_vocab_binned.extend(
            [f"{code}_{i}" for code in lab_vocab for i in range(num_bins)],
        )
        with open(self.vocab_dir + "/lab_vocab.json", "w") as f:
            json.dump(lab_vocab_binned, f)


if __name__ == "__main__":
    collector = FHIRDataCollector(
        db_path="postgresql://postgres:pwd@localhost:5432/mimiciv-2.0",
        schema="mimic_fhir",
        save_dir="/mnt/data/odyssey/mimiciv_fhir",
        buffer_size=10000,
    )
    collector.get_patient_data()
    collector.get_encounter_data()
    collector.get_procedure_data()
    collector.get_medication_data()
    collector.get_lab_data()
    collector.filter_lab_data()
    collector.process_lab_values()
