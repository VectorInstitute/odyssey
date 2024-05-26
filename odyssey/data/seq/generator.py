"""Create patient sequences from the events dataframes."""

import json
import logging
import math
import os
import time
from ast import literal_eval
from datetime import datetime
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from dateutil import parser

from odyssey.data.constants import (
    CLASS,
    LAB,
    MED,
    PAD,
    PROC,
    REGISTER,
    TIME_DELTA,
    VISIT_END,
    VISIT_START,
)
from odyssey.data.seq._data_containers import EventData, PatientData
from odyssey.data.seq._encounters import EncounterProcessor
from odyssey.odyssey.data.seq._events import EventProcessor
from odyssey.odyssey.data.seq._tokens import TokenConfig, TokenGenerator
from odyssey.utils.log import setup_logging


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


def _to_list(input_value: Union[np.ndarray, List]) -> Any:
    """Convert the input value to a list if it is an instance of numpy array.

    Parameters
    ----------
    input_value : Union[np.ndarray, List]
        Input value.

    Returns
    -------
    Any
        Converted value

    """
    return input_value.tolist() if isinstance(input_value, np.ndarray) else input_value


class SequenceSaver:
    """Save patient sequences to disk."""

    def __init__(self, save_dir):
        """Initialize the sequence saver."""
        self.save_dir = save_dir

    def save_sequences(self, sequences, round_number):
        """Save the patient sequences to disk."""
        pass


class PatientSequenceGenerator:
    """Generate patient sequences from the events dataframes."""

    def __init__(
        self,
        max_seq_length: int,
        data_dir: str = "data_files",
        json_dir: str = "json_files",
        save_dir: str = "data_files",
    ):
        self.max_seq_length = max_seq_length
        self.token_config = TokenConfig()
        self.token_generator = TokenGenerator(
            max_seq_length, self.token_config, reference_time="2020-01-01 00:00:00"
        )
        self.data_dir = data_dir
        self.json_dir = json_dir
        self.max_dir = os.path.join(save_dir, str(self.max_seq_length))
        self.all_dir = os.path.join(save_dir, "all")

        self.encounter_processor = EncounterProcessor()
        self.event_processor = EventProcessor(token_config=self.token_config)
        self.after_death_events: List[str] = []

        os.makedirs(self.max_dir, exist_ok=True)
        os.makedirs(self.all_dir, exist_ok=True)

    @property
    def get_max_column_names(self) -> List[str]:
        """Get the column names for the max sequence length."""
        return [
            "patient_id",
            "num_visits",
            "deceased",
            "death_after_start",
            "death_after_end",
            "length",
            "token_length",
            f"event_tokens_{self.max_seq_length}",
            f"type_tokens_{self.max_seq_length}",
            f"age_tokens_{self.max_seq_length}",
            f"time_tokens_{self.max_seq_length}",
            f"visit_tokens_{self.max_seq_length}",
            f"position_tokens_{self.max_seq_length}",
            f"elapsed_tokens_{self.max_seq_length}",
            "common_conditions",
            "rare_conditions",
        ]

    @property
    def get_all_column_names(self) -> List[str]:
        """Get the column names for all sequence lengths."""
        return [
            "patient_id",
            "num_visits",
            "deceased",
            "death_after_start",
            "death_after_end",
            "length",
            "token_length",
            "event_tokens",
            "type_tokens",
            "age_tokens",
            "time_tokens",
            "visit_tokens",
            "position_tokens",
            "elapsed_tokens",
            "common_conditions",
            "rare_conditions",
        ]

    def _add_tokens(
        self,
        row: pd.Series,
        encounter_row: pd.Series,
    ) -> pd.Series:
        """Add tokens to the events.

        Parameters
        ----------
        row : pd.Series
            Combined events row
        encounter_row : pd.Series
            Encounter row

        Returns
        -------
        pd.Series
            Updated row with tokens
        """
        events = row["proc_codes"]
        if len(events) == 0:
            row["event_tokens"] = []
            row["age_tokens"] = []
            row["time_tokens"] = []
            row["visit_tokens"] = []
            row["type_ids"] = []
            row["position_tokens"] = []
            row["num_visits"] = 0
            return row

        events_encounters = row["encounter_ids"]
        events_types = row["type_ids"]
        events_elapsed_time = row["elapsed_time"]
        ecounters = encounter_row["encounter_ids"]
        intervals = encounter_row["intervals"]
        eq_encounters = encounter_row["eq_encounters"]
        age_mapping = dict(zip(ecounters, encounter_row["ages"]))
        time_mapping = dict(zip(ecounters, encounter_row["times"]))

        event_tokens = [self.token_config.class_token]
        type_tokens = [self.token_config.token_type_mapping.get(CLASS)]
        age_tokens = [0]
        time_tokens = [0]
        visit_segments = [0]
        position_tokens = [0]
        elapsed_tokens = [-2]

        segment_value = 1
        position_value = 0

        prev_encounter = None

        for event, event_encounter, event_type, elapsed_time in zip(
            events,
            events_encounters,
            events_types,
            events_elapsed_time,
        ):
            is_different_encounter = event_encounter != prev_encounter
            has_no_equal = event_encounter not in eq_encounters
            is_not_equal = prev_encounter not in eq_encounters.get(event_encounter, [])

            if is_different_encounter and (has_no_equal or is_not_equal):
                if prev_encounter is not None:
                    # Adding Visit End Token
                    event_tokens.append(self.token_config.visit_end_token)
                    type_tokens.append(
                        self.token_config.token_type_mapping.get(VISIT_END)
                    )
                    age_tokens.append(age_mapping[prev_encounter])
                    time_tokens.append(time_mapping[prev_encounter])
                    visit_segments.append(segment_value)
                    position_tokens.append(position_value)
                    elapsed_tokens.append(-2)

                    # Adding Register Token
                    event_tokens.append(self.token_config.register_token)
                    type_tokens.append(
                        self.token_config.token_type_mapping.get(REGISTER)
                    )
                    age_tokens.append(age_mapping[prev_encounter])
                    time_tokens.append(time_mapping[prev_encounter])
                    visit_segments.append(segment_value)
                    position_tokens.append(position_value)
                    elapsed_tokens.append(-2)

                    # Adding interval token
                    event_tokens.append(intervals[event_encounter])
                    type_tokens.append(
                        self.token_config.token_type_mapping.get(TIME_DELTA)
                    )
                    age_tokens.append(0)
                    time_tokens.append(0)
                    visit_segments.append(0)
                    position_tokens.append(position_value)
                    elapsed_tokens.append(-2)

                # Adding Visit Start Token
                event_tokens.append(self.token_config.visit_start_token)
                type_tokens.append(
                    self.token_config.token_type_mapping.get(VISIT_START)
                )
                age_tokens.append(age_mapping[event_encounter])
                time_tokens.append(time_mapping[event_encounter])
                elapsed_tokens.append(-1)

                segment_value = 1 if segment_value == 2 else 2
                visit_segments.append(segment_value)

                if len(event_tokens) == 1 or event_tokens[-2] != "W0":
                    position_value = position_value + 1
                position_tokens.append(position_value)

            # Adding intermediate tokens
            event_tokens.append(event)
            type_tokens.append(event_type)
            age_tokens.append(age_mapping[event_encounter])
            time_tokens.append(time_mapping[event_encounter])
            visit_segments.append(segment_value)
            position_tokens.append(position_value)
            elapsed_tokens.append(elapsed_time)
            prev_encounter = event_encounter

        # Adding Visit End Token
        event_tokens.append(self.token_config.visit_end_token)
        type_tokens.append(self.token_config.token_type_mapping.get(VISIT_END))
        age_tokens.append(age_mapping[event_encounter])
        time_tokens.append(time_mapping[event_encounter])
        visit_segments.append(segment_value)
        position_tokens.append(position_value)
        elapsed_tokens.append(-2)

        # Adding Register Token
        event_tokens.append(self.token_config.register_token)
        type_tokens.append(self.token_config.token_type_mapping.get(REGISTER))
        age_tokens.append(age_mapping[event_encounter])
        time_tokens.append(time_mapping[event_encounter])
        visit_segments.append(segment_value)
        position_tokens.append(position_value)
        elapsed_tokens.append(-2)

        assert (
            len(event_tokens)
            == len(type_tokens)
            == len(age_tokens)
            == len(time_tokens)
            == len(visit_segments)
            == len(position_tokens)
            == len(elapsed_tokens)
        )

        row["token_length"] = len(event_tokens)
        row["event_tokens"] = event_tokens
        row["type_tokens"] = type_tokens
        row["age_tokens"] = age_tokens
        row["time_tokens"] = time_tokens
        row["visit_tokens"] = visit_segments
        row["position_tokens"] = position_tokens
        row["elapsed_tokens"] = elapsed_tokens
        row["num_visits"] = len(set(position_tokens)) - 1
        return row

    def _get_mortality_label(
        self,
        row: pd.Series,
        patient_row: pd.Series,
        encounter_row: pd.Series,
    ) -> pd.Series:
        """Get the mortality label for the patient.

        Parameters
        ----------
        row : pd.Series
            Combined events row
        patient_row : pd.Series
            Patient row
        encounter_row : pd.Series
            Encounter row

        Returns
        -------
        pd.Series
            Updated row with mortality label
        """
        death_date = patient_row["deceasedDateTime"]
        if not isinstance(death_date, str) and math.isnan(float(death_date)):
            row["deceased"] = 0
            return row
        death_date = datetime.strptime(death_date, "%Y-%m-%d").date()
        encounter_starts = encounter_row["starts"]
        encounter_ends = encounter_row["ends"]
        last_start = parser.parse(encounter_starts[-1]).replace(tzinfo=None).date()
        last_end = parser.parse(encounter_ends[-1]).replace(tzinfo=None).date()
        last_code_date = parser.parse(row["proc_dates"][-1]).replace(tzinfo=None).date()

        death_start = death_date - last_start
        death_end = death_date - last_end
        death_code = death_date - last_code_date

        if death_code.days < 0:
            LOGGER.info(f"After death events {row['patient_id']}")
            self.after_death_events.append(row["patient_id"])

        row["deceased"] = 1
        row["death_after_start"] = death_start.days
        row["death_after_end"] = death_end.days
        return row

    def _get_condition_label(
        self,
        row: pd.Series,
        condition_row: pd.Series,
    ) -> pd.Series:
        """Get the condition labels for the patient."""
        encounter_ids = set(row["encounter_ids"])
        common_file = os.path.join(self.json_dir, "common_conditions.json")
        rare_file = os.path.join(self.json_dir, "rare_conditions.json")

        with open(common_file, "r") as f:
            common_conditions = json.load(f)
        with open(rare_file, "r") as f:
            rare_conditions = json.load(f)

        conditions = condition_row["encounter_conditions"]
        valid_conditions = {k: v for k, v in conditions.items() if k in encounter_ids}
        patient_conditions = [
            item for sublist in valid_conditions.values() for item in sublist
        ]

        common_labels = []
        for _, codes in common_conditions.items():
            condition_found = set(codes).intersection(set(patient_conditions))
            if condition_found:
                common_labels.append(1)
            else:
                common_labels.append(0)

        rare_labels = []
        for _, codes in rare_conditions.items():
            condition_found = set(codes).intersection(set(patient_conditions))
            if condition_found:
                rare_labels.append(1)
            else:
                rare_labels.append(0)

        row["common_conditions"] = common_labels
        row["rare_conditions"] = rare_labels
        return row

    def _create_postition_tokens(self, event_tokens: List[str]) -> List[int]:
        """Create position tokens."""
        position_tokens = []
        token = 0
        for i in range(len(event_tokens)):
            if event_tokens[i] == self.token_config.visit_start_token:
                token = token + 1
            position_tokens.append(token)
        return position_tokens

    def _create_visit_tokens(self, event_tokens: List[str]) -> List[int]:
        """Create visit tokens."""
        visit_segments = [0]
        segment = 1
        for i in range(1, len(event_tokens)):
            if event_tokens[i] == self.token_config.visit_start_token:
                segment = 1 if segment == 2 else 2
            visit_segments.append(segment)
        return visit_segments

    def _truncate_or_pad(
        self,
        row: pd.Series,
        pad_events: bool = False,
    ) -> pd.Series:
        """Truncate or pads the sequence to max_seq_length.

        Parameters
        ----------
        row : pd.Series
            row of the combined events dataframe

        Returns
        -------
        pd.Series
            Updated row with truncated or padded sequence
        """
        sequence = _to_list(row["event_tokens"])
        t_type = _to_list(row["type_tokens"])
        age = _to_list(row["age_tokens"])
        time = _to_list(row["time_tokens"])
        visit = _to_list(row["visit_tokens"])
        position = _to_list(row["position_tokens"])
        elapsed = _to_list(row["elapsed_tokens"])
        seq_length = row["token_length"]
        truncated = False
        padded_length = 0

        if seq_length == self.max_seq_length:
            row[f"event_tokens_{self.max_seq_length}"] = sequence
            row[f"type_tokens_{self.max_seq_length}"] = t_type
            row[f"age_tokens_{self.max_seq_length}"] = age
            row[f"time_tokens_{self.max_seq_length}"] = time
            row[f"visit_tokens_{self.max_seq_length}"] = visit
            row[f"position_tokens_{self.max_seq_length}"] = position
            row[f"elapsed_tokens_{self.max_seq_length}"] = elapsed
            return row

        if seq_length > self.max_seq_length:
            truncated = True
            # get the recent max_length tokens
            start_index = int(seq_length - self.max_seq_length) + 2
            end_index = int(seq_length)

            if sequence[start_index] == self.token_config.visit_end_token:
                start_index += 3
            elif sequence[start_index] == self.token_config.register_token:
                start_index += 2
            elif sequence[start_index].startswith(("[W_", "[M_", "[LT")):
                start_index += 1

            if sequence[start_index] != self.token_config.visit_start_token:
                new_sequence = [
                    self.token_config.class_token,
                    self.token_config.visit_start_token,
                ] + sequence[start_index:end_index]
                new_type = [
                    self.token_config.token_type_mapping.get(CLASS),
                    self.token_config.token_type_mapping.get(VISIT_START),
                ] + t_type[start_index:end_index]
                new_age = [0, age[start_index]] + age[start_index:end_index]
                new_time = [0, time[start_index]] + time[start_index:end_index]
                new_visit = self._create_visit_tokens(new_sequence)
                new_position = self._create_postition_tokens(new_sequence)
                new_elasped = [-2, -1] + elapsed[start_index:end_index]
            else:
                new_sequence = [self.token_config.class_token] + sequence[
                    start_index:end_index
                ]
                new_type = [self.token_config.token_type_mapping.get(CLASS)] + t_type[
                    start_index:end_index
                ]
                new_age = [0] + age[start_index:end_index]
                new_time = [0] + time[start_index:end_index]
                new_visit = self._create_visit_tokens(new_sequence)
                new_position = self._create_postition_tokens(new_sequence)
                new_elasped = [-2] + elapsed[start_index:end_index]

            row[f"event_tokens_{self.max_seq_length}"] = new_sequence
            row[f"type_tokens_{self.max_seq_length}"] = new_type
            row[f"age_tokens_{self.max_seq_length}"] = new_age
            row[f"time_tokens_{self.max_seq_length}"] = new_time
            row[f"visit_tokens_{self.max_seq_length}"] = new_visit
            row[f"position_tokens_{self.max_seq_length}"] = new_position
            row[f"elapsed_tokens_{self.max_seq_length}"] = new_elasped
            seq_length = len(new_sequence)

        if seq_length < self.max_seq_length:
            padded_length = int(max(0, self.max_seq_length - seq_length))
            if truncated:
                if pad_events:
                    row[f"event_tokens_{self.max_seq_length}"] = (
                        row[f"event_tokens_{self.max_seq_length}"]
                        + [self.token_config.pad_token] * padded_length
                    )
                else:
                    # padding will be done in the tokenizer
                    row[f"event_tokens_{self.max_seq_length}"] = row[
                        f"event_tokens_{self.max_seq_length}"
                    ]

                row[f"type_tokens_{self.max_seq_length}"] = (
                    row[f"type_tokens_{self.max_seq_length}"]
                    + [self.token_config.token_type_mapping.get(PAD)] * padded_length
                )
                row[f"age_tokens_{self.max_seq_length}"] = (
                    row[f"age_tokens_{self.max_seq_length}"] + [0] * padded_length
                )
                row[f"time_tokens_{self.max_seq_length}"] = (
                    row[f"time_tokens_{self.max_seq_length}"] + [0] * padded_length
                )
                row[f"visit_tokens_{self.max_seq_length}"] = (
                    row[f"visit_tokens_{self.max_seq_length}"] + [0] * padded_length
                )
                row[f"position_tokens_{self.max_seq_length}"] = (
                    row[f"position_tokens_{self.max_seq_length}"] + [0] * padded_length
                )
            else:
                if pad_events:
                    row[f"event_tokens_{self.max_seq_length}"] = (
                        sequence + [self.token_config.pad_token] * padded_length
                    )
                else:
                    # padding will be done in the tokenizer
                    row[f"event_tokens_{self.max_seq_length}"] = sequence

                row[f"type_tokens_{self.max_seq_length}"] = (
                    t_type
                    + [self.token_config.token_type_mapping.get(PAD)] * padded_length
                )
                row[f"age_tokens_{self.max_seq_length}"] = age + [0] * padded_length
                row[f"time_tokens_{self.max_seq_length}"] = time + [0] * padded_length
                row[f"visit_tokens_{self.max_seq_length}"] = visit + [0] * padded_length
                row[f"position_tokens_{self.max_seq_length}"] = (
                    position + [0] * padded_length
                )
                row[f"elapsed_tokens_{self.max_seq_length}"] = elapsed

        for key in [
            f"type_tokens_{self.max_seq_length}",
            f"age_tokens_{self.max_seq_length}",
            f"time_tokens_{self.max_seq_length}",
            f"visit_tokens_{self.max_seq_length}",
            f"position_tokens_{self.max_seq_length}",
        ]:
            assert (
                len(row[key]) == self.max_seq_length
            ), f"Length of {key} is {len(row[key])} and max_length is {self.max_seq_length}"
            row[key] = np.array(row[key])

        return row

    def _remove_patients_with_no_encounters(
        self,
        patient_data: PatientData,
    ) -> PatientData:
        """Filter patients with no encounters.

        Parameters
        ----------
        patient_data : PatientData
            Patient data

        Returns
        -------
        PatientData
            Updated patient data

        """
        valid_indices = patient_data.encounters[
            patient_data.encounters["length"] > 0
        ].index
        patient_data.encounters = patient_data.encounters.loc[
            valid_indices
        ].reset_index(drop=True)
        patient_data.patients = patient_data.patients.loc[valid_indices].reset_index(
            drop=True
        )
        patient_data.conditions = patient_data.conditions.loc[
            valid_indices
        ].reset_index(drop=True)
        for event_type, _ in patient_data.events.items():
            patient_data.events[event_type].data = (
                patient_data.events[event_type]
                .data.loc[valid_indices]
                .reset_index(drop=True)
            )

        return patient_data

    def _process_encounters(
        self,
        patient_data: PatientData,
    ) -> PatientData:
        """Process encounters, remove patient data with no encounters.

        Parameters
        ----------
        patient_data : PatientData
            Patient data

        Returns
        -------
        PatientData
            Updated patient data

        """
        patient_data.encounters = patient_data.encounters.apply(
            lambda row: self.encounter_processor.validate_encounters(
                row,
                PROC=patient_data.events[PROC].data.iloc[row.name],
                MED=patient_data.events[MED].data.iloc[row.name],
                LAB=patient_data.events[LAB].data.iloc[row.name],
            ),
            axis=1,
        )
        patient_data = self._remove_patients_with_no_encounters(patient_data)
        patient_data.encounters = patient_data.encounters.apply(
            lambda row: self.encounter_processor.sort_encounters(row),
            axis=1,
        )
        patient_data.encounters = patient_data.encounters.apply(
            lambda row: self.encounter_processor.calculate_patient_ages(
                row,
                patient_data.patients.iloc[row.name],
            ),
            axis=1,
        )
        patient_data.encounters = patient_data.encounters.apply(
            lambda row: self.encounter_processor.calculate_encounter_times(
                row,
                self.token_generator.reference_time,
            ),
            axis=1,
        )
        patient_data.encounters = patient_data.encounters.apply(
            lambda row: self.encounter_processor.calculate_intervals(row),
            axis=1,
        )

        return patient_data

    def _process_events(
        self,
        patient_data: PatientData,
    ) -> PatientData:
        """Process events, keep valid rows and edit the datetimes of the events.

        Parameters
        ----------
        patient_data : PatientData
            Patient data

        Returns
        -------
        PatientData
            Updated patient data

        """
        for event_type, _ in patient_data.events.items():
            patient_data.events[event_type].data = patient_data.events[
                event_type
            ].data.apply(
                lambda row: self.event_processor.edit_event_datetimes(
                    row,
                    patient_data.encounters.iloc[row.name],
                    event_type,
                ),
                axis=1,
            )

        return patient_data

    def create_patient_sequence(
        self,
        chunksize: Optional[int] = None,
        min_events: int = 0,
        min_visits: int = 0,
        pad_events: bool = False,
    ) -> None:
        """Create patient sequences and saves them as a parquet file."""
        file_paths = [
            f"{self.data_dir}/inpatient.csv",
            f"{self.data_dir}/encounters.csv",
            f"{self.data_dir}/procedures.csv",
            f"{self.data_dir}/med_requests.csv",
            f"{self.data_dir}/processed_labs.csv",
            f"{self.data_dir}/conditions.csv",
        ]
        rounds = 0
        more_chunks = True
        readers = [pd.read_csv(path, chunksize=chunksize) for path in file_paths]
        while more_chunks:
            try:
                dataframes = [next(reader).reset_index(drop=True) for reader in readers]
            except StopIteration:
                more_chunks = False
                break
            patients, encounters, procedures, medications, labs, conditions = dataframes
            patient_data = PatientData(
                patients=patients,
                encounters=encounters,
                conditions=conditions,
                events={
                    PROC: EventData(event_type=PROC, data=procedures),
                    MED: EventData(event_type=MED, data=medications),
                    LAB: EventData(event_type=LAB, data=labs),
                },
            )
            start_time = time.time()
            ## Process encounters.
            patient_data = self._process_encounters(patient_data)

            # Process events.
            patient_data = self._process_events(patient_data)

            # Conditions.
            patient_data.conditions["encounter_conditions"] = patient_data.conditions[
                "encounter_conditions"
            ].apply(literal_eval)

            # Combine events.
            combined_events = self.event_processor.combine_events(
                patient_data,
            )
            # Filter patients based on min_events.
            combined_events = combined_events[combined_events["length"] > min_events]
            # add elapsed time after admission for all events
            combined_events = combined_events.apply(
                lambda row: self.event_processor.calculate_time_after_admission(
                    row,
                    patient_data.encounters.iloc[row.name],
                ),
                axis=1,
            )
            # add special tokens to the events
            combined_events = combined_events.apply(
                lambda row: self._add_tokens(
                    row, patient_data.encounters.iloc[row.name]
                ),
                axis=1,
            )
            # filter patients based on min_visits
            combined_events = combined_events[
                combined_events["num_visits"] > min_visits
            ]
            # get mortality label
            combined_events = combined_events.apply(
                lambda row: self._get_mortality_label(
                    row,
                    patient_data.patients.iloc[row.name],
                    patient_data.encounters.iloc[row.name],
                ),
                axis=1,
            )
            combined_events = combined_events[
                ~combined_events["patient_id"].isin(self.after_death_events)
            ]
            combined_events = combined_events.apply(
                lambda row: self._truncate_or_pad(row, pad_events=pad_events),
                axis=1,
            )
            # get condition label for common and rare conditions
            combined_events = combined_events.apply(
                lambda row: self._get_condition_label(
                    row,
                    patient_data.conditions.iloc[row.name],
                ),
                axis=1,
            )
            # drop rows with nan values for events if any
            combined_events = combined_events.dropna(
                subset=[f"event_tokens_{self.max_seq_length}"],
            )
            # save the combined events
            combined_events_all = combined_events.loc[
                :, combined_events.columns.intersection(self.get_all_column_names)
            ]
            combined_events_max = combined_events.loc[
                :, combined_events.columns.intersection(self.get_max_column_names)
            ]

            combined_events_all.to_parquet(
                self.all_dir + f"/patient_sequences_{rounds}.parquet",
                engine="pyarrow",
            )
            combined_events_max.to_parquet(
                self.max_dir
                + f"/patient_sequences_{self.max_seq_length}_{rounds}.parquet",
                engine="pyarrow",
            )
            round_time = time.time() - start_time
            rounds += 1
            LOGGER.info(
                f"Round {rounds} done in {round_time:.2f} s, {chunksize * rounds} samples done."
            )

    def reapply_truncation(
        self,
        file_paths: Union[str, List[str]],
        pad_events: bool = False,
    ) -> None:
        """
        Reapply truncation to Parquet file(s).

        Parameters
        ----------
        file_paths : Union[str, List[str]]
            Path or list of paths to Parquet files to be processed.
        pad_events : bool
            Whether to pad the events or not, defaults to False

        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        for i, file_path in enumerate(sorted(file_paths)):
            df = pd.read_parquet(file_path)
            df = df.apply(
                lambda row: self._truncate_or_pad(row, pad_events=pad_events),
                axis=1,
            )
            df = df[self.get_max_column_names]
            base_name = f"patient_sequences_{self.max_seq_length}"
            suffix = f"_{i}" if len(file_paths) > 1 else ""
            file_name = f"{base_name}{suffix}.parquet"
            df.to_parquet(
                os.path.join(self.max_dir, file_name),
                engine="pyarrow",
            )


if __name__ == "__main__":
    generator = PatientSequenceGenerator(
        max_seq_length=2048,
        data_dir="/mnt/data/odyssey/mimiciv_fhir1/csv_files",
        json_dir="/mnt/data/odyssey/mimiciv_fhir1/vocab",
        save_dir="/mnt/data/odyssey/mimiciv_fhir1/parquet_files",
    )
    generator.create_patient_sequence(
        chunksize=10,
        min_events=10,
        min_visits=0,
    )
