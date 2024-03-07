"""Create patient sequences from the events dataframes."""
import os
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from dateutil import parser


class SequenceGenerator:
    """Generate patient sequences from the events dataframes."""

    def __init__(
        self,
        max_seq_length: int,
        pad_token: str = "[PAD]",
        mask_token: str = "[MASK]",
        start_token: str = "[VS]",
        end_token: str = "[VE]",
        class_token: str = "[CLS]",
        register_token: str = "[REG]",
        unknown_token: str = "[UNK]",
        reference_time: str = "2020-01-01 00:00:00",
        data_dir: str = "data_files",
        save_dir: str = "data_files",
    ):
        self.max_seq_length = max_seq_length
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.start_token = start_token
        self.end_token = end_token
        self.class_token = class_token
        self.register_token = register_token
        self.unknown_token = unknown_token
        self.reference_time = parser.parse(reference_time)
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.after_death_events: List[str] = []

        os.makedirs(save_dir, exist_ok=True)

    @property
    def time_delta_tokens(self) -> List[str]:
        """Get the time delta tokens."""
        return (
            [f"[W_{i}]" for i in range(0, 4)]
            + [f"[M_{i}]" for i in range(0, 13)]
            + ["[LT]"]
        )

    @property
    def special_tokens(self) -> List[str]:
        """Get the special tokens."""
        return [
            self.pad_token,
            self.mask_token,
            self.start_token,
            self.end_token,
            self.class_token,
            self.register_token,
            self.unknown_token,
        ] + self.time_delta_tokens

    @property
    def get_token_type_dict(self) -> Dict[str, int]:
        """Get the token type dictionary."""
        return {
            "pad": 0,
            "class": 1,
            "start": 2,
            "end": 3,
            "time": 4,
            "lab": 5,
            "med": 6,
            "proc": 7,
            "reg": 8,
        }

    @staticmethod
    def to_list(input_value: Union[np.ndarray, List]) -> List:
        """Convert the input value to a list if it is an instance of numpy array."""
        return (
            input_value.tolist() if isinstance(input_value, np.ndarray) else input_value
        )

    def _sort_encounters(self, encounter_row: pd.Series) -> pd.Series:
        """Sort the encounters by start time.

        Parameters
        ----------
        encounter_row : pd.Series
            Encounter row

        Returns
        -------
        pd.Series
            Sorted encounter row
        """
        for c_name in ["encounter_ids", "starts", "ends"]:
            encounter_row[c_name] = eval(encounter_row[c_name])
        if encounter_row["length"] == 0:
            return encounter_row
        starts = encounter_row["starts"]
        ends = encounter_row["ends"]
        ids = encounter_row["encounter_ids"]
        sorted_lists = sorted(zip(starts, ends, ids))
        starts, ends, ids = zip(*sorted_lists)
        encounter_row["starts"] = list(starts)
        encounter_row["ends"] = list(ends)
        encounter_row["encounter_ids"] = list(ids)
        return encounter_row

    def _get_encounters_age(
        self,
        encounter_row: pd.Series,
        patient_row: pd.Series,
    ) -> pd.Series:
        """Get the age of the patient at the time of the encounters.

        Parameters
        ----------
        encounter_row : pd.Series
            Encounter row
        patient_row : pd.Series
            Patient row

        Returns
        -------
        pd.Series
            Encounter row with ages
        """
        birth_date = patient_row["birthDate"]
        encounter_dates = encounter_row["starts"]
        birth_date = datetime.strptime(birth_date, "%Y-%m-%d")
        ages = [
            (parser.parse(e_date).replace(tzinfo=None) - birth_date).days // 365
            for e_date in encounter_dates
        ]
        encounter_row["ages"] = ages
        return encounter_row

    def _get_encounters_time(self, encounter_row: pd.Series) -> pd.Series:
        """Get the time of the encounters in weeks with respect to a reference time.

        Parameters
        ----------
        encounter_row : pd.Series
            Encounter row

        Returns
        -------
        pd.Series
            Updated row with encounter times
        """
        first_encounter = parser.parse(encounter_row["starts"][0]).replace(tzinfo=None)
        initial_value = (first_encounter - self.reference_time).days // 7
        ages = encounter_row["ages"]
        time_values = [initial_value + (age - ages[0]) * 53 for age in ages]
        encounter_row["times"] = time_values
        return encounter_row

    def _calculate_intervals(self, encounter_row: pd.Series) -> pd.Series:
        """Calculate the intervals between encounters.

        Parameters
        ----------
        encounter_row : pd.Series
            Encounter row

        Returns
        -------
        pd.Series
            Updated row with intervals
        """
        start_times = encounter_row["starts"]
        end_times = encounter_row["ends"]
        intervals: Dict[str, str] = {}
        eq_encounters: Dict[str, List[str]] = {}
        for i in range(len(start_times) - 1):
            start = parser.parse(start_times[i + 1])
            start_id = encounter_row["encounter_ids"][i + 1]

            end = parser.parse(end_times[i])
            end_id = encounter_row["encounter_ids"][i]

            delta = start - end
            days = delta.days

            # If the difference between the end of the current encounter
            # and the start of the next encounter is negative, we consider
            # them to be a single encounter
            if days < 0:
                if start_id not in eq_encounters:
                    eq_encounters[start_id] = []
                if end_id not in eq_encounters:
                    eq_encounters[end_id] = []

                eq_encounters[start_id].append(end_id)
                eq_encounters[end_id].append(start_id)
                continue

            days = abs(days)
            if days < 28:
                week_num = days // 7
                intervals[start_id] = f"[W_{week_num}]"
            elif 28 <= days <= 365:
                month_num = days // 30
                intervals[start_id] = f"[M_{month_num}]"
            else:
                intervals[start_id] = "[LT]"

        encounter_row["intervals"] = intervals
        encounter_row["eq_encounters"] = eq_encounters
        return encounter_row

    def _edit_datetimes(
        self,
        row: pd.Series,
        encounter_row: pd.Series,
        concept_name: str,
    ) -> pd.Series:
        """Edit the datetimes of the events.

        Done so that they won't fall out of the corresponding encounter time frame.

        Parameters
        ----------
        row : pd.Series
            Events row
        encounter_row : pd.Series
            Encounter row
        concept_name : str
            Name of the event concept

        Returns
        -------
        pd.Series
            Updated row with edited datetimes
        """
        if concept_name == "lab":
            for name in [
                "encounter_ids",
                "lab_dates",
                "lab_codes",
                "binned_values",
                "lab_values",
                "lab_units",
            ]:
                row[name] = eval(row[name])
        elif concept_name == "med":
            for name in ["encounter_ids", "med_dates", "med_codes"]:
                row[name] = eval(row[name])
        elif concept_name == "proc":
            for name in ["encounter_ids", "proc_dates", "proc_codes"]:
                row[name] = eval(row[name])
        if row["length"] == 0:
            return row
        dates = []
        date_column = f"{concept_name}_dates"
        for i, date in enumerate(row[date_column]):
            encounter_index = encounter_row["encounter_ids"].index(
                row["encounter_ids"][i],
            )
            encounter_start = encounter_row["starts"][encounter_index]
            encounter_end = encounter_row["ends"][encounter_index]
            start_parsed = parser.parse(encounter_start)
            end_parsed = parser.parse(encounter_end)
            date_parsed = parser.parse(date)

            enc_date = date
            if date_parsed < start_parsed:
                enc_date = encounter_start
            elif date_parsed > end_parsed:
                enc_date = encounter_end
            dates.append(enc_date)
        row[date_column] = dates
        return row

    def _concat_concepts(
        self,
        procedures: pd.DataFrame,
        medications: pd.DataFrame,
        labs: pd.DataFrame,
        encounters: pd.DataFrame,
    ) -> pd.DataFrame:
        """Concatenate the events of different concepts.

        Parameters
        ----------
        procedures : pd.DataFrame
            Procedures dataframe
        medications : pd.DataFrame
            Medications dataframe
        labs : pd.DataFrame
            _description_
        encounters : pd.DataFrame
            Encounters dataframe

        Returns
        -------
        pd.DataFrame
            Combined events dataframe
        """
        procedures["type_ids"] = [None] * len(procedures)
        for i in range(len(procedures)):
            proc_codes = procedures.iloc[i]["proc_codes"]
            proc_codes = [code.upper() for code in proc_codes]

            med_codes = medications.iloc[i]["med_codes"]
            med_codes = [code.upper() for code in med_codes]

            lab_codes = labs.iloc[i]["lab_codes"]
            lab_codes = [code.upper() for code in lab_codes]
            lab_bins = labs.iloc[i]["binned_values"]
            lab_codes = [f"{el1}_{el2}" for el1, el2 in zip(lab_codes, lab_bins)]

            proc_type_ids = [self.get_token_type_dict["proc"]] * len(proc_codes)
            med_type_ids = [self.get_token_type_dict["med"]] * len(med_codes)
            lab_type_ids = [self.get_token_type_dict["lab"]] * len(lab_codes)

            proc_dates = procedures.iloc[i]["proc_dates"]
            med_dates = medications.iloc[i]["med_dates"]
            lab_dates = labs.iloc[i]["lab_dates"]

            proc_encounters = procedures.iloc[i]["encounter_ids"]
            med_encounters = medications.iloc[i]["encounter_ids"]
            lab_encounters = labs.iloc[i]["encounter_ids"]

            combined_codes = proc_codes + med_codes + lab_codes
            combined_dates = proc_dates + med_dates + lab_dates
            combined_encounters = proc_encounters + med_encounters + lab_encounters
            combined_type_ids = proc_type_ids + med_type_ids + lab_type_ids

            encounter_order = {
                encounter: i
                for i, encounter in enumerate(encounters.iloc[i]["encounter_ids"])
            }

            combined = sorted(
                zip(
                    combined_dates,
                    combined_codes,
                    combined_encounters,
                    combined_type_ids,
                ),
                key=lambda x: (x[0], encounter_order.get(x[2])),
            )
            sorted_dates, sorted_codes, sorted_encounters, sorted_type_ids = (
                zip(*combined) if combined else ([], [], [], [])
            )
            procedures.at[i, "proc_dates"] = list(sorted_dates)
            procedures.at[i, "proc_codes"] = list(sorted_codes)
            procedures.at[i, "encounter_ids"] = list(sorted_encounters)
            procedures.at[i, "type_ids"] = list(sorted_type_ids)
            procedures.at[i, "length"] = len(sorted_dates)

        return procedures

    # pylint: disable=too-many-statements
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
            return row

        events_encounters = row["encounter_ids"]
        events_types = row["type_ids"]
        ecounters = encounter_row["encounter_ids"]
        intervals = encounter_row["intervals"]
        eq_encounters = encounter_row["eq_encounters"]
        age_mapping = dict(zip(ecounters, encounter_row["ages"]))
        time_mapping = dict(zip(ecounters, encounter_row["times"]))

        event_tokens = [self.class_token]
        type_tokens = [self.get_token_type_dict["class"]]
        age_tokens = [0]
        time_tokens = [0]
        visit_segments = [0]
        position_tokens = [0]

        segment_value = 1
        position_value = 1

        prev_encounter = None

        for event, event_encounter, event_type in zip(
            events,
            events_encounters,
            events_types,
        ):
            if (
                event_encounter != prev_encounter
                and event_encounter not in eq_encounters
            ):
                if prev_encounter is not None:
                    # Adding Visit End Token
                    event_tokens.append(self.end_token)
                    type_tokens.append(self.get_token_type_dict["end"])
                    age_tokens.append(age_mapping[prev_encounter])
                    time_tokens.append(time_mapping[prev_encounter])
                    visit_segments.append(segment_value)
                    position_tokens.append(position_value)

                    # Adding Register Token
                    event_tokens.append(self.register_token)
                    type_tokens.append(self.get_token_type_dict["reg"])
                    age_tokens.append(age_mapping[prev_encounter])
                    time_tokens.append(time_mapping[prev_encounter])
                    visit_segments.append(segment_value)
                    position_tokens.append(position_value)

                    # Adding interval token
                    event_tokens.append(intervals[event_encounter])
                    type_tokens.append(self.get_token_type_dict["time"])
                    age_tokens.append(0)
                    time_tokens.append(0)
                    visit_segments.append(0)
                    position_tokens.append(position_value)

                # Adding Visit Start Token
                event_tokens.append(self.start_token)
                type_tokens.append(self.get_token_type_dict["start"])
                age_tokens.append(age_mapping[event_encounter])
                time_tokens.append(time_mapping[event_encounter])

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
            prev_encounter = event_encounter

        event_tokens.append(self.end_token)
        type_tokens.append(self.get_token_type_dict["end"])
        age_tokens.append(age_mapping[event_encounter])
        time_tokens.append(time_mapping[event_encounter])
        visit_segments.append(segment_value)
        position_tokens.append(position_value)

        assert (
            len(event_tokens)
            == len(type_tokens)
            == len(age_tokens)
            == len(time_tokens)
            == len(visit_segments)
            == len(position_tokens)
        )

        row["token_length"] = len(event_tokens)
        row["event_tokens"] = event_tokens
        row["type_tokens"] = type_tokens
        row["age_tokens"] = age_tokens
        row["time_tokens"] = time_tokens
        row["visit_tokens"] = visit_segments
        row["position_tokens"] = position_tokens
        row["num_visits"] = len(set(position_tokens))
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
        if death_date is np.nan:
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
            print("after death events", row["patient_id"])
            self.after_death_events.append(row["patient_id"])

        row["deceased"] = 1
        row["death_after_start"] = death_start.days
        row["death_after_end"] = death_end.days
        return row

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
        sequence = self.to_list(row["event_tokens"])
        t_type = self.to_list(row["type_tokens"])
        age = self.to_list(row["age_tokens"])
        time = self.to_list(row["time_tokens"])
        visit = self.to_list(row["visit_tokens"])
        position = self.to_list(row["position_tokens"])
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
            return row

        if seq_length > self.max_seq_length:
            truncated = True
            # get the recent max_length tokens
            start_index = int(seq_length - self.max_seq_length) + 2
            end_index = int(seq_length)

            if sequence[start_index] == self.end_token:
                start_index += 3
            elif sequence[start_index] == self.register_token:
                start_index += 2
            elif sequence[start_index].startswith(("[W_", "[M_", "[LT")):
                start_index += 1

            if sequence[start_index] != self.start_token:
                new_sequence = [self.class_token, self.start_token] + sequence[
                    start_index:end_index
                ]
                new_type = [
                    self.get_token_type_dict["class"],
                    self.get_token_type_dict["start"],
                ] + t_type[start_index:end_index]
                new_age = [0, age[start_index]] + age[start_index:end_index]
                new_time = [0, time[start_index]] + time[start_index:end_index]
                new_visit = [0, visit[start_index]] + visit[start_index:end_index]
                new_position = [0, position[start_index]] + position[
                    start_index:end_index
                ]
            else:
                new_sequence = [self.class_token] + sequence[start_index:end_index]
                new_type = [self.get_token_type_dict["class"]] + t_type[
                    start_index:end_index
                ]
                new_age = [0] + age[start_index:end_index]
                new_time = [0] + time[start_index:end_index]
                new_visit = [0] + visit[start_index:end_index]
                new_position = [0] + position[start_index:end_index]

            new_position = [i - new_position[0] + 1 for i in new_position]

            row[f"event_tokens_{self.max_seq_length}"] = new_sequence
            row[f"type_tokens_{self.max_seq_length}"] = new_type
            row[f"age_tokens_{self.max_seq_length}"] = new_age
            row[f"time_tokens_{self.max_seq_length}"] = new_time
            row[f"visit_tokens_{self.max_seq_length}"] = new_visit
            row[f"position_tokens_{self.max_seq_length}"] = new_position
            seq_length = len(new_sequence)

        if seq_length < self.max_seq_length:
            padded_length = int(max(0, self.max_seq_length - seq_length))
            if truncated:
                if pad_events:
                    row[f"event_tokens_{self.max_seq_length}"] = (
                        row[f"event_tokens_{self.max_seq_length}"]
                        + [self.pad_token] * padded_length
                    )
                else:
                    # padding will be done in the tokenizer
                    row[f"event_tokens_{self.max_seq_length}"] = row[
                        f"event_tokens_{self.max_seq_length}"
                    ]

                row[f"type_tokens_{self.max_seq_length}"] = (
                    row[f"type_tokens_{self.max_seq_length}"]
                    + [self.get_token_type_dict["pad"]] * padded_length
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
                    row[f"position_tokens_{self.max_seq_length}"]
                    + [self.max_seq_length + 1] * padded_length
                )
            else:
                if pad_events:
                    row[f"event_tokens_{self.max_seq_length}"] = (
                        sequence + [self.pad_token] * padded_length
                    )
                else:
                    # padding will be done in the tokenizer
                    row[f"event_tokens_{self.max_seq_length}"] = sequence

                row[f"type_tokens_{self.max_seq_length}"] = (
                    t_type + [self.get_token_type_dict["pad"]] * padded_length
                )
                row[f"age_tokens_{self.max_seq_length}"] = age + [0] * padded_length
                row[f"time_tokens_{self.max_seq_length}"] = time + [0] * padded_length
                row[f"visit_tokens_{self.max_seq_length}"] = visit + [0] * padded_length
                row[f"position_tokens_{self.max_seq_length}"] = (
                    position + [self.max_seq_length + 1] * padded_length
                )

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

    def create_patient_sequence(
        self,
        chunksize: int = None,
        min_events: int = 0,
        min_visits: int = 0,
        pad_events: bool = False,
        all_columns: bool = True,
    ) -> None:
        """Create patient sequences and saves them as a parquet file."""
        file_paths = [
            f"{self.data_dir}/inpatient.csv",
            f"{self.data_dir}/encounters.csv",
            f"{self.data_dir}/procedures.csv",
            f"{self.data_dir}/med_requests.csv",
            f"{self.data_dir}/processed_labs.csv",
        ]
        rounds = 0
        readers = [pd.read_csv(path, chunksize=chunksize) for path in file_paths]
        while True:
            try:
                # read dataframes
                dataframes = [next(reader).reset_index(drop=True) for reader in readers]
                patients, encounters, procedures, medications, labs = dataframes
                # process encounters
                encounters = encounters.apply(self._sort_encounters, axis=1)
                encounters = encounters.apply(
                    lambda row: self._get_encounters_age(row, patients.iloc[row.name]),
                    axis=1,
                )
                encounters = encounters.apply(
                    lambda row: self._get_encounters_time(row),
                    axis=1,
                )
                encounters = encounters.apply(self._calculate_intervals, axis=1)
                # process events
                procedures = procedures.apply(
                    lambda row: self._edit_datetimes(
                        row,
                        encounters.iloc[row.name],
                        "proc",
                    ),
                    axis=1,
                )
                medications = medications.apply(
                    lambda row: self._edit_datetimes(
                        row,
                        encounters.iloc[row.name],
                        "med",
                    ),
                    axis=1,
                )
                labs = labs.apply(
                    lambda row: self._edit_datetimes(
                        row,
                        encounters.iloc[row.name],
                        "lab",
                    ),
                    axis=1,
                )
                # combine events
                combined_events = self._concat_concepts(
                    procedures,
                    medications,
                    labs,
                    encounters,
                )
                # filter patients based on min_events
                combined_events = combined_events[
                    combined_events["length"] > min_events
                ]
                # add special tokens to the events
                combined_events = combined_events.apply(
                    lambda row: self._add_tokens(row, encounters.iloc[row.name]),
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
                        patients.iloc[row.name],
                        encounters.iloc[row.name],
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
                if not all_columns:
                    output_columns = [
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
                    ]
                else:
                    output_columns = [
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
                        f"event_tokens_{self.max_seq_length}",
                        f"type_tokens_{self.max_seq_length}",
                        f"age_tokens_{self.max_seq_length}",
                        f"time_tokens_{self.max_seq_length}",
                        f"visit_tokens_{self.max_seq_length}",
                        f"position_tokens_{self.max_seq_length}",
                    ]
                combined_events = combined_events[output_columns]
                combined_events = combined_events.dropna(
                    subset=[f"event_tokens_{self.max_seq_length}"],
                )
                combined_events.to_parquet(
                    self.save_dir
                    + f"/patient_sequences_{self.max_seq_length}_{rounds}.parquet",
                    engine="pyarrow",
                )
                print(f"Round {rounds} done")
                rounds += 1
            except StopIteration:
                break

    def reapply_truncation(
        self,
        file_paths: Union[str, List[str]],
        all_columns: bool = False,
    ) -> None:
        """
        Reapply truncation to Parquet file(s).

        Parameters
        ----------
        file_paths : Union[str, List[str]]
            Path or list of paths to Parquet files to be processed.

        Returns
        -------
        None
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        for i, file_path in enumerate(sorted(file_paths)):
            df = pd.read_parquet(file_path)
            df = df.apply(
                lambda row: self._truncate_or_pad(row, pad_events=False),
                axis=1,
            )

            if not all_columns:
                output_columns = [
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
                ]
            else:
                output_columns = [
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
                    f"event_tokens_{self.max_seq_length}",
                    f"type_tokens_{self.max_seq_length}",
                    f"age_tokens_{self.max_seq_length}",
                    f"time_tokens_{self.max_seq_length}",
                    f"visit_tokens_{self.max_seq_length}",
                    f"position_tokens_{self.max_seq_length}",
                ]
            df = df[output_columns]
            base_name = f"patient_sequences_{self.max_seq_length}"
            suffix = f"_{i}" if len(file_paths) > 1 else ""
            file_name = f"{base_name}{suffix}.parquet"
            df.to_parquet(
                os.path.join(self.save_dir, file_name),
                engine="pyarrow",
            )


if __name__ == "__main__":
    generator = SequenceGenerator(
        max_seq_length=512,
        data_dir="/mnt/data/odyssey/mimiciv_fhir/csv_files",
        save_dir="/mnt/data/odyssey/mimiciv_fhir/parquet_files",
    )
    generator.create_patient_sequence(
        chunksize=10000,
        min_events=10,
        min_visits=0,
        all_columns=True,
    )
