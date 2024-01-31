import os
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil import parser


class SequenceGenerator:
    """Generates patient sequences from the events dataframes.s"""

    def __init__(
        self,
        max_seq_length: int = 512,
        pad_token: str = "[PAD]",
        mask_token: str = "[MASK]",
        start_token: str = "[VS]",
        end_token: str = "[VE]",
        reference_time: str = "2020-01-01 00:00:00",
        data_dir: str = "data_files",
        save_dir: str = "data_files",
    ):
        self.max_seq_length = max_seq_length
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.start_token = start_token
        self.end_token = end_token
        self.reference_time = parser.parse(reference_time)
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.after_death_events = []

        os.makedirs(save_dir, exist_ok=True)

    @property
    def time_delta_tokens(self) -> list:
        """Gets the time delta tokens."""
        return (
            [f"W_{i}" for i in range(0, 4)] + [f"M_{i}" for i in range(0, 13)] + ["LT"]
        )

    @property
    def special_tokens(self) -> list:
        """Gets the special tokens."""
        return [
            self.pad_token,
            self.mask_token,
            self.start_token,
            self.end_token,
        ] + self.time_delta_tokens

    def _load_patients(self) -> pd.DataFrame:
        """Loads the patients dataframe."""
        patients = pd.read_csv(os.path.join(self.data_dir, "inpatient.csv"))
        return patients

    def _load_encounters(self) -> pd.DataFrame:
        """Loads the encounters dataframe."""
        encounters = pd.read_csv(os.path.join(self.data_dir, "encounters.csv"))
        return encounters

    def _load_procedures(self) -> pd.DataFrame:
        """Loads the procedures dataframe."""
        procedures = pd.read_csv(os.path.join(self.data_dir, "procedures.csv"))
        return procedures

    def _load_medications(self) -> pd.DataFrame:
        """Loads the medications dataframe."""
        medications = pd.read_csv(os.path.join(self.data_dir, "med_requests.csv"))
        return medications

    def _load_labs(self) -> pd.DataFrame:
        """Loads the labs dataframe."""
        labs = pd.read_csv(os.path.join(self.data_dir, "processed_labs.csv"))
        return labs

    def _sort_encounters(self, encounter_row: pd.Series) -> pd.Series:
        """Sorts the encounters by start time.

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
        self, encounter_row: pd.Series, patient_row: pd.Series
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
        """Gets the time of the encounters in weeks with respect to a reference start time.

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
        """Calculates the intervals between encounters.

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
        intervals = {}
        eq_encounters = {}
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
                if start_id not in eq_encounters.keys():
                    eq_encounters[start_id] = []
                if end_id not in eq_encounters.keys():
                    eq_encounters[end_id] = []

                eq_encounters[start_id].append(end_id)
                eq_encounters[end_id].append(start_id)
                continue

            days = abs(days)
            if days < 28:
                week_num = days // 7
                intervals[start_id] = f"W_{week_num}"
            elif 28 <= days <= 365:
                month_num = days // 30
                intervals[start_id] = f"M_{month_num}"
            else:
                intervals[start_id] = "LT"

        encounter_row["intervals"] = intervals
        encounter_row["eq_encounters"] = eq_encounters
        return encounter_row

    def _edit_datetimes(
        self, row: pd.Series, encounter_row: pd.Series, concept_name: str
    ) -> pd.Series:
        """Edits the datetimes of the events so that they won't fall
        out of the corresponding encounter time frame.

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
                row["encounter_ids"][i]
            )
            encounter_start = encounter_row["starts"][encounter_index]
            encounter_end = encounter_row["ends"][encounter_index]
            start_parsed = parser.parse(encounter_start)
            start_parsed = parser.parse(encounter_end)
            date_parsed = parser.parse(date)
            if date_parsed < start_parsed:
                date = encounter_start
            elif date_parsed > start_parsed:
                date = encounter_end
            dates.append(date)
        row[date_column] = dates
        return row

    def _concat_concepts(
        self,
        procedures: pd.DataFrame,
        medications: pd.DataFrame,
        labs: pd.DataFrame,
        encounters: pd.DataFrame,
    ) -> pd.DataFrame:
        """Concatenates the events of different concepts.

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
        for i in range(len(procedures)):
            proc_codes = procedures.iloc[i]["proc_codes"]
            proc_codes = [code.upper() for code in proc_codes]

            med_codes = medications.iloc[i]["med_codes"]
            med_codes = [code.upper() for code in med_codes]

            lab_codes = labs.iloc[i]["lab_codes"]
            lab_codes = [code.upper() for code in lab_codes]
            lab_bins = labs.iloc[i]["binned_values"]
            lab_codes = [f"{el1}_{el2}" for el1, el2 in zip(lab_codes, lab_bins)]

            proc_dates = procedures.iloc[i]["proc_dates"]
            med_dates = medications.iloc[i]["med_dates"]
            lab_dates = labs.iloc[i]["lab_dates"]

            proc_encounters = procedures.iloc[i]["encounter_ids"]
            med_encounters = medications.iloc[i]["encounter_ids"]
            lab_encounters = labs.iloc[i]["encounter_ids"]

            combined_codes = proc_codes + med_codes + lab_codes
            combined_dates = proc_dates + med_dates + lab_dates
            combined_encounters = proc_encounters + med_encounters + lab_encounters

            encounter_order = {
                encounter: i
                for i, encounter in enumerate(encounters.iloc[i]["encounter_ids"])
            }

            combined = sorted(
                zip(combined_dates, combined_codes, combined_encounters),
                key=lambda x: (x[0], encounter_order.get(x[2])),
            )
            sorted_dates, sorted_codes, sorted_encounters = (
                zip(*combined) if combined else ([], [], [])
            )
            procedures.at[i, "proc_dates"] = list(sorted_dates)
            procedures.at[i, "proc_codes"] = list(sorted_codes)
            procedures.at[i, "encounter_ids"] = list(sorted_encounters)
            procedures.at[i, "length"] = len(sorted_dates)

        return procedures

    def _add_tokens(self, row: pd.Series, encounter_row: pd.Series) -> pd.Series:
        """Adds tokens to the events.

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
            return row

        events_encounters = row["encounter_ids"]
        ecounters = encounter_row["encounter_ids"]
        intervals = encounter_row["intervals"]
        eq_encounters = encounter_row["eq_encounters"]
        age_mapping = dict(zip(ecounters, encounter_row["ages"]))
        time_mapping = dict(zip(ecounters, encounter_row["times"]))
        event_tokens = []
        age_tokens = []
        time_tokens = []
        visit_segments = []
        position_tokens = []

        segment_value = 1
        position_value = 0

        prev_encounter = None

        for event, event_encounter in zip(events, events_encounters):
            if (
                event_encounter != prev_encounter
                and event_encounter not in eq_encounters.keys()
            ):
                if prev_encounter is not None:
                    # Adding Visit End Token
                    event_tokens.append(self.end_token)
                    age_tokens.append(age_mapping[event_encounter])
                    time_tokens.append(time_mapping[event_encounter])
                    visit_segments.append(segment_value)
                    position_tokens.append(position_value)

                    # Adding interval token
                    event_tokens.append(intervals[event_encounter])
                    age_tokens.append(0)
                    time_tokens.append(0)
                    visit_segments.append(0)
                    position_tokens.append(position_value)

                # Adding Visit Start Token
                event_tokens.append(self.start_token)
                age_tokens.append(age_mapping[event_encounter])
                time_tokens.append(time_mapping[event_encounter])

                segment_value = 1 if segment_value == 2 else 2
                visit_segments.append(segment_value)

                if len(event_tokens) == 1 or event_tokens[-2] != "W0":
                    position_value = position_value + 1
                position_tokens.append(position_value)

            # Adding intermediate tokens
            event_tokens.append(event)
            age_tokens.append(age_mapping[event_encounter])
            time_tokens.append(time_mapping[event_encounter])
            visit_segments.append(segment_value)
            position_tokens.append(position_value)
            prev_encounter = event_encounter

        event_tokens.append(self.end_token)
        age_tokens.append(age_mapping[event_encounter])
        time_tokens.append(time_mapping[event_encounter])
        visit_segments.append(segment_value)
        position_tokens.append(position_value)

        assert (
            len(event_tokens)
            == len(age_tokens)
            == len(time_tokens)
            == len(visit_segments)
            == len(position_tokens)
        )

        row["token_length"] = len(event_tokens)
        row["event_tokens"] = event_tokens
        row["age_tokens"] = age_tokens
        row["time_tokens"] = time_tokens
        row["visit_tokens"] = visit_segments
        row["position_tokens"] = position_tokens
        row["num_visits"] = len(set(position_tokens))
        return row

    def _get_mortality_label(
        self, row: pd.Series, patient_row: pd.Series, encounter_row: pd.Series
    ) -> pd.Series:
        """Gets the mortality label for the patient.

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

    def _truncate_or_pad(self, row: pd.Series) -> pd.Series:
        """Truncates or pads the sequence to max_seq_length.

        Parameters
        ----------
        row : pd.Series
            row of the combined events dataframe

        Returns
        -------
        pd.Series
            Updated row with truncated or padded sequence
        """
        sequence = row["event_tokens"]
        age = row["age_tokens"]
        time = row["time_tokens"]
        visit = row["visit_tokens"]
        position = row["position_tokens"]
        seq_length = row["token_length"]
        truncated = False

        if seq_length == self.max_seq_length:
            row["event_tokens"] = sequence
            return row

        if seq_length > self.max_seq_length:
            truncated = True
            # get the recent max_length tokens
            start_index = int(seq_length - self.max_seq_length) + 1
            end_index = int(seq_length)

            if sequence[start_index] == self.end_token:
                start_index += 2
            elif sequence[start_index].startswith(("W_", "M_", "LT")):
                start_index += 1

            if sequence[start_index] != self.start_token:
                new_sequence = [self.start_token] + sequence[start_index:end_index]
                new_age = [age[start_index]] + age[start_index:end_index]
                new_time = [time[start_index]] + time[start_index:end_index]
                new_visit = [visit[start_index]] + visit[start_index:end_index]
                new_position = [position[start_index]] + position[start_index:end_index]
            else:
                new_sequence = sequence[start_index:end_index]
                new_age = age[start_index:end_index]
                new_time = time[start_index:end_index]
                new_visit = visit[start_index:end_index]
                new_position = position[start_index:end_index]

            new_position = [i - new_position[0] + 1 for i in new_position]

            row["event_tokens"] = new_sequence
            row["age_tokens"] = new_age
            row["time_tokens"] = new_time
            row["visit_tokens"] = new_visit
            row["position_tokens"] = new_position
            seq_length = len(new_sequence)

        if seq_length < self.max_seq_length:
            padded_length = int(max(0, self.max_seq_length - seq_length))
            row["event_tokens"] = row["event_tokens"] + [self.pad_token] * padded_length
            row["age_tokens"] = row["age_tokens"] + [0] * padded_length
            row["time_tokens"] = row["time_tokens"] + [0] * padded_length
            row["visit_tokens"] = row["visit_tokens"] + [0] * padded_length
            row["position_tokens"] = (
                row["position_tokens"] + [self.max_seq_length + 1] * padded_length
            )

        for key in [
            "event_tokens",
            "age_tokens",
            "time_tokens",
            "visit_tokens",
            "position_tokens",
        ]:
            assert (
                len(row[key]) == self.max_seq_length
            ), f"Length of {key} is {len(row[key])} and max_length is {self.max_seq_length}"
            row[key] = np.array(row[key])

        return row

    def create_patient_sequence(self) -> None:
        """Creates patient sequences and saves them as a parquet file."""
        patients = self._load_patients()
        encounters = self._load_encounters()
        procedures = self._load_procedures()
        medications = self._load_medications()
        labs = self._load_labs()
        # process encounters
        encounters = encounters.apply(self._sort_encounters, axis=1)
        encounters = encounters.apply(
            lambda row: self._get_encounters_age(row, patients.iloc[row.name]), axis=1
        )
        encounters = encounters.apply(
            lambda row: self._get_encounters_time(row), axis=1
        )
        encounters = encounters.apply(self._calculate_intervals, axis=1)
        # process events
        procedures = procedures.apply(
            lambda row: self._edit_datetimes(row, encounters.iloc[row.name], "proc"),
            axis=1,
        )
        medications = medications.apply(
            lambda row: self._edit_datetimes(row, encounters.iloc[row.name], "med"),
            axis=1,
        )
        labs = labs.apply(
            lambda row: self._edit_datetimes(row, encounters.iloc[row.name], "lab"),
            axis=1,
        )
        # combine events
        combined_events = self._concat_concepts(
            procedures, medications, labs, encounters
        )
        combined_events = combined_events[combined_events["length"] > 0]
        combined_events = combined_events.apply(
            lambda row: self._add_tokens(row, encounters.iloc[row.name]), axis=1
        )
        combined_events = combined_events.apply(
            lambda row: self._get_mortality_label(
                row, patients.iloc[row.name], encounters.iloc[row.name]
            ),
            axis=1,
        )
        combined_events = combined_events[
            ~combined_events["patient_id"].isin(self.after_death_events)
        ]
        combined_events = combined_events.apply(
            lambda row: self._truncate_or_pad(row), axis=1
        )

        output_columns = [
            "patient_id",
            "num_visits",
            "deceased",
            "death_after_start",
            "death_after_end",
            "length",
            "token_length",
            "event_tokens",
            "age_tokens",
            "time_tokens",
            "visit_tokens",
            "position_tokens",
        ]
        combined_events = combined_events[output_columns]
        combined_events.to_parquet(
            self.save_dir + "/patient_sequences.parquet",
            engine="pyarrow",
        )


if __name__ == "__main__":
    generator = SequenceGenerator()
    generator.create_patient_sequence()
