"""Encounter processor module for patient sequence generation."""

from ast import literal_eval
from datetime import datetime
from typing import Dict, List

import pandas as pd
from dateutil import parser


class EncounterProcessor:
    """Encounter processor for the patient sequences."""

    def validate_encounters(
        self,
        encounter_row: pd.Series,
        **events_row: pd.Series,
    ) -> pd.Series:
        """Identify valid encounters based on the events.

        Parameters
        ----------
        encounter_row : pd.Series
            A single row from the encounters DataFrame.
        event_row : pd.Series

        Returns
        -------
        pd.Series
            Valid encounters.

        """
        encounter_ids = []
        for _, event_row in events_row.items():
            event_encounter_ids = set(literal_eval(event_row["encounter_ids"]))
            encounter_ids.append(event_encounter_ids)
        valid_encounters = set.union(*encounter_ids)
        for col in ["encounter_ids", "starts", "ends"]:
            encounter_row[col] = literal_eval(encounter_row[col])
        encounter_ids = encounter_row["encounter_ids"]
        encounter_starts = encounter_row["starts"]
        encounter_ends = encounter_row["ends"]
        filtered_ids = [eid for eid in encounter_ids if eid in valid_encounters]
        filtered_starts = [
            start
            for eid, start in zip(encounter_ids, encounter_starts)
            if eid in valid_encounters
        ]
        filtered_ends = [
            end
            for eid, end in zip(encounter_ids, encounter_ends)
            if eid in valid_encounters
        ]
        encounter_row["encounter_ids"] = filtered_ids
        encounter_row["starts"] = filtered_starts
        encounter_row["ends"] = filtered_ends
        encounter_row["length"] = len(filtered_ids)

        return encounter_row

    def sort_encounters(self, encounter_row: pd.Series) -> pd.Series:
        """Sort the valid encounters by start time.

        Parameters
        ----------
        encounter_row : pd.Series
            Encounter row

        Returns
        -------
        pd.Series
            Sorted encounter row
        """
        starts = encounter_row["starts"]
        ends = encounter_row["ends"]
        ids = encounter_row["encounter_ids"]
        sorted_lists = sorted(zip(starts, ends, ids))
        starts, ends, ids = zip(*sorted_lists)
        encounter_row["starts"] = list(starts)
        encounter_row["ends"] = list(ends)
        encounter_row["encounter_ids"] = list(ids)

        return encounter_row

    def calculate_patient_ages(
        self, encounter_row: pd.Series, patient_row: pd.Series
    ) -> pd.Series:
        """Calculate patient ages at the time of encounters.

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

    def calculate_encounter_times(
        self, encounter_row: pd.Series, reference_time: str
    ) -> pd.Series:
        """Calculate time of encounters in weeks with respect to a reference time.

        Parameters
        ----------
        encounter_row : pd.Series
            Encounter row
        reference_time : str
            Reference time

        Returns
        -------
        pd.Series
            Updated row with encounter times

        """
        first_encounter = parser.parse(encounter_row["starts"][0]).replace(tzinfo=None)
        initial_value = (first_encounter - reference_time).days // 7
        ages = encounter_row["ages"]
        time_values = [initial_value + (age - ages[0]) * 53 for age in ages]
        encounter_row["times"] = time_values

        return encounter_row

    def calculate_intervals(self, encounter_row: pd.Series) -> pd.Series:
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
