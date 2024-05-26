"""Events processor module for patient sequence generation."""

from ast import literal_eval
from typing import Dict, List

import pandas as pd
from dateutil import parser

from odyssey.data.constants import LAB, MED, PROC
from odyssey.data.seq._data_containers import PatientData
from odyssey.odyssey.data.seq._tokens import TokenConfig


class EventProcessor:
    """Process events dataframes.

    Parameters
    ----------
    token_config : TokenConfig
        Token configuration

    """

    def __init__(self, token_config: TokenConfig):
        """Initialize the event processor."""
        self.token_config = token_config

    def edit_event_datetimes(
        self,
        row: pd.Series,
        encounter_row: pd.Series,
        concept_name: str,
    ) -> pd.Series:
        """Edit the datetimes of events to fit within the encounter time frame.

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
        if concept_name == LAB:
            for name in [
                "encounter_ids",
                "lab_dates",
                "lab_codes",
                "binned_values",
                "lab_values",
                "lab_units",
            ]:
                row[name] = literal_eval(row[name])
        elif concept_name == MED:
            for name in ["encounter_ids", "med_dates", "med_codes"]:
                row[name] = literal_eval(row[name])
        elif concept_name == PROC:
            for name in ["encounter_ids", "proc_dates", "proc_codes"]:
                row[name] = literal_eval(row[name])
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

    def _update_encounter_order(
        self,
        encounter_order: Dict[str, int],
        eq_encounters: Dict[str, List[str]],
    ) -> Dict[str, int]:
        """Update the encounter order based on the equal encounters."""
        if eq_encounters == {}:
            return encounter_order
        visited = set()
        for encounter, eq_encs in eq_encounters.items():
            if encounter in encounter_order:
                for eq_enc in eq_encs:
                    if eq_enc in visited:
                        continue
                    new_order = min(
                        encounter_order[eq_enc],
                        encounter_order[encounter],
                    )
                    encounter_order[eq_enc] = new_order
                    encounter_order[encounter] = new_order
                    visited.add(encounter)

        return encounter_order

    def _concat_conditions(
        self,
        encounter_conditions: Dict[str, List[str]],
        encounter_order: Dict[str, int],
    ) -> Dict[int, List[str]]:
        """Concatenate the conditions of the encounters using the encounter order."""
        order_groups = {}
        for encounter, order in encounter_order.items():
            if order not in order_groups:
                order_groups[order] = [encounter]
            else:
                order_groups[order].append(encounter)

        concatenated_conditions = {}
        for order, encounters in order_groups.items():
            all_conditions = []
            for encounter in encounters:
                all_conditions.extend(encounter_conditions.get(encounter, []))
            concatenated_conditions[order] = all_conditions

        return concatenated_conditions

    def calculate_time_after_admission(
        self,
        row: pd.Series,
        encounter_row: pd.Series,
    ) -> pd.Series:
        """Calculate the time of the events after the admission."""
        elapsed_times = []
        for event_time, encounter_id in zip(row["proc_dates"], row["encounter_ids"]):
            encounter_index = encounter_row["encounter_ids"].index(encounter_id)
            start_time = encounter_row["starts"][encounter_index]
            start_time = parser.parse(start_time)
            event_time_ = parser.parse(event_time)
            elapsed_time = round((event_time_ - start_time).total_seconds() / 3600, 2)
            elapsed_times.append(elapsed_time)
        row["elapsed_time"] = elapsed_times

        return row

    def combine_events(
        self,
        patient_data: PatientData,
    ) -> pd.DataFrame:
        """Combine the events of different concepts.

        Parameters
        ----------
        patient_data : PatientData
            Patient data

        Returns
        -------
        pd.DataFrame
            Combined events

        """
        procedures = patient_data.events[PROC].data
        procedures["type_ids"] = [None] * len(procedures)
        if patient_data.conditions is not None:
            procedures["encounter_conditions"] = [None] * len(procedures)

        for i in range(len(procedures)):
            proc_codes = procedures.iloc[i]["proc_codes"]
            proc_codes = [code.upper() for code in proc_codes]

            med_codes = patient_data.events[MED].data.iloc[i]["med_codes"]
            med_codes = [code.upper() for code in med_codes]

            lab_codes = patient_data.events[LAB].data.iloc[i]["lab_codes"]
            lab_codes = [code.upper() for code in lab_codes]
            lab_bins = patient_data.events[LAB].data.iloc[i]["binned_values"]
            lab_codes = [f"{el1}_{el2}" for el1, el2 in zip(lab_codes, lab_bins)]

            proc_type_ids = [self.token_config.token_type_mapping.get(PROC)] * len(
                proc_codes
            )
            med_type_ids = [self.token_config.token_type_mapping.get(MED)] * len(
                med_codes
            )
            lab_type_ids = [self.token_config.token_type_mapping.get(LAB)] * len(
                lab_codes
            )
            proc_dates = procedures.iloc[i]["proc_dates"]
            med_dates = patient_data.events[MED].data.iloc[i]["med_dates"]
            lab_dates = patient_data.events[LAB].data.iloc[i]["lab_dates"]

            proc_encounters = procedures.iloc[i]["encounter_ids"]
            med_encounters = patient_data.events[MED].data.iloc[i]["encounter_ids"]
            lab_encounters = patient_data.events[LAB].data.iloc[i]["encounter_ids"]

            combined_codes = proc_codes + med_codes + lab_codes
            combined_dates = proc_dates + med_dates + lab_dates
            combined_encounters = proc_encounters + med_encounters + lab_encounters
            combined_type_ids = proc_type_ids + med_type_ids + lab_type_ids

            encounter_order = {
                encounter: i
                for i, encounter in enumerate(
                    patient_data.encounters.iloc[i]["encounter_ids"]
                )
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

            if patient_data.conditions is not None:
                cond_codes = patient_data.conditions.iloc[i]["encounter_conditions"]
                # keep the conditions of the existing encounters
                filtered_conditions = {
                    k: v for k, v in cond_codes.items() if k in combined_encounters
                }
                sorted_conditions = dict(
                    sorted(
                        filtered_conditions.items(),
                        key=lambda item: encounter_order.get(item[0], float("inf")),
                    ),
                )
                # Concat conditions if their encounters are equal
                encounter_order_eq = self._update_encounter_order(
                    encounter_order,
                    patient_data.encounters.iloc[i]["eq_encounters"],
                )
                concatenated_conditions = self._concat_conditions(
                    sorted_conditions,
                    encounter_order_eq,
                )
                procedures.at[i, "encounter_conditions"] = [concatenated_conditions]

            procedures.at[i, "proc_dates"] = list(sorted_dates)
            procedures.at[i, "proc_codes"] = list(sorted_codes)
            procedures.at[i, "encounter_ids"] = list(sorted_encounters)
            procedures.at[i, "type_ids"] = list(sorted_type_ids)
            procedures.at[i, "length"] = len(sorted_dates)

        return procedures
