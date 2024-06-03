"""Data containers for the patient sequences."""

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class EventData:
    """Event data for the patient sequences.

    Parameters
    ----------
    event_type : str
        Event type, e.g. procedures, medications, labs.
    data : pd.DataFrame
        Dataframe with the event data, one row per patient.

    """

    event_type: str
    data: pd.DataFrame


@dataclass
class PatientData:
    """Patient data for the patient sequences.

    Each dataframe has one row per patient.

    Parameters
    ----------
    patients: pd.DataFrame
        Patients dataframe.
    encounters: pd.DataFrame
        Encounters dataframe.
    conditions: pd.DataFrame
        Conditions dataframe.
    events: Dict[str, EventData]
        Event dataframes.

    """

    patients: pd.DataFrame
    encounters: pd.DataFrame
    conditions: pd.DataFrame
    events: Dict[str, EventData]
