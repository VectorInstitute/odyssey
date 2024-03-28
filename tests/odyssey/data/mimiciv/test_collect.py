"""Test FHIRDataCollector."""

import os
import shutil
from unittest import TestCase

from odyssey.data.mimiciv.collect import (
    DATA_COLLECTION_CONFIG,
    PATIENT,
    FHIRDataCollector,
)


class TestFHIRDataCollector(TestCase):
    """Test FHIRDataCollector."""

    def setUp(self) -> None:
        """Set up FHIRDataCollector."""
        self.save_dir = "./_data"
        self.collector = FHIRDataCollector(
            db_path="postgresql://postgres:pwd@localhost:5432/mimiciv-2.0",
            schema="mimic_fhir",
            save_dir=self.save_dir,
            buffer_size=10,
        )

    def tearDown(self) -> None:  # noqa: N802
        """Tear down FHIRDataCollector."""
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    def test_save_to_csv(self):
        """Test save_to_csv."""
        buffer = []
        save_path = os.path.join(
            self.save_dir,
            "test.csv",
        )
        for i in range(15):
            dummy_data = {
                "patient_id": i + 1,
                "birthDate": "2021-01-01",
                "gender": "M",
                "deceasedBoolean": False,
                "deceasedDateTime": None,
            }
            buffer.append(dummy_data)
            self.collector.save_to_csv(
                buffer,
                DATA_COLLECTION_CONFIG[PATIENT]["columns"],
                save_path,
            )
        self.assertTrue(os.path.exists(save_path))
        self.assertEqual(len(buffer), 5)
