from __future__ import annotations

import unittest

from server import app


class ServerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = app.openapi()

    def test_start_endpoint_uses_typed_request(self) -> None:
        request_schema = self.schema["paths"]["/api/start_session"]["post"][
            "requestBody"
        ]["content"]["application/json"]["schema"]
        self.assertEqual(
            request_schema["$ref"],
            "#/components/schemas/StartSessionRequest",
        )

    def test_sensor_schema_exposes_both_supported_input_modes(self) -> None:
        properties = self.schema["components"]["schemas"]["SensorData"]["properties"]
        self.assertIn("normalized_features", properties)
        self.assertIn("raw_ecg", properties)
        self.assertIn("baseline_features", properties)
        self.assertIn("baseline_ecg", properties)
        normalized_schema = properties["normalized_features"]["anyOf"][0]
        self.assertEqual(normalized_schema["minItems"], 12)
        self.assertEqual(normalized_schema["maxItems"], 12)


if __name__ == "__main__":
    unittest.main()
