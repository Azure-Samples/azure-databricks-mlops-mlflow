import os
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
from diabetes.scoring.api.run import app as flask_app

flask_app.testing = True


class TestScoringAPI(TestCase):
    @patch("diabetes.scoring.api.run.model")
    def test_post_request(self, mock_predict):
        with flask_app.test_client() as client:
            score_data_file = os.path.join("tests/diabetes/data", "scoring_dataset.csv")
            score_df = pd.read_csv(score_data_file).drop(columns=["SEX"])
            json_data = score_df.to_json(orient="values")
            expected_result = {"prediction": [60.75743442, 67.10061271]}
            mock_predict.predict.return_value = [60.75743442, 67.10061271]
            result = client.post(
                "/predict",
                data=json_data,
                content_type="application/json",
            )
            print(f"result: {result.json}")
            self.assertEqual(result.json, expected_result)
