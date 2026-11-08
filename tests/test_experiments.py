"""Check leakage boundaries and prediction fallbacks with controlled fixtures."""

import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.data import NUMERIC, CATEGORICAL, redact_text, load_data
from experiments.estimators import MedianPricePerSqm, native_features, structured_preprocessor, text_pipeline
from experiments.evaluation import inverse_log, metrics, select_candidate, comparison_table


class ProtocolTests(unittest.TestCase):
    def test_median_uses_training_statistics_and_all_fallbacks(self):
        train = pd.DataFrame({"province_name": ["A", "A", "B"],
                              "district_name": ["D", "D", "E"],
                              "property_type_name": ["house", "house", "land"],
                              "area": [10., 20., 10.]})
        model = MedianPricePerSqm().fit(train, [100., 400., 1000.])
        evaluation = pd.DataFrame({"province_name": ["A", "A", "C", "C"],
                                   "district_name": ["D", "new", "new", "new"],
                                   "property_type_name": ["house", "house", "land", "new"],
                                   "area": [10., 10., 10., np.nan],
                                   "price": [999999.] * 4})
        prediction, levels = model.predict_with_fallback(evaluation)
        np.testing.assert_allclose(prediction, [150., 150., 1000., 200.])
        self.assertEqual(list(levels), ["district_type", "province_type", "type", "global"])
        evaluation["price"] = -1
        np.testing.assert_allclose(prediction, model.predict(evaluation))

    def test_text_redaction_removes_amounts_and_price_clauses(self):
        text = redact_text("Nhà đẹp. Giá 7,45 tỷ. Gần công viên; 3150tr; 4PN thoáng mát.")
        self.assertNotRegex(text, r"\d|\bgia\b|\bty\b|3150tr")
        self.assertIn("cong vien", text)

    def test_tfidf_vocabulary_is_training_only(self):
        model = text_pipeline(1., (1, 2), 100, 1, 1.)
        model.fit(["house garden", "apartment balcony", "house balcony"], [1., 2., 3.])
        vocabulary = model.named_steps["tfidf"].vocabulary_.copy()
        model.predict(["unseenvalidationtoken"])
        self.assertEqual(vocabulary, model.named_steps["tfidf"].vocabulary_)
        self.assertNotIn("unseenvalidationtoken", vocabulary)

    def test_native_categories_and_imputation_are_training_only(self):
        train = pd.DataFrame({name: [1., np.nan, 3., 4., 5., 6.] for name in NUMERIC})
        for name in CATEGORICAL:
            train[name] = "known"
        validation = train.iloc[:1].copy()
        validation["area"] = 99999.
        for name in CATEGORICAL:
            validation[name] = "evaluation_only"
        _, levels = native_features(train)
        encoded, _ = native_features(validation, levels)
        self.assertTrue(encoded[CATEGORICAL].isna().all().all())
        preprocessor = structured_preprocessor(scale=True).fit(train)
        imputer = preprocessor.named_transformers_["numeric"].named_steps["impute"]
        statistics = imputer.statistics_.copy()
        transformed = preprocessor.transform(validation)
        values = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(statistics, imputer.statistics_)
        self.assertEqual(statistics[0], 4.)

    def test_metrics_original_scale_and_validation_selection(self):
        score = metrics([10., 20.], [12., 18.])
        self.assertEqual(score["MAE"], 2.)
        self.assertEqual(score["RMSE"], 2.)
        self.assertAlmostEqual(score["MAPE"], 15.)
        model, parameters, _ = select_candidate(
            [{"value": 10.}, {"value": 15.}], lambda p: p["value"],
            [10., 20.], lambda m: np.repeat(m, 2))
        self.assertEqual(model, 10.)
        self.assertEqual(parameters, {"value": 10.})
        np.testing.assert_allclose(inverse_log([1000., -1.], [10., 20.]), [20., 0.])

    def test_frozen_cohort_temporal_and_entity_separation(self):
        parts, manifest = load_data()
        self.assertEqual(manifest["source"]["dataset"], "vduydong/vietnam-real-estates")
        self.assertEqual(manifest["counts"]["raw_rows"], 1_000_000)
        self.assertEqual(sum(map(len, parts.values())), manifest["cohort_rows"])

    def test_stale_artifacts_do_not_block_a_fresh_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            stale = {"protocol_id": "old", "metrics": {"test": {"MAE": 999.}}}
            fresh = {"protocol_id": "current", "metrics": {"test": {
                "MAE": 1., "RMSE": 2., "R2": .5, "MAPE": 10.}}}
            (path / "dummy.json").write_text(json.dumps(stale), encoding="utf-8")
            (path / "ridge.json").write_text(json.dumps(fresh), encoding="utf-8")
            with patch("experiments.evaluation.REPORTS", path), patch(
                    "experiments.evaluation.protocol_id", return_value="current"):
                table = comparison_table({})
            self.assertEqual(table.Model.tolist(), ["Ridge Regression"])


if __name__ == "__main__":
    unittest.main()
