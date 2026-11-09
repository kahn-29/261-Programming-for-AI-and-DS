"""Check leakage boundaries and prediction fallbacks with controlled fixtures."""

import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.data import NUMERIC, CATEGORICAL, redact_text, load_data, split_fingerprint, normalize
from experiments.estimators import MedianPricePerSqm, native_features, structured_preprocessor, text_pipeline
from experiments.evaluation import inverse_log, metrics, select_candidate, comparison_table, environment


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
            manifest = {"split_sha256": "current", "source": {"revision": "r1"}}
            stale = {"split_sha256": "old", "revision": "r1", "metrics": {"test": {"MAE": 999.}}}
            fresh = {"split_sha256": "current", "revision": "r1", "metrics": {"test": {
                "MAE": 1., "RMSE": 2., "R2": .5, "MAPE": 10.}}}
            (path / "dummy.json").write_text(json.dumps(stale), encoding="utf-8")
            (path / "ridge.json").write_text(json.dumps(fresh), encoding="utf-8")
            with patch("experiments.evaluation.REPORTS", path):
                table = comparison_table(manifest)
            self.assertEqual(table.Model.tolist(), ["Ridge Regression"])

    def test_split_fingerprint_ignores_operating_system(self):
        frame = pd.DataFrame({"row_id": ["a", "b"], "group_id": ["g1", "g2"],
                              "split": ["train", "test"]})
        with patch("os.linesep", "\n"):
            unix = split_fingerprint(frame)
        with patch("os.linesep", "\r\n"):
            windows = split_fingerprint(frame)
        self.assertEqual(unix, windows)

    def test_environment_tolerates_missing_packages(self):
        with patch("experiments.evaluation.PACKAGES", ["numpy", "surely-not-installed-pkg"]):
            packages = environment()["packages"]
        self.assertIsNotNone(packages["numpy"])
        self.assertIsNone(packages["surely-not-installed-pkg"])

    def test_accented_redaction_matches_tfidf_redaction(self):
        from experiments.text_vi import redact_accented
        samples = ["Bán đất 70m², giá 10,85 tỷ. Ô tô vào nhà, sổ đỏ chính chủ",
                   "Căn hộ 2PN view biển; liên hệ [phone_number]",
                   "Nhà mặt phố. Giá thỏa thuận! Hướng Đông Nam"]
        for text in samples:
            accented = redact_accented(text)
            self.assertEqual(normalize(accented), redact_text(text))
        self.assertEqual(redact_accented(samples[0]), "Bán đất Ô tô vào nhà sổ đỏ chính chủ")

    def test_tabular_encoder_is_training_only_and_qualifies_locations(self):
        from experiments.deep import TabularEncoder
        base = {c: "__MISSING__" for c in CATEGORICAL}
        base.update({n: 1.0 for n in NUMERIC})
        train = pd.DataFrame([{**base, "province_name": p, "district_name": "Q1"}
                              for p in ["HN"] * 5 + ["HCM"] * 5])
        train.loc[0, "area"] = 1e9                       # extreme value
        encoder = TabularEncoder(min_frequency=5).fit(train)
        self.assertEqual(len(encoder.vocab_["district_name"]), 2)   # HN|Q1 and HCM|Q1
        unseen = train.iloc[:1].assign(province_name="DN")
        dense, codes = encoder.transform(unseen)
        district = CATEGORICAL.index("district_name")
        self.assertEqual(codes[0, district], 0)          # unseen -> rare/unseen index
        self.assertTrue(np.isfinite(dense).all())
        self.assertLessEqual(dense[0, NUMERIC.index("area")],
                             (encoder.upper_["area"] - encoder.mean_["area"]) / encoder.std_["area"] + 1e-6)


    def test_phobert_layerwise_lr_and_delayed_encoder_warmup(self):
        import torch
        from transformers import RobertaConfig, RobertaModel
        from experiments.phobert import PhoBERTRegressor, layerwise_groups, warmup_linear
        config = RobertaConfig(vocab_size=50, hidden_size=16, num_hidden_layers=2,
                               num_attention_heads=2, intermediate_size=32,
                               max_position_embeddings=40)
        model = PhoBERTRegressor(RobertaModel(config), head_hidden=8)
        groups = layerwise_groups(model, base_lr=1e-4, head_lr=1e-3, decay=0.5, weight_decay=0.01)
        covered = {id(p) for group in groups for p in group["params"]}
        expected = {id(p) for name, p in model.named_parameters() if ".pooler." not in f".{name}"}
        self.assertEqual(covered, expected)
        rates = {round(group["lr"], 10) for group in groups}
        self.assertEqual(rates, {1e-3, 1e-4, 5e-5, 2.5e-5})   # head, layer1, layer0, embeddings
        self.assertTrue(all(g["weight_decay"] == 0.0 for g in groups
                            if any(p.ndim == 1 for p in g["params"])))
        optimizer = torch.optim.AdamW(groups)
        scheduler = warmup_linear(optimizer, warmup_steps=4, total_steps=20, encoder_start=5)
        for _ in range(5):
            optimizer.step(); scheduler.step()
        head = [g["lr"] for g in optimizer.param_groups if g["is_head"]]
        encoder = [g["lr"] for g in optimizer.param_groups if not g["is_head"]]
        self.assertTrue(all(lr > 0 for lr in head))
        self.assertTrue(all(0 < lr <= 0.25 * 1e-4 + 1e-12 for lr in encoder))  # first warm-up step


if __name__ == "__main__":
    unittest.main()
