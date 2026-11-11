"""Independent expectations for subject grouping and aggregation semantics."""

import unittest
import numpy as np

from har_compare.subject_eval import grouped_metrics, summarize


class SubjectMetricTests(unittest.TestCase):
    def test_grouped_confusions_reconstruct_all_windows_and_keep_six_classes(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        prediction = np.array([0, 1, 1, 1, 0, 2])
        people = np.array([8, 8, 8, 20, 20, 20])
        pooled, groups = grouped_metrics(truth, prediction, people)
        self.assertEqual([x['subject'] for x in groups], [8, 20])
        self.assertEqual(pooled['n_samples'], 6)
        self.assertEqual(pooled['accuracy'], 4/6)
        expected = np.zeros((6, 6), dtype=int)
        expected[:3, :3] = [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
        np.testing.assert_array_equal(pooled['confusion_matrix'], expected)
        np.testing.assert_array_equal(np.sum([x['metrics']['confusion_matrix'] for x in groups], axis=0), expected)
        # Absent activities stay in the denominator: person 8 has two nonzero F1s.
        self.assertAlmostEqual(groups[0]['metrics']['macro_f1'], (2/3 + 2/3)/6)

    def test_equal_person_mean_is_not_window_weighted_or_pooled_f1(self):
        truth = np.tile(np.arange(6), 4)
        people = np.repeat([8, 20], [6, 18])
        prediction = truth.copy()
        prediction[6:] = (prediction[6:] + 1) % 6
        pooled, groups = grouped_metrics(truth, prediction, people)
        runs = [{"model": m, "seed": s, "pooled": pooled, "subjects": groups}
                for m in ('cnn', 'rnn', 'lstm') for s in (42, 43, 44)]
        result, rows = summarize(runs)
        self.assertAlmostEqual(result['cnn']['pooled_window_accuracy']['mean'], .25)
        self.assertAlmostEqual(result['cnn']['pooled_window_macro_f1']['mean'], .25)
        self.assertAlmostEqual(result['cnn']['equal_subject_accuracy']['mean'], .5)
        self.assertAlmostEqual(result['cnn']['equal_subject_macro_f1']['mean'], .5)
        self.assertEqual(len(rows), 6)
        self.assertIsNone(result['cnn']['subjects'][0]['sit_stand_share_of_errors_pct'])

    def test_invalid_group_alignment_or_missing_seed_is_rejected(self):
        with self.assertRaises(ValueError):
            grouped_metrics(np.array([0, 1]), np.array([0, 1]), np.array([7]))
        with self.assertRaises(ValueError):
            grouped_metrics(np.array([0, 1]), np.array([0, 1]), np.array([7., 7.]))
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == '__main__':
    unittest.main()
