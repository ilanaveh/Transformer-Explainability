import importlib
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from compare_models.wasserstein import normalize_map, wasserstein_distance_2d, make_2d_cost_matrix
from compare_models.calc_wasserstein_dist import compare_models_wasserstein
from compare_models.comp_wasserstein_dist import (
    prepare_paired_wasserstein_comparison, filter_all_predictions_correct,
    paired_wasserstein_tests, permutation_sign_flip_test,
)


class WassersteinTests(unittest.TestCase):
    def test_config_run_saves_effective_settings_and_source(self):
        import json
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            np.save(folder / 'map.npy', np.eye(2))
            rows = [dict(image_id='one', model=model, sigma=sigma,
                         true_label='Happy', pred_label='Happy', target_label='Happy',
                         correct=True, map_path='map.npy')
                    for model, sigma in [('A', 8), ('B', 0)]]
            pd.DataFrame(rows).to_csv(folder / 'metadata.csv', index=False)
            config = dict(metadata='metadata.csv', model_a='A', model_b='B',
                          sigma_a=8, sigma_b=0, target_mode='pred_label',
                          size=(2, 2), map_root='.', output_dir='runs')
            source = 'CONFIG = ' + repr(config) + '\n'
            config_path = folder / 'config.py'
            config_path.write_text(source)
            for _ in range(2):
                run = subprocess.run([sys.executable, '-m', 'compare_models.calc_wasserstein_dist',
                                      '--config', str(config_path)], cwd=root,
                                     capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
            outputs = list((folder / 'runs').glob('*/distances.csv'))
            self.assertEqual(len(outputs), 2)
            for output in outputs:
                self.assertEqual(len(pd.read_csv(output)), 1)
                saved = json.loads(output.with_suffix('.config.json').read_text())
                self.assertEqual(saved['sigma_a'], 8)
                self.assertEqual(saved['sigma_b'], 0)
                self.assertEqual(saved['target_mode'], 'pred_label')
                self.assertEqual(saved['size'], [2, 2])
                self.assertEqual(Path(saved['map_root']), folder)
                self.assertEqual((output.parent / 'wasserstein_config.py').read_text(), source)

    def test_known_distances_and_scale(self):
        a = np.array([[1., 0], [0, 0]])
        b = np.array([[0., 0], [0, 1]])
        c = np.array([[0., 1], [0, 0]])
        self.assertAlmostEqual(wasserstein_distance_2d(a, a * 7), 0)
        self.assertAlmostEqual(wasserstein_distance_2d(a, b), 1)
        self.assertAlmostEqual(wasserstein_distance_2d(a, c), 1 / np.sqrt(2))
        self.assertAlmostEqual(wasserstein_distance_2d(c, a), 1 / np.sqrt(2))
        self.assertAlmostEqual(wasserstein_distance_2d(a, b, normalize_cost=False), np.sqrt(2))

    def test_invalid_maps(self):
        for a in (np.zeros((2, 2)), np.full((2, 2), np.nan),
                  np.full((2, 2), np.inf), np.ones(3)):
            with self.subTest(a=a), self.assertRaises(ValueError):
                normalize_map(a)
        with self.assertRaises(ValueError):
            normalize_map(np.array([[-1., 2]]), clip_negative=False)
        with self.assertRaises(ValueError):
            wasserstein_distance_2d(np.ones((2, 2)), np.ones((3, 3)))

    def test_all_pipelines_use_same_scale(self):
        a, b = np.eye(2), np.fliplr(np.eye(2))
        expected = wasserstein_distance_2d(a, b)
        for module in ('calc_image_separability', 'calc_prototype_separability'):
            m = importlib.import_module('compare_models.' + module)
            np.testing.assert_allclose(m.create_spatial_cost_matrix(2, 2), make_2d_cost_matrix(2, 2))
        m = importlib.import_module('compare_models.calc_prototype_separability')
        self.assertAlmostEqual(m.wasserstein_2d_distance(a, b, transport_size=(2, 2)), expected)

    def test_old_cache_scale_is_rejected(self):
        from compare_models.calc_image_separability import _cache_metadata_matches, _make_pairwise_cache_metadata
        current = _make_pairwise_cache_metadata('A', np.array(['one']),
                                                np.array(['Happy']), (2, 2), True)
        old = dict(current)
        old.pop('distance_metric_version')
        self.assertFalse(_cache_metadata_matches(old, current))
        self.assertTrue(_cache_metadata_matches(current, current))

    def test_imports_have_no_experiment_side_effects(self):
        root = str(Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as directory:
            code = (f'import sys; sys.path.insert(0, {root!r}); '
                    'import compare_models.calc_wasserstein_dist; '
                    'import compare_models.comp_wasserstein_dist; '
                    'import compare_models.calc_image_separability; '
                    'import compare_models.calc_prototype_separability; '
                    'import compare_models.plot_wasserstein_comparison')
            run = subprocess.run([sys.executable, '-c', code], cwd=directory,
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cli_writes_distances_and_paired_tests(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            np.save(folder / 'map.npy', np.eye(2))
            rows = [dict(image_id=image, model=model, sigma=0,
                         true_label='Happy', pred_label='Happy', target_label='Happy',
                         correct=True, map_path='map.npy')
                    for model in ('A', 'B') for image in ('one', 'two')]
            pd.DataFrame(rows).to_csv(folder / 'metadata.csv', index=False)
            distance_path = folder / 'output' / 'distances.csv'
            run = subprocess.run([sys.executable, '-m', 'compare_models.calc_wasserstein_dist',
                                  str(folder / 'metadata.csv'), '--model-a', 'A', '--model-b', 'B',
                                  '--output', str(distance_path)], cwd=root,
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(len(pd.read_csv(distance_path)), 2)
            paired_dir = folder / 'paired'
            run = subprocess.run([sys.executable, '-m', 'compare_models.comp_wasserstein_dist',
                                  str(distance_path), str(distance_path), '--all-correct',
                                  '--n-boot', '10', '--n-perm', '10', '--output-dir', str(paired_dir)],
                                 cwd=root, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(pd.read_csv(paired_dir / 'paired_tests.csv').permutation_p.iloc[0], 1)

    def test_matching_conditions_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / 'map.npy', np.eye(2))
            rows = [dict(image_id='one', model=model, sigma=sigma,
                         true_label='Happy', pred_label='Happy', target_label='Happy',
                         correct=True, map_path='map.npy')
                    for model in ('A', 'B') for sigma in (0, 8)]
            df = pd.DataFrame(rows)
            result = compare_models_wasserstein(df, 'A', 'B', map_root=root)
            self.assertEqual(len(result), 2)
            self.assertTrue((result.sigma_a == result.sigma_b).all())
            self.assertTrue((result.transport_height == 2).all())
            cross = compare_models_wasserstein(df, 'A', 'B', sigma_a=0, sigma_b=8, map_root=root)
            self.assertEqual(len(cross), 1)
            self.assertEqual((cross.sigma_a.iloc[0], cross.sigma_b.iloc[0]), (0, 8))
            for bad in (pd.concat([df, df.iloc[:1]]), df[df.model == 'A']):
                with self.assertRaises(ValueError):
                    compare_models_wasserstein(bad, 'A', 'B', map_root=root)
            with self.assertRaises(ValueError):
                compare_models_wasserstein(df, 'A', 'B', sigma_a=0, map_root=root)
            np.save(root / 'map.npy', np.zeros((2, 2)))
            with self.assertRaisesRegex(ValueError, 'image_id=one'):
                compare_models_wasserstein(df, 'A', 'B', map_root=root)


class ComparisonTests(unittest.TestCase):
    def table(self):
        return pd.DataFrame(dict(image_id=['a', 'b'], target_label=['Happy'] * 2,
                                 wasserstein_distance=[0.1, 0.2],
                                 correct_a=[True, True], correct_b=[True, False]))

    def test_pairing_and_correctness(self):
        a = self.table()
        b = a.copy()
        b.wasserstein_distance += .1
        paired = prepare_paired_wasserstein_comparison(a, b)
        np.testing.assert_allclose(paired.delta_a_minus_b, -.1)
        x, y, keys = filter_all_predictions_correct(a, b)
        self.assertEqual((len(x), len(y), len(keys)), (1, 1, 1))
        with self.assertRaises(ValueError):
            prepare_paired_wasserstein_comparison(pd.concat([a, a]), b)
        a['distance_scale'], b['distance_scale'] = 'diagonal', 'grid_pixels'
        with self.assertRaises(ValueError):
            prepare_paired_wasserstein_comparison(a, b)

    def test_zero_deltas_and_permutation(self):
        result = paired_wasserstein_tests(pd.DataFrame({'delta_a_minus_b': [0., 0.]}),
                                          n_boot=10, n_perm=10)
        self.assertEqual(result.wilcoxon_p.iloc[0], 1)
        self.assertEqual(result.paired_t_p.iloc[0], 1)
        self.assertEqual(result.permutation_p.iloc[0], 1)
        p = permutation_sign_flip_test([-1.] * 20, n_perm=100, alternative='less')
        self.assertGreater(p, 0)
        self.assertLess(p, .1)


if __name__ == '__main__':
    unittest.main()
