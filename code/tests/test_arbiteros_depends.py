import os
import sys
import unittest


TRAE_AGENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trae_agent'))
if TRAE_AGENT_DIR not in sys.path:
    sys.path.insert(0, TRAE_AGENT_DIR)

from agents.arbiteros_depends import DependsConfig, build_keep_set


class FakeManager:
    def __init__(self, dependencies):
        self.steps = [
            [
                {
                    'role': 'assistant',
                    'content': f'step {idx}',
                    'agent_depends_on': list(deps),
                }
            ]
            for idx, deps in enumerate(dependencies)
        ]


class HybridKeepSetTests(unittest.TestCase):
    def make_config(self, *, first=2, recent=2, hops=2):
        return DependsConfig(
            mode='arbiteros_hybrid',
            protect_first_steps=first,
            protect_recent_steps=recent,
            frontier_hops=hops,
        )

    def test_keeps_first_recent_and_exactly_two_dependency_levels(self):
        manager = FakeManager([
            [],              # step_0: protected first
            [],              # step_1: protected first
            [],              # step_2: third level, must remain compressible
            ['step_2'],      # step_3: second level from step_7
            ['step_2'],      # step_4: second level from step_8
            ['step_3'],      # step_5: first level from step_7
            ['step_4'],      # step_6: first level from step_8
            ['step_5'],      # step_7: protected recent
            ['user', 'step_6'],  # step_8: protected recent
        ])

        keep = build_keep_set(manager, self.make_config())

        self.assertEqual(
            keep,
            {
                'user',
                'step_0', 'step_1',
                'step_3', 'step_4', 'step_5', 'step_6',
                'step_7', 'step_8',
            },
        )
        self.assertNotIn('step_2', keep)

    def test_every_recent_step_seeds_dependency_traversal(self):
        manager = FakeManager([
            [],
            [],
            [],
            ['step_0'],
            ['step_1'],
        ])

        keep = build_keep_set(
            manager,
            self.make_config(first=0, recent=2, hops=1),
        )

        self.assertEqual(keep, {'user', 'step_0', 'step_1', 'step_3', 'step_4'})

    def test_zero_hops_keeps_only_protected_windows(self):
        manager = FakeManager([
            ['step_2'],
            [],
            [],
            ['step_1'],
            ['step_2'],
        ])

        keep = build_keep_set(
            manager,
            self.make_config(first=1, recent=2, hops=0),
        )

        self.assertEqual(keep, {'user', 'step_0', 'step_3', 'step_4'})

    def test_first_steps_do_not_seed_dependency_traversal(self):
        manager = FakeManager([
            ['step_2'],
            [],
            [],
            [],
        ])

        keep = build_keep_set(
            manager,
            self.make_config(first=1, recent=1, hops=2),
        )

        self.assertEqual(keep, {'user', 'step_0', 'step_3'})
        self.assertNotIn('step_2', keep)


if __name__ == '__main__':
    unittest.main()
