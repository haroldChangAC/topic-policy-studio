"""Protect pre-model synthetic expectations and evidence offsets."""
import hashlib
import json
import unittest
from pathlib import Path

from studio.policy_assist import evidence_matches
from studio.runtime import ROOT


class LocalPolicyFixtureTests(unittest.TestCase):
    def test_frozen_cases_and_expected_positions(self):
        path = ROOT / 'tests/fixtures/policy_local_eval_2026-11-07.json'
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), '07d31c521f897d77e808f861ed5e30121dd2dc83674700e5e7c6d3f0d7c32f8a')
        fixture = json.loads(raw)
        self.assertTrue(fixture['prepared_before_inference'])
        cases = fixture['cases']
        self.assertEqual([c['case_id'] for c in cases], ['S01_explicit','S02_absent','S03_near_words','S04_repeated','S05_late'])
        for case in cases:
            self.assertTrue(case['synthetic'])
            self.assertEqual(len(case['text']), case['length_chars'])
            if case['pre_model_exact_evidence']:
                matches = evidence_matches(case['text'],case['pre_model_exact_evidence'])[1]
                self.assertEqual([[x['start'],x['end']] for x in matches],case['pre_model_evidence_positions'])
        self.assertEqual(len(cases[3]['pre_model_evidence_positions']),2)
        self.assertGreater(cases[4]['pre_model_evidence_positions'][0][0] / cases[4]['length_chars'],0.95)


if __name__=='__main__': unittest.main()
