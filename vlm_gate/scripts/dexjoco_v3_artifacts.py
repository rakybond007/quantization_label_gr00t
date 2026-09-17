"""Export/verify DexJoCo from its executed message builder, without API access."""
import hashlib
import json
from pathlib import Path

import dexjoco_v3_checks as checks

ROOT = Path(__file__).resolve().parents[1]


def panel():
    return json.loads((ROOT / 'tests/fixtures/dexjoco_v3_panel.json').read_text())


def assembled(row):
    # Only text is serialized. API callers supply actual PIL images in these slots.
    return checks.build_messages([None, None], row['instruction'], row['facts'], row['views'])


def artifacts():
    row = panel()[0]
    messages = assembled(row)
    filled = checks.text_dump(messages)
    template = checks.text_dump(assembled(dict(row, instruction='{instruction}')))
    template = template.replace(checks.normalize_facts(row['facts']), '{computed facts}')
    weights = '\n'.join(f'{k}  {checks.SIGN[k]:+d}  {checks.WEIGHT[k]:.12g}' for k in 'ABCD')
    return {
        'dexjoco_v3_FULL.txt': template,
        'dexjoco_v3_FILLED.txt': filled,
        'dexjoco_v3_guidance.txt': checks.GUIDANCE + '\n',
        'dexjoco_v3_questions.txt': checks.ASK + '\n',
        'dexjoco_v3_facts_example.txt': checks.normalize_facts(row['facts']) + '\n',
        'dexjoco_v3_sign_weight.txt': (
            'NGRADE 5; provisional comparison weights, not fitted deployment weights\n'
            'question  sign  weight\n' + weights + '\n\n'
            'g=(grade-1)/4\nconf=(1 + sum_safe(w*g) - sum_risk(w*g))/2\n'
            'Gemini returns integer grades, no logprobs. No speed label is assigned.\n'),
    }


def verify(out=None):
    out = Path(out) if out is not None else ROOT / 'prompts'
    bad = []
    for name, expected in artifacts().items():
        path = out / name
        if not path.exists() or path.read_text() != expected:
            bad.append(f'dexjoco: {name} differs from message builder')
    for row in panel():
        text = checks.text_dump(assembled(row))
        if hashlib.sha256(text.encode()).hexdigest() != row['prompt_sha256']:
            bad.append(f"dexjoco: runtime snapshot mismatch {row['task']}:{row['frame_index']}")
    for name in ('robocasa_v22', 'libero_v3d', 'humandata_v3'):
        text = (ROOT / 'prompts' / (name + '_FULL.txt')).read_text()
        prefix = text[text.index('The measurements above'):text.index('\nA)') + 1]
        if prefix != checks.ASK_PREFIX:
            bad.append(f'dexjoco: common ASK prefix differs from {name}')
    return bad


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        expected = artifacts()
        for name, text in expected.items():
            (out / name).write_text(text)
        assert not verify(out), 'unchanged artifacts must pass'
        for name, text in expected.items():
            (out / name).write_text(text + 'CORRUPTION\n')
            assert verify(out), f'missed mutation in {name}'
            (out / name).write_text(text)
        # Ensure actual builder contract rejects the historical speed-demand input.
        row = panel()[0]
        try:
            assembled(dict(row, facts=row['facts'] + '\nAt 2x block-last compression'))
        except AssertionError:
            pass
        else:
            raise AssertionError('speed-demand input was accepted')
    return True
