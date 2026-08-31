"""What was submitted, what is still running, and what actually produced output.

Nobody watches the cluster any more. The agent runs on a laptop, submits and
goes away, so the only way to know where things stand is to reconstruct it —
and slurm alone cannot tell you. A labelling job's last command is `kill`, so
its exit code is `kill`'s and it reports COMPLETED whatever happened. Twice in
this project a run that slurm called COMPLETED had written nothing.

So each entry says what it is and what artefact would prove it finished, and
this reconciles three sources: squeue for what is live, sacct for how it
ended, and the artefact for whether that ending meant anything. The category
worth having is the disagreement — ended clean, produced nothing.

The ledger is `vlm_gate/local/jobs.jsonl`, which is committed and inside the
sync map, so it cannot become another file that exists only on the server.
"""
import json
import subprocess
from pathlib import Path

from . import paths

LEDGER = paths.VG / "local" / "jobs.jsonl"

LIVE = {"R": "running", "PD": "pending", "CG": "completing"}


def _read():
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(json.loads(line))
    return out


def add(job_id, what, expect=None):
    """Append one submission. Called right after sbatch, never later."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    entry = {"id": str(job_id), "what": what}
    if expect:
        entry["expect"] = expect
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _squeue(user):
    fmt = "%i|%T|%M|%R"
    p = subprocess.run(["squeue", "-h", "-u", user, "-o", fmt],
                       capture_output=True, text=True)
    live = {}
    for line in p.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        jid, state, elapsed, reason = (x.strip() for x in parts[:4])
        live.setdefault(jid.split("_")[0], []).append((state, elapsed, reason))
    return live


def _sacct(user):
    """Terminal states, collapsed per parent job id.

    An array is one entry in the ledger but many rows here, so the states are
    counted: "16 COMPLETED" is the useful summary, and a single FAILED among
    them is the thing you need to see.
    """
    p = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-u", user, "-S", "now-7days",
         "--format=JobID,State,Elapsed"],
        capture_output=True, text=True)
    if p.returncode != 0:
        return {}
    got = {}
    for line in p.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        jid, state, elapsed = (x.strip() for x in parts[:3])
        got.setdefault(jid.split("_")[0], []).append((state.split()[0], elapsed))
    return got


def _artefact(expect):
    """(ok, note). `ok` is None when the entry declares nothing to check."""
    if not expect:
        return None, "no artefact declared"
    kind = expect.get("kind")

    if kind == "labels":
        from . import labels
        try:
            r = labels.scan(expect["tag"], expect.get("rows"))
        except FileNotFoundError as e:
            return False, str(e).split(" under ")[0]
        note = f"{r['rows']} rows, {r['unique_chunks']} unique, {r['shards']} shards"
        if r["problems"]:
            note += f" — {r['problems'][0]}"
        # `ok` from scan means the rows are internally consistent, not that they
        # are all there: with no expected count, 49 rows out of 266,693 passes.
        # Completeness needs the number, which is why the entry has to declare
        # it — the same `--expected` the labelling check has always required.
        if expect.get("rows") is None:
            return None, note + " (no expected count declared)"
        return r["ok"], note

    if kind == "path":
        p = Path(expect["path"])
        if not p.is_absolute():
            p = paths.WS / p
        return p.exists(), str(p).replace(str(paths.WS) + "/", "")

    if kind == "glob":
        p = expect["path"]
        base = paths.WS if not p.startswith("/") else Path("/")
        hits = sorted(base.glob(p.lstrip("/")))
        want = expect.get("min", 1)
        return len(hits) >= want, f"{len(hits)} matching (want {want})"

    return None, f"unknown check {kind!r}"


def survey(user):
    live, done = _squeue(user), _sacct(user)
    rows = []
    for e in _read():
        jid = e["id"]
        if jid in live:
            states = [s for s, _, _ in live[jid]]
            elapsed = max(el for _, el, _ in live[jid])
            reason = live[jid][0][2]
            label = LIVE.get(states[0], states[0].lower())
            if len(live[jid]) > 1:
                label += f" ({len(live[jid])} tasks)"
            slurm = f"{label} {elapsed}"
            if states[0] == "PD":
                slurm += f" [{reason}]"
            status = "live"
        elif jid in done:
            counts = {}
            for s, _ in done[jid]:
                counts[s] = counts.get(s, 0) + 1
            slurm = ", ".join(f"{n} {s}" if n > 1 else s
                              for s, n in sorted(counts.items()))
            status = "ended"
        else:
            slurm, status = "not in squeue or last 7d of sacct", "unknown"

        ok, note = _artefact(e.get("expect"))
        rows.append({"id": jid, "what": e["what"], "status": status,
                     "slurm": slurm, "artefact_ok": ok, "artefact": note,
                     "clean": all(s == "COMPLETED" for s, _ in done.get(jid, []))})
    return rows


# COMPLETED proves nothing here, but anything else is still a real signal: a
# FAILED or TIMEOUT task means part of the work did not run, whatever landed on
# disk. So slurm can only ever veto, never confirm.
def verdict(r):
    """The one line that says what to do about this row."""
    if r["status"] == "live":
        return "running"
    if r["status"] == "ended" and not r["clean"]:
        return "FAILED"
    if r["artefact_ok"] is False:
        # Ended clean and produced nothing. The category this exists for.
        return "NO OUTPUT"
    if r["artefact_ok"] is True:
        return "done"
    return "unchecked" if r["status"] == "ended" else r["status"]
