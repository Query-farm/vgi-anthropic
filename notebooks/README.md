# Anthropic worker notebooks

[01 · Where did the default workspace's spend go?](01_where_default_workspace_spend_went.ipynb)
investigates a cost-report reconciliation error: an inner join drops the default
workspace because its ID is NULL and the directory omits it. SQL audits the
missing matches, preserves them in a left join, verifies exact cents/USD amounts,
and compares daily composition. The saved notebook includes successful outputs.

## Install and run

Install [uv](https://docs.astral.sh/uv/). This directory is self-contained: it can
be copied away from the worker checkout, with no root environment or local worker
build. Run these commands from this directory:

```sh
uv sync --locked
uv run --locked python execute.py
```

The notebook environment pins Haybarn **2.0.0rc11** and locks JupySQL, Seaborn,
Matplotlib and the execution tools. A temporary kernel uses this environment's
Python; it does not change registered kernels. The visible literal ATTACH launches
`uvx` with Python 3.13 and Git revision
`4da037833b6737c47ffd5ff4cf677c2a947ec7fb` from the public worker repository.
SDK **vgi-python 0.51.1**, **vgi-rpc 0.53.1**, and HTTPX **0.28.1** are pinned in
that command. Git must be installed. Initial extension, Python and package
downloads require internet access. No worker compilation is required.

For an existing Jupyter frontend, register this environment's kernel:

```sh
uv run --locked python -m ipykernel install --user \
  --name vgi-anthropic-notebooks --display-name "Python (vgi-anthropic notebooks)"
```

Select that kernel and **Restart Kernel → Run All** after changing parameters.
The runner executes with this directory as cwd and replaces a notebook only after
all cells succeed. It refuses parameter overrides without a separate destination.

## Focused experiment

```sh
uv run --locked python execute.py --parameters '{"REPORT_DAYS": 3}' \
  --output-dir artifacts/three-days
```

Compare the recovered dollar amount, its window share, and the daily share panel.
The fixture's later days have a larger default-workspace allocation, so the
three-day window should lose less money and a smaller share under the inner join.
Restore `REPORT_DAYS = 7` for the default. Experiment copies include assets, this
guide, fixture, helpers, runner, and lockfile; generated artifacts are ignored.

## Data provenance and boundaries

**Every accounting value is synthetic.** `synthetic_admin_api.py` serves only
`/v1/organizations/workspaces` and `/v1/organizations/cost_report` on loopback.
The seven UTC days beginning 2026-09-01 contain invented Production, Staging and
default-workspace ledger entries. Their trends are deliberate teaching examples,
not sampled organizational activity or model token prices. An archived directory
entry illustrates the distinction between directory membership and window spend.

The fixture follows the payload and pagination shapes in the repository's
`tests/fake_admin_api.py` at the pinned revision (Query Farm, MIT). The real remote
worker still performs HTTP requests, pagination, parsing and decimal conversion;
Python does not manufacture notebook query results. Both endpoints page at two
items/buckets per page. Unsupported windows and endpoints fail explicitly. No
requests are forwarded to Anthropic. A public dummy SQL secret authenticates only
to this fixture. No environment credentials are read or saved. Cleanup closes
the database and fixture and restores the previous base-URL environment setting.

Only full UTC days are supported: start inclusive, end exclusive, 1–7 days from
2026-09-01. The notebook explicitly groups by workspace before treating NULL as
the default workspace. In an ungrouped report, NULL instead describes the aggregate.
Unknown non-NULL workspace IDs retain separate labels. Money reconciliation stays
in decimal SQL; charts convert results for display. A real cost report is not an
invoice reconciliation and does not include every category, including Priority Tier
spend. No live account or paid model calls were used for validation.

## Reproducibility and upgrades

The worker revision's root lock pins SDK 0.50.0, which speaks VGI protocol 3.2.
The community extension available during validation speaks 3.3; that original
combination fails at ATTACH. This notebook pins compatible SDK 0.51.1 in its uvx
launch. It leaves the worker's source and root lock unchanged. Source validation
with a separate SDK overlay is reported separately from the baseline failure.
The community extension is downloaded by `INSTALL vgi FROM community`; its release
can change independently of the Python lock. When upgrading the worker or SDK,
change the visible pin, rerun both windows in fresh kernels, and review the saved
charts and reconciliation checks before replacing the successful default.

The unchanged local Strata Sun and VGI assets link to Query.Farm and VGI.

## Validation recorded for this example

Both 7-day and 3-day runs executed from a temporary copy containing only this
directory. Each saved 19 code cells, two charts, and eight passing SQL relationships.
The repository's 196 offline tests pass with the SDK 0.51.1 overlay; its 26 live
tests were excluded. Ruff checks and both entrypoint help commands pass. Structural
and behavioral vgi-lint checks using the existing `tests/run_worker.py` evaluator
adapter report 100/100, no errors, and one existing VGI1011 warning about missing
secret-parameter documentation (plus six informational findings); the warning
threshold therefore still exits nonzero. The unmodified root-lock protocol mismatch
and dev-environment evaluator mismatch are not counted as passing checks.
