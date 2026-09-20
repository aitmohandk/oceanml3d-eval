# Licensing and provenance

This repository is distributed under the **European Union Public Licence v. 1.2 (EUPL-1.2)**, the
licence of `oceanml3d-core`, whose products it evaluates and with which it shares
`product_contract.py` byte for byte. `LICENSE` holds the verbatim licence text (sha256
`57fb42fbcd0b037ce528ed8f72f1ec095d67bc6825ecf1448ff39be1fe68a4b4`, the same file as in `core`),
`NOTICE` holds the copyright, contributor and provenance statement. This note is a factual summary,
not legal advice.

## What was wrong before

`pyproject.toml` declared `license = { text = "MIT" }` and the repository carried **no `LICENSE`
file**. A public repository without a licence file is all rights reserved by default: nobody could
legally reuse it, whatever the metadata said. And MIT was not available to declare, for the same
reason as in `core`: `legacy_from_fm/` (~11.8 k lines) is copied from `CIA-Oceanix/4dvarnet-fm-opencode`,
which published no terms at all. The terms of that material come from the rightsholder's permission
recorded in `oceanml3d-core/LICENSING.md` — IMT Atlantique / OceaniX holds both repositories.

## The vendored LGPL package

`third_party/velocity_metrics/` is OceanDataLab's `velocity_metrics`, LGPL-3.0-or-later. Its own
README asked for the licence text to be added "before redistributing", which became due the moment
this repository was made public: `LICENSE` (LGPL-3.0) and `COPYING` (GPL-3.0, which the LGPL
extends) are now distributed with it.

It stays under its own licence, not the EUPL. It is vendored unchanged, installed separately
(`pip install -e third_party/velocity_metrics`), optional — the `lagrangian_drifters` metric has a
built-in RK4 backend — and used through its public interface by
`oceanml3d_eval/velocity_metrics_backend.py`. Modifying those sources would require distributing the
modifications under the LGPL; adding a backend on this side does not.

## Provenance by directory

| Path | Origin | Licence |
|---|---|---|
| `oceanml3d_eval/` (metrics, products, regions, references, benchmarks, report) | written for this repository | EUPL-1.2 |
| `oceanml3d_eval/product_contract.py` | shared byte for byte with `oceanml3d-core` | EUPL-1.2 |
| `legacy_from_fm/` | `4dvarnet-fm-opencode` (`evaluation/` + the root `eval_*.py`), being converted into benchmarks | EUPL-1.2, by the permission above |
| `benchmarks/`, `products/`, `regions/` | written here; region JSON in the NOSC / velocity_metrics format | EUPL-1.2 |
| `third_party/velocity_metrics/` | OceanDataLab, vendored unchanged | LGPL-3.0-or-later |
