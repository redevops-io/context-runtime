# License

`context-runtime` is **source-available**, not open source. It is licensed under the
**GNU Affero General Public License v3.0 or later**, as modified by the
**Commons Clause License Condition v1.0**.

- Full AGPL text: [LICENSE](LICENSE)
- The added condition: [LICENSE-COMMONS-CLAUSE](LICENSE-COMMONS-CLAUSE)

Copyright (c) 2025-2026 RedevOps.

    SPDX-License-Identifier: LicenseRef-AGPL-3.0-or-later-with-Commons-Clause

There is no registered SPDX identifier for this combination (the Commons Clause is
a licence *condition*, not an SPDX exception), so a `LicenseRef-` name is used. Tooling
expecting a standard identifier will report this project as unrecognised, and GitHub will not
label it AGPL. That is accurate: it is not plain AGPL. This repository is public.

## Effective point, and what stays as it was

Relicensed on **2026-10-22**, by the commit that added this file. `e87f5c8` is the last commit under
the previous declared terms (plain AGPL-3.0-or-later) (the latest tag at that point is `w4`). Anything obtained at or before that
commit was distributed under AGPL-3.0-or-later alone and remains available on those terms: a
licence change is not retroactive and cannot revoke a grant already made.

## What this permits, and what it does not

You may read, run, modify and self-host this software, including inside a company for its own business.
Every AGPL obligation applies, most importantly §13: if you modify it and let users interact with the
result over a network, those users are entitled to the source of your modified version.

You may **not sell it**, in the Commons Clause's sense: you may not charge third parties for the software
itself, for hosting it, or for consulting or support whose value derives entirely or substantially from it.

## Why this changed

`context-runtime` is part of the same family as `agentic-os`, `runtime-contracts`, `redevops-projects` and `redevops-rag`, which are AGPL-3.0-or-later with the Commons Clause (see `agentic-os/LICENSE.md` for the reasoning, including why a pure-AGPL repository cannot sit under a Commons-Clause dependency). This repository is aligned with them so one licence covers the family.

Commons Clause's own FAQ is explicit that adding it means the result is not open source in the OSI sense;
this repository therefore says *source-available* and names the licence instead of calling itself open source.
Descriptions of third-party software that really is open source are unchanged.

## Contribution history

All authoring commits in this repository are by RedevOps (Alex Mats, under several git identities, plus the
`arybach` GitHub account operating merges and some commits on `redevops-io/*` branches). No third-party
authorship was found. On that basis RedevOps holds the copyright needed to relicense. Recorded here because
relicensing over outside contributions without a CLA is the usual way this goes wrong. **Contributions from
anyone else are accepted under the licence in this file**, and a contributor licence agreement should be in
place before the first outside contribution is merged.

This file states the terms and the reasoning. It is not legal advice.
