# Evidence cards

One YAML card per quantitative claim kept in this repository. `status` is exactly one of:
*Illustrative/synthetic* | *Internally reproduced* | *Externally reproduced* | *Field validated*
(the single exception is the ContractNLI card, whose status says the source run was located in a private repository but its per-case predictions were not retained).
No claim here is externally reproduced or field validated. Unknown fields say `unknown` or `NOT_IN_REPO`.
`python scripts/docs_check.py` checks the schema.
