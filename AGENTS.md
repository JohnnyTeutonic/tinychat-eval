# Evaluation workspace

Keep evaluation independent of trainer and engine implementation. Core tests use
`python -m unittest discover -q`. Preserve immutable run receipts and synthetic suite
versions. Never train on evaluation fixtures or recorded evaluation conversations.
Read only explicitly designated environment files at runtime; never print secrets.
No cloud/model calls unless the command explicitly opts in and supplies its call cap.
Use the parent workspace CPU budget for jobs over a minute. No local CUDA jobs.
Exact contract scores, operational failures, repeatability and fallible judge ratings
remain separate. Do not repair gold answers after seeing candidate output.
