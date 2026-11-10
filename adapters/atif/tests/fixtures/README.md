# Fixture provenance

`harbor-timeout.trajectory.json` is an unmodified copy of
`tests/golden/terminus_2/hello-world-timeout.trajectory.json` from
[Harbor](https://github.com/harbor-framework/harbor), commit
`9a2e3b135cc8fb1e41131020f83370cb2f12ae93` (Apache-2.0, as is this repository).
It exercises real producer task extraction, absent trajectory IDs, observations,
and exclusion of token arrays from judge prompts. The embedded task text is
untrusted input to the judge, not instructions for running these tests.

The other fixtures are adapter-authored contract examples. The failure corpus
has labels for future empirical evaluations; mocked tests do not establish
real-model categorization quality.
