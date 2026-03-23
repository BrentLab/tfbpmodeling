# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-06-01

### Fixes

- [#57](https://github.com/BrentLab/tfbpmodeling/issues/57): Adds a default
  threshold on the `extract_significant_coefficients` of threshold = 1e-14.
  This means that a non-zero coefficient must be greater than 1e-14 in
  absolute value in order to be considered significant. This is meant to
  avoid numeric instability at or beyond 1e-15
- [#101](https://github.com/BrentLab/tfbpmodeling/issues/101): Set the random_state
  default to 42 instead of None. Additionally, make the stage3_bootstrap random
  state random_state + 20 to ensure different bootstrap indices from the all-data
  and top-n stages.

[1.0.0]: https://github.com/BrentLab/tfbpmodeling/releases/tag/v1.0.0
[1.1.0]: https://github.com/BrentLab/tfbpmodeling/compare/v1.0.0...v1.1.0

## [1.0.0] - 2025-03-18

Moving out of development and beginning to track versions
