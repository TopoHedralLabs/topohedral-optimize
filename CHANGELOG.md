# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- Validated constructors and builder methods for optimizer configuration.
- Optional `serde` support for public configuration and result types.
- Structured validation errors and deterministic bound iteration.
- Runnable crate and standalone examples.

### Changed

- Corrected public names including `BoundConstraints`, `scalar_minimize`,
  `line_search`, `line_search_1d`, `Bfgs`, and `Dfp`.
- Made option representations private and exposed stable accessors.
- Removed unnecessary `Debug` bounds from objective traits.

### Deprecated

- The former misspelled or abbreviated names remain temporarily available as
  migration aliases.
