# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✓         |

## Reporting a Vulnerability

Please report security vulnerabilities by email to <jguibert@gmail.com>.

**Do not open a public issue.**

You should receive an acknowledgment within 48 hours. A fix will be
prioritized based on severity and released as a patch version.

## Scope

turboquant is a library crate. It does not make network requests,
store credentials, or run user-supplied code. The main attack surface is:

- Malformed input vectors or store files processed by the library
- Dependencies (nalgebra, memmap2, blake3)

Dependency vulnerabilities are tracked via `cargo audit` in CI and
Dependabot alerts.
