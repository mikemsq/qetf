# src/quantetf/

The Python package containing all platform components.

Design goals:
- Keep components composable through small interfaces (alpha, risk, portfolio construction)
- Keep backtests deterministic via dataset snapshot IDs
- Keep production runs auditable via manifests and config hashes
