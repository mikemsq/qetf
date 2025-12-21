from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import yaml

from quantetf.utils.logging import configure_logging

app = typer.Typer(help="QuantETF strategy platform CLI")


@app.command()
def hello() -> None:
    """Smoke check that the CLI is installed."""
    typer.echo("quantetf: OK")


@app.command()
def print_config(path: Path) -> None:
    """Print a YAML config file as JSON."""
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    typer.echo(json.dumps(obj, indent=2))


def main() -> None:
    configure_logging()
    app()


if __name__ == "__main__":
    main()
