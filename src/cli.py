import typer
from src.config import load_config
from src.ingest import run_ingest

app = typer.Typer(no_args_is_help=True)


@app.command()
def ingest(config: str = "configs/config.yaml"):
    """Liest Regulatorik-Dokumente ein und erzeugt Text-Segmente."""
    cfg = load_config(config)
    n = run_ingest(cfg)
    typer.echo(f"Erstellte Segmente: {n} -> {cfg.ingest.processed_out}")


if __name__ == "__main__":
    app()
