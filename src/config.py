from pydantic import BaseModel
import yaml


class IngestConfig(BaseModel):
    input_dir: str = "Regulatories"
    processed_out: str = "data/processed/segments.jsonl"
    min_section_chars: int = 700
    max_section_chars: int = 1300
    overlap_chars: int = 120


class AppConfig(BaseModel):
    ingest: IngestConfig = IngestConfig()


def load_config(path: str = "configs/config.yaml") -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        return AppConfig(**yaml.safe_load(f))
