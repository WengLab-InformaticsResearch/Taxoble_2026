from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathConfig:
    domain: str
    folder_name: str
    data_root: str = "data/processed"
    log_subdir: str = "logs"
    experiment_name: str = "experiment1"

    def __post_init__(self) -> None:
        data_root = Path(self.data_root)
        log_subdir = Path(self.log_subdir)
        if not data_root.is_absolute():
            self.data_root = str(_PROJECT_ROOT / data_root)
        if not log_subdir.is_absolute():
            self.log_subdir = str(_PROJECT_ROOT / log_subdir)

    @property
    def file_path(self) -> str:
        return f"{self.data_root}/{self.folder_name}"


@dataclass
class LLMConfig:
    provider: str = "ollama"
    base_url: Optional[str] = "http://localhost:11435"
    model: str = "gpt-oss:20b"
    api_key: Optional[str] = None
    temperature: float = 0.0
    retry_temperature: float = 0.1
    top_p: float = 1.0
    seed: Optional[int] = 0
    timeout: int = 20
    max_output_tokens: int = 4096
    max_retries: int = 5
    retry_delay: int = 1
    verbose: bool = True


@dataclass
class EmbeddingConfig:
    model_name: str = "all-mpnet-base-v2"
    batch_size: int = 1024
    device: Optional[str] = None
    similarity_threshold: float = 0.9


@dataclass
class NMFConfig:
    w_recon: float = 0.40
    w_sparsity: float = 0.25
    w_coherence: float = 0.35
    coherence_floor_ratio: Optional[float] = None
    rel_cutoff: float = 0.2
    min_terms: int = 1
    add_singletons: bool = True
    split_threshold: int = 20
    k_nn_cap: int = 50


@dataclass
class TaxonomyConfig:
    paths: PathConfig
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    nmf: NMFConfig = field(default_factory=NMFConfig)
