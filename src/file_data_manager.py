from pathlib import Path
from typing import Literal, Optional, Sequence, Union
from datetime import datetime
import os
import io

import pandas as pd
from pydantic import BaseModel, Field, field_validator

# Try to import storage backend, but make it optional for backward compatibility
try:
    from storage_backend import StorageBackend, StorageConfig, create_storage_backend
    STORAGE_BACKEND_AVAILABLE = True
except ImportError:
    STORAGE_BACKEND_AVAILABLE = False
    StorageBackend = None
    StorageConfig = None
    create_storage_backend = None

class FileConfig(BaseModel):
    """
    Configuration for where and how to read/write files.
    Supports both local filesystem and storage backends (S3, etc.).
    """
    base_dir: Union[Path, str] = Path("data/time_series")
    file_format: Literal["parquet", "csv", "json"] = "parquet"
    
    # Storage backend configuration (optional)
    use_storage_backend: bool = Field(
        default_factory=lambda: os.getenv("USE_STORAGE_BACKEND", "false").lower() == "true",
        description="Use storage backend instead of local filesystem"
    )
    storage_config: Optional[dict] = Field(
        default=None,
        description="Storage backend configuration (dict with StorageConfig parameters)"
    )

    @field_validator("base_dir")
    def ensure_base_dir_exists(cls, v):
        # Only create directory if not using storage backend
        # (storage backend will handle this)
        if isinstance(v, str):
            v = Path(v)
        # Don't create directory here - let storage backend or local storage handle it
        return v

    class Config:
        arbitrary_types_allowed = True

class FilePathHandler:
    """
    Shared helper for resolving file paths.
    Works with both local filesystem and storage backends.
    """
    _EXTENSIONS = {
        "parquet": ".parquet",
        "csv":     ".csv",
        "json":    ".json",
    }

    def __init__(self, config: FileConfig, storage_backend: Optional[StorageBackend] = None):
        self.config = config
        self.storage_backend = storage_backend
        self._use_backend = storage_backend is not None

    def _get_file_key(self, table: str, identifier: str) -> str:
        """Get storage key (path) for a file"""
        if not isinstance(identifier, str) or not identifier:
            raise TypeError("`identifier` must be a non-empty string")
        ext = self._EXTENSIONS.get(self.config.file_format)
        if ext is None:
            raise ValueError(f"Unsupported file format: {self.config.file_format!r}")
        # Use forward slashes for storage keys (works for both local and S3)
        return f"{table}/{identifier}{ext}"

    def _get_folder(self, table: str) -> Path:
        """Get local folder path (only used when not using storage backend)"""
        if not isinstance(table, str) or not table:
            raise TypeError("`table` must be a non-empty string")
        base_dir = Path(self.config.base_dir) if isinstance(self.config.base_dir, str) else self.config.base_dir
        folder = base_dir / table
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _get_file_path(self, table: str, identifier: str) -> Path:
        """Get local file path (only used when not using storage backend)"""
        folder = self._get_folder(table)
        ext = self._EXTENSIONS.get(self.config.file_format)
        if ext is None:
            raise ValueError(f"Unsupported file format: {self.config.file_format!r}")
        return folder / f"{identifier}{ext}"

class DataStore(FilePathHandler):
    """
    Responsible for writing new files and updating existing ones.
    Supports both local filesystem and storage backends.
    """
    def store(
        self,
        df: pd.DataFrame,
        table: str,
        identifier: str,
        update_history: bool = True,
        key_cols: Sequence[str] = ("date", "sid")
    ) -> None:
        # type checking
        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a pandas DataFrame")
        if not isinstance(update_history, bool):
            raise TypeError("`update_history` must be a bool")

        if self._use_backend:
            key = self._get_file_key(table, identifier)
            if update_history:
                self._write_file_backend(df, key)
            else:
                self._update_file_backend(key, df, key_cols)
        else:
            path = self._get_file_path(table, identifier)
            if update_history:
                self._write_file_local(df, path)
            else:
                self._update_file_local(path, df, key_cols)

    def _write_file_local(self, df: pd.DataFrame, path: Path) -> None:
        """Write file to local filesystem"""
        fmt = self.config.file_format
        if fmt == "parquet":
            df.to_parquet(path, index=False)
        elif fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "json":
            df.to_json(path, orient="records", date_format="iso")
        else:
            raise ValueError(f"Unsupported format {fmt!r}")

    def _write_file_backend(self, df: pd.DataFrame, key: str) -> None:
        """Write file to storage backend"""
        fmt = self.config.file_format
        
        if fmt == "parquet":
            buffer = io.BytesIO()
            try:
                df.to_parquet(buffer, index=False, engine='pyarrow')
                buffer.seek(0)
                data = buffer.read()
                self.storage_backend.write(key, data)
            finally:
                buffer.close()
        elif fmt == "csv":
            # For CSV, we need to use text mode
            buffer = io.StringIO()
            try:
                df.to_csv(buffer, index=False)
                data = buffer.getvalue().encode('utf-8')
                self.storage_backend.write(key, data)
            finally:
                buffer.close()
        elif fmt == "json":
            buffer = io.StringIO()
            try:
                df.to_json(buffer, orient="records", date_format="iso")
                data = buffer.getvalue().encode('utf-8')
                self.storage_backend.write(key, data)
            finally:
                buffer.close()
        else:
            raise ValueError(f"Unsupported format {fmt!r}")

    def _update_file_local(
        self,
        path: Path,
        df_new: pd.DataFrame,
        key_cols: Sequence[str]
    ) -> None:
        """Update file on local filesystem"""
        # load existing if present
        if path.exists():
            df_existing = self._read_file_local(path)

            # incremental by 'date' column if present
            if "date" in df_existing.columns and "date" in df_new.columns:
                last_date = (df_existing["date"]).max()
                if last_date in list(df_new.date.unique()):
                    df_existing = df_existing[df_existing["date"] < last_date]
                df_new = df_new[df_new["date"] >= last_date]

            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        # sort & dedupe
        if set(key_cols).issubset(df_combined.columns):
            df_combined.sort_values(list(key_cols), inplace=True)
            if 'exposure' not in df_combined.columns:
                df_combined.drop_duplicates(subset=key_cols, inplace=True)

        # write back
        self._write_file_local(df_combined, path)

    def _update_file_backend(
        self,
        key: str,
        df_new: pd.DataFrame,
        key_cols: Sequence[str]
    ) -> None:
        """Update file in storage backend"""
        # load existing if present
        if self.storage_backend.exists(key):
            df_existing = self._read_file_backend(key)

            # incremental by 'date' column if present
            if "date" in df_existing.columns and "date" in df_new.columns:
                last_date = (df_existing["date"]).max()
                if last_date in list(df_new.date.unique()):
                    df_existing = df_existing[df_existing["date"] < last_date]
                df_new = df_new[df_new["date"] >= last_date]

            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        # sort & dedupe
        if set(key_cols).issubset(df_combined.columns):
            df_combined.sort_values(list(key_cols), inplace=True)
            if 'exposure' not in df_combined.columns:
                df_combined.drop_duplicates(subset=key_cols, inplace=True)

        # write back
        self._write_file_backend(df_combined, key)

    def _read_file_local(self, path: Path) -> pd.DataFrame:
        """Read file from local filesystem"""
        fmt = self.config.file_format
        if fmt == "parquet":
            return pd.read_parquet(path)
        elif fmt == "csv":
            return pd.read_csv(path)
        elif fmt == "json":
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported format {fmt!r}")

    def _read_file_backend(self, key: str) -> pd.DataFrame:
        """Read file from storage backend"""
        data = self.storage_backend.read(key)
        fmt = self.config.file_format
        buffer = io.BytesIO(data)
        
        try:
            if fmt == "parquet":
                return pd.read_parquet(buffer)
            elif fmt == "csv":
                # For CSV, decode from bytes
                buffer = io.StringIO(data.decode('utf-8'))
                return pd.read_csv(buffer)
            elif fmt == "json":
                buffer = io.StringIO(data.decode('utf-8'))
                return pd.read_json(buffer)
            else:
                raise ValueError(f"Unsupported format {fmt!r}")
        finally:
            buffer.close()

    def get_last_date(
        self,
        table: str,
        identifier: str,
        date_col: str = "date"
    ) -> Optional[datetime]:
        """
        Return the max(date_col) in the existing file, or None if no file.
        """
        if self._use_backend:
            key = self._get_file_key(table, identifier)
            if not self.storage_backend.exists(key):
                return None
            df = self._read_file_backend(key)
        else:
            path = self._get_file_path(table, identifier)
            if not path.exists():
                return None
            df = self._read_file_local(path)
        
        if date_col not in df.columns:
            raise KeyError(f"Column {date_col!r} not found in file")
        return pd.to_datetime(df[date_col]).max()

class DataLoader(FilePathHandler):
    """
    Responsible for reading files from disk or storage backend.
    """
    def load(self, table: str, identifier: str) -> pd.DataFrame:
        if self._use_backend:
            key = self._get_file_key(table, identifier)
            if not self.storage_backend.exists(key):
                raise FileNotFoundError(f"Data file not found: {key}")
            return self._read_file_backend(key)
        else:
            path = self._get_file_path(table, identifier)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
            return self._read_file_local(path)
    
    def _read_file_local(self, path: Path) -> pd.DataFrame:
        """Read file from local filesystem"""
        fmt = self.config.file_format
        if fmt == "parquet":
            return pd.read_parquet(path)
        elif fmt == "csv":
            return pd.read_csv(path)
        elif fmt == "json":
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported format {fmt!r}")
    
    def _read_file_backend(self, key: str) -> pd.DataFrame:
        """Read file from storage backend"""
        data = self.storage_backend.read(key)
        fmt = self.config.file_format
        buffer = io.BytesIO(data)
        
        try:
            if fmt == "parquet":
                return pd.read_parquet(buffer)
            elif fmt == "csv":
                buffer = io.StringIO(data.decode('utf-8'))
                return pd.read_csv(buffer)
            elif fmt == "json":
                buffer = io.StringIO(data.decode('utf-8'))
                return pd.read_json(buffer)
            else:
                raise ValueError(f"Unsupported format {fmt!r}")
        finally:
            buffer.close()

class FileDataManager:
    """
    High-level API combining DataStore and DataLoader,
    with the same convenience methods you had before.
    Supports both local filesystem and storage backends (S3, etc.).
    """
    def __init__(self, config: Optional[FileConfig] = None):
        self.config = config or FileConfig()
        
        # Initialize storage backend if configured
        storage_backend = None
        if self.config.use_storage_backend:
            if not STORAGE_BACKEND_AVAILABLE:
                raise ImportError(
                    "Storage backend requested but not available. "
                    "Make sure storage_backend module is available and dependencies (boto3) are installed."
                )
            
            # Create storage config from FileConfig
            storage_config_dict = self.config.storage_config or {}
            storage_config_dict.setdefault("storage_type", os.getenv("STORAGE_TYPE", "auto"))
            storage_config_dict.setdefault("base_path", str(self.config.base_dir))
            storage_config_dict.setdefault("s3_bucket", os.getenv("S3_BUCKET"))
            storage_config_dict.setdefault("s3_region", os.getenv("AWS_REGION", "us-east-1"))
            storage_config_dict.setdefault("enable_cache", os.getenv("ENABLE_CACHE", "true").lower() == "true")
            storage_config_dict.setdefault("cache_dir", os.getenv("CACHE_DIR", ".cache"))
            
            storage_config = StorageConfig(**storage_config_dict)
            storage_backend = create_storage_backend(storage_config)
        
        self._store = DataStore(self.config, storage_backend)
        self._load = DataLoader(self.config, storage_backend)

    # exposures
    def store_exposures(self, df: pd.DataFrame, identifier: str, update_history: bool = True):
        self._store.store(df, "exposures", identifier, update_history)

    def load_exposures(self, identifier: str) -> pd.DataFrame:
        return self._load.load("exposures", identifier)

    # benchmark weights
    def store_benchmark_weights(self, df: pd.DataFrame, identifier: str, update_history: bool = True):
        self._store.store(df, "benchmarks", f"{identifier}_weights", update_history)

    def load_benchmark_weights(self, identifier: str) -> pd.DataFrame:
        return self._load.load("benchmarks", f"{identifier}_weights")

    # factors
    def store_factors(self, df: pd.DataFrame, identifier: str, update_history: bool = True):
        self._store.store(df, "factors", identifier, update_history)

    def load_factors(self, identifier: str) -> pd.DataFrame:
        return self._load.load("factors", identifier)

    # prices
    def store_prices(self, df: pd.DataFrame, identifier: str, update_history: bool = True):
        self._store.store(df, "prices", identifier, update_history)

    def load_prices(self, identifier: str) -> pd.DataFrame:
        return self._load.load("prices", identifier)

    # returns
    def store_returns(self, df: pd.DataFrame, identifier: str, update_history: bool = True):
        self._store.store(df, "returns", identifier, update_history)

    def load_returns(self, identifier: str) -> pd.DataFrame:
        return self._load.load("returns", identifier)

    # last-date helper
    def get_last_date(self, table: str, identifier: str, date_col: str = "date"):
        return self._store.get_last_date(table, identifier, date_col)

# usage
if __name__=="__main__":
    cfg = FileConfig(base_dir="my_data", file_format="csv")
    mgr = FileDataManager(cfg)

