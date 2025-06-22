from pathlib import Path
from typing import Literal, Optional, Sequence
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, field_validator

class FileConfig(BaseModel):
    """
    Configuration for where and how to read/write files.
    """
    base_dir: Path = Path("data/time_series")
    file_format: Literal["parquet", "csv", "json"] = "parquet"

    @field_validator("base_dir")
    def ensure_base_dir_exists(cls, v):
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p

    class Config:
        arbitrary_types_allowed = True

class FilePathHandler:
    """
    Shared helper for resolving file paths.
    """
    _EXTENSIONS = {
        "parquet": ".parquet",
        "csv":     ".csv",
        "json":    ".json",
    }

    def __init__(self, config: FileConfig):
        self.config = config

    def _get_folder(self, table: str) -> Path:
        if not isinstance(table, str) or not table:
            raise TypeError("`table` must be a non-empty string")
        folder = self.config.base_dir / table
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _get_file_path(self, table: str, identifier: str) -> Path:
        if not isinstance(identifier, str) or not identifier:
            raise TypeError("`identifier` must be a non-empty string")
        folder = self._get_folder(table)
        ext = self._EXTENSIONS.get(self.config.file_format)
        if ext is None:
            raise ValueError(f"Unsupported file format: {self.config.file_format!r}")
        return folder / f"{identifier}{ext}"

class DataStore(FilePathHandler):
    """
    Responsible for writing new files and updating existing ones.
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

        path = self._get_file_path(table, identifier)
        if update_history:
            self._write_file(df, path)
        else:
            self._update_file(path, df, key_cols)

    def _write_file(self, df: pd.DataFrame, path: Path) -> None:
        fmt = self.config.file_format
        if fmt == "parquet":
            df.to_parquet(path, index=False)
        elif fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "json":
            df.to_json(path, orient="records", date_format="iso")
        else:
            # this should never happen because FileConfig field_validator guards it
            raise ValueError(f"Unsupported format {fmt!r}")

    def _update_file(
        self,
        path: Path,
        df_new: pd.DataFrame,
        key_cols: Sequence[str]
    ) -> None:
        # load existing if present
        if path.exists():
            df_existing = self._read_file(path)

            # incremental by 'date' column if present
            if "date" in df_existing.columns and "date" in df_new.columns:
                # last_date = pd.to_datetime(df_existing["date"]).max().date()
                last_date = (df_existing["date"]).max()
                if last_date in list(df_new.date.unique()):
                    df_existing = df_existing[df_existing["date"] < last_date]
                df_new = df_new[df_new["date"] >= last_date]
                # df_new = df_new[pd.to_datetime(df_new["date"]) >= last_date]

            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        # sort & dedupe
        if set(key_cols).issubset(df_combined.columns):
            df_combined.sort_values(list(key_cols), inplace=True)
            df_combined.drop_duplicates(subset=key_cols, inplace=True)

        # write back
        self._write_file(df_combined, path)

    def _read_file(self, path: Path) -> pd.DataFrame:
        # helper for update only
        fmt = self.config.file_format
        if fmt == "parquet":
            return pd.read_parquet(path)
        elif fmt == "csv":
            return pd.read_csv(path)
        elif fmt == "json":
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported format {fmt!r}")

    def get_last_date(
        self,
        table: str,
        identifier: str,
        date_col: str = "date"
    ) -> Optional[datetime]:
        """
        Return the max(date_col) in the existing file, or None if no file.
        """
        path = self._get_file_path(table, identifier)
        if not path.exists():
            return None

        df = self._read_file(path)
        if date_col not in df.columns:
            raise KeyError(f"Column {date_col!r} not found in {path}")
        return pd.to_datetime(df[date_col]).max()

class DataLoader(FilePathHandler):
    """
    Responsible for reading files from disk.
    """
    def load(self, table: str, identifier: str) -> pd.DataFrame:
        path = self._get_file_path(table, identifier)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        fmt = self.config.file_format
        if fmt == "parquet":
            return pd.read_parquet(path)
        elif fmt == "csv":
            return pd.read_csv(path)
        elif fmt == "json":
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported format {fmt!r}")

class FileDataManager:
    """
    High-level API combining DataStore and DataLoader,
    with the same convenience methods you had before.
    """
    def __init__(self, config: Optional[FileConfig] = None):
        self.config = config or FileConfig()
        self._store = DataStore(self.config)
        self._load = DataLoader(self.config)

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

