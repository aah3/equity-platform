import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
from pathlib import Path
from src.file_data_manager_v1 import FileDataManager

logger = logging.getLogger(__name__)

class FileDataIntegration:
    """
    Integration class between qFactor framework and file-based storage (parquet files)
    Mimics the API of FactorDatabaseIntegration but uses FileDataManager.
    """
    def __init__(self, file_manager: FileDataManager):
        self.file_manager = file_manager

    def store_model_input(self, model_input, config_id: str) -> str:
        """Store model input configuration as a parquet or json file (optional)."""
        # Optionally, save model_input as a json or pickle file
        # Not implemented for now, just return config_id
        logger.info(f"Model input stored with config_id: {config_id}")
        return config_id

    def store_security_master_data(self, security_master, config_id: str) -> None:
        """Store security master data (e.g., prices, returns) as parquet files."""
        if hasattr(security_master, 'df_price') and security_master.df_price is not None:
            self.file_manager.store_prices(security_master.df_price, config_id)
            logger.info(f"Stored prices for config_id: {config_id}")
        if hasattr(security_master, 'get_returns_long'):
            returns_df = security_master.get_returns_long()
            if returns_df is not None:
                self.file_manager.store_returns(returns_df, config_id)
                logger.info(f"Stored returns for config_id: {config_id}")

    def store_factor_data(self, factor_dict: Dict, config_id: str) -> None:
        """Store factor data as parquet files."""
        for factor_name, factor_data in factor_dict.items():
            factor_obj = factor_data.get('factor_eq')
            if factor_obj:
                df = factor_obj.data.copy()
                self.file_manager.store_factors(df, f"{config_id}_{factor_name}")
                logger.info(f"Stored factor {factor_name} for config_id: {config_id}")

    def store_exposures(self, exposures_df: pd.DataFrame, config_id: str) -> None:
        """Store exposures as parquet file."""
        self.file_manager.store_exposures(exposures_df, config_id)
        logger.info(f"Stored exposures for config_id: {config_id}")

    def load_exposures(self, config_id: str) -> pd.DataFrame:
        """Load exposures from parquet file."""
        return self.file_manager.load_exposures(config_id)

    def store_benchmark_weights(self, df: pd.DataFrame, config_id: str) -> None:
        """Store benchmark weights weights as parquet file."""
        self.file_manager.store_benchmark_weights(df, config_id)
        logger.info(f"Stored benchmark weights for config_id: {config_id}")

    def load_benchmark_weights(self, config_id: str) -> pd.DataFrame:
        """Load benchmark weights from parquet file."""
        return self.file_manager.load_benchmark_weights(config_id)

    def store_portfolio_results(self, config_id: str, weights_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
        """Store portfolio weights and results as parquet files."""
        if weights_df is not None:
            self.file_manager.store_factors(weights_df, f"{config_id}_weights")
            logger.info(f"Stored portfolio weights for config_id: {config_id}")
        if results_df is not None:
            self.file_manager.store_factors(results_df, f"{config_id}_results")
            logger.info(f"Stored portfolio results for config_id: {config_id}")

    def load_factor_model_data(self, config_id: str, factor_names: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """Load returns and factor data for a given config_id and factor_names."""
        returns_df = self.file_manager.load_returns(config_id)
        factor_dict = {}
        for factor_name in factor_names:
            try:
                df = self.file_manager.load_factors(f"{config_id}_{factor_name}")
                factor_dict[factor_name] = {'data': df}
            except FileNotFoundError:
                logger.warning(f"Factor {factor_name} not found for config_id: {config_id}")
        return returns_df, factor_dict 