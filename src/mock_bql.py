"""
Mock BQL (Bloomberg Query Language) Module - FIXED VERSION
===========================================================
Fixes "Service is not a known attribute of module" error.
Complete mock implementation for development without Bloomberg terminal.

Usage in your qFactor.py:
    try:
        import bql
    except ImportError:
        import mock_bql as bql
"""

from typing import Dict, Any, Optional, List, Union, Callable
import warnings
from datetime import datetime, date
import random


class MockBQLData:
    """Mock BQL data operations."""
    
    def name(self, *args, **kwargs) -> Dict[str, Any]:
        return {'VALUE': 'MOCK_NAME', 'ID': 'MOCK_ID', 'FIELD': 'NAME', 'ERROR': None}
    
    def id(self, *args, **kwargs) -> Dict[str, Any]:
        return {
            'WEIGHTS': [0.25, 0.25, 0.25, 0.25],
            'ID': ['MOCK_ID_1', 'MOCK_ID_2', 'MOCK_ID_3', 'MOCK_ID_4'],
            'VALUE': 'MOCK_VALUE',
            'ERROR': None
        }
    
    def px_last(self, *args, **kwargs) -> Dict[str, Any]:
        return {'VALUE': 100.0 + random.random() * 50, 'DATE': date.today(), 'ERROR': None}
    
    def px_volume(self, *args, **kwargs) -> Dict[str, Any]:
        return {'VALUE': random.randint(1000000, 10000000), 'DATE': date.today(), 'ERROR': None}
    
    def cur_mkt_cap(self, *args, **kwargs) -> Dict[str, Any]:
        return {'VALUE': random.randint(1000000000, 100000000000), 'CURRENCY': 'USD', 'ERROR': None}
    
    def beta(self, *args, **kwargs) -> Dict[str, Any]:
        return {'VALUE': 0.8 + random.random() * 0.4, 'PERIOD': '1Y', 'ERROR': None}
    
    def __getattr__(self, name: str) -> Callable:
        """Dynamic field methods."""
        def mock_method(*args, **kwargs) -> Dict[str, Any]:
            return {'VALUE': f'MOCK_{name.upper()}_VALUE', 'ERROR': None}
        return mock_method


class MockBQLMembers:
    """Mock BQL members operations."""
    
    def __call__(self, universe: Any = None, *args, **kwargs) -> List[str]:
        return [f'TICKER_{i}' for i in range(1, 11)] if universe else []
    
    def list(self, *args, **kwargs) -> List[str]:
        return self(*args, **kwargs)
    
    def count(self, *args, **kwargs) -> int:
        return len(self(*args, **kwargs))


class MockBQLFunction:
    """Mock BQL function operations."""
    
    def __call__(self, func_name: str, *args, **kwargs) -> Dict[str, Any]:
        functions = {
            'returns': lambda: {'VALUE': [0.01 * i for i in range(252)]},
            'volatility': lambda: {'VALUE': 0.15 + random.random() * 0.1},
            'sharpe': lambda: {'VALUE': 1.2 + random.random() * 0.5},
        }
        func = functions.get(func_name.lower(), lambda: {'VALUE': None, 'ERROR': 'FUNCTION_NOT_FOUND'})
        return func()
    
    def __getattr__(self, name: str) -> Callable:
        def mock_function(*args, **kwargs) -> Dict[str, Any]:
            return self(name, *args, **kwargs)
        return mock_function


class MockBQLResult:
    """Mock BQL query results."""
    
    def __init__(self, data: Optional[Any] = None, success: bool = True):
        self.data = data if data is not None else {}
        self.success = success
        self.error = None if success else "Mock error"
    
    def empty(self) -> bool:
        return self.data is None or not self.data
    
    def values(self) -> List[Any]:
        return list(self.data.values()) if isinstance(self.data, dict) else []


class Request:
    """Mock BQL Request for building queries."""
    
    def __init__(self, universe: Any = None, items: Any = None, dates: Any = None):
        self.universe = universe
        self.items = items
        self.dates = dates
    
    def with_dates(self, dates: Any) -> 'Request':
        self.dates = dates
        return self
    
    def with_items(self, items: Any) -> 'Request':
        self.items = items
        return self
    
    def with_universe(self, universe: Any) -> 'Request':
        self.universe = universe
        return self
    
    def _generate_mock_data(self) -> Dict[str, Any]:
        if isinstance(self.items, dict):
            result = {}
            for key, item in self.items.items():
                if hasattr(item, '__call__'):
                    result[key] = item()
                else:
                    result[key] = {'VALUE': f'MOCK_{key}_DATA'}
            return result
        elif isinstance(self.items, list):
            return {item: {'VALUE': f'MOCK_{item}_DATA'} for item in self.items}
        else:
            return {'data': 'MOCK_REQUEST_DATA'}


class Service_v0:
    """
    Mock BQL Service - Main class that mimics Bloomberg Query Language Service.
    This fixes the "Service is not a known attribute" error.
    """
    
    def __init__(self, preferences: Optional[Dict[str, Any]] = None):
        """Initialize mock BQL Service with all required attributes."""
        self.preferences = preferences or {}
        self.data = MockBQLData()
        self.members = MockBQLMembers()
        self.function = MockBQLFunction()
        self._last_request = None
        
        warnings.warn(
            "Using mock BQL service. Replace with actual Bloomberg BQL when available.",
            UserWarning, 
            stacklevel=2
        )
    
    def execute(self, request: Union[Request, Any]) -> MockBQLResult:
        """Execute a BQL request and return results."""
        self._last_request = request
        
        if isinstance(request, Request):
            mock_data = request._generate_mock_data()
            return MockBQLResult(data=mock_data, success=True)
        else:
            return MockBQLResult(data={'VALUE': 'MOCK_EXECUTE_RESULT'}, success=True)
    
    def query(self, *args, **kwargs) -> MockBQLResult:
        """Execute a query with arguments."""
        return MockBQLResult(data={'VALUE': 'MOCK_QUERY_RESULT'}, success=True)
    
    def bulk(self, *args, **kwargs) -> MockBQLResult:
        """Execute bulk operations."""
        return MockBQLResult(data={'VALUE': 'MOCK_BULK_RESULT'}, success=True)
    
    def universe(self, identifier: str) -> List[str]:
        """Get universe members."""
        return self.members(identifier)

class Service:
    """Mock BQL Service class - Main entry point for BQL operations."""
    
    def __init__(self, preferences: Optional[Dict[str, Any]] = None):
        """Initialize the mock BQL Service."""
        self.preferences = preferences or {}
        # The following are now attributes referencing the mock classes/methods, 
        # but they fulfill the intent of providing access to data, members, and functions.
        # Warnings are kept from the original code.
        warnings.warn("Using mock BQL service. Replace with actual Bloomberg BQL when available.", 
                     UserWarning, stacklevel=2)

    # --- Requested Access Methods ---

    def data(self) -> MockBQLData:
        """Returns the mock BQL data object (equivalent to bq.data)."""
        return MockBQLData()

    def func(self) -> MockBQLFunction:
        """Returns the mock BQL function object (equivalent to bq.function)."""
        return MockBQLFunction()

    # The original implementation uses 'members', which is the equivalent of 'univ' 
    # (universe/member access) in a typical BQL workflow.
    def univ(self) -> MockBQLMembers:
        """Returns the mock BQL members/universe object (equivalent to bq.members)."""
        return MockBQLMembers()
    
    def execute(self, request: Any) -> MockBQLResult:
        """Execute a BQL request."""
        if isinstance(request, Request):
            return MockBQLResult(data=request._generate_mock_data())
        return MockBQLResult(data={'VALUE': 'MOCK_EXECUTE_RESULT'})
    
    # Keeping the original query method for completeness
    def query(self, *args, **kwargs) -> MockBQLResult:
        """Execute a query."""
        return MockBQLResult(data={'VALUE': 'MOCK_QUERY_RESULT'})
        
# Module-level convenience functions
def request(*args, **kwargs) -> MockBQLResult:
    """Mock request function."""
    return MockBQLResult(data={'VALUE': 'MOCK_REQUEST_FUNCTION'})


def bulk(*args, **kwargs) -> MockBQLResult:
    """Mock bulk request function."""
    return MockBQLResult(data={'VALUE': 'MOCK_BULK_FUNCTION'})


# CRITICAL: Explicitly define __all__ to ensure proper exports
__all__ = [
    'Service',      # Main class - MUST be first
    'Request', 
    'MockBQLResult',
    'MockBQLData',
    'MockBQLMembers',
    'MockBQLFunction',
    'request',
    'bulk',
]

# CRITICAL: Ensure Service is accessible at module level
# This fixes the "Service is not a known attribute" error
if not hasattr(__builtins__, '__IPYTHON__'):
    # Standard Python environment
    globals()['Service'] = Service
    globals()['Request'] = Request
    globals()['MockBQLResult'] = MockBQLResult