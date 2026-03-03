"""Feature engineering package.

Public API:
    FeatureEngineer  — main class for building features
    FeatureGroup     — base class for adding new feature groups
    register         — decorator for registering feature groups
    FEATURE_COLUMNS  — list of all registered feature column names
"""

from ml.features.registry import FeatureGroup, register, get_all_feature_columns, get_all_groups, get_feature_group_map

# Import all feature groups to trigger @register decorators.
# The import order does NOT matter — the registry's resolve_order()
# handles dependency ordering at runtime.
from ml.features import (  # noqa: F401
    momentum,
    momentum_academic,
    volume,
    volatility,
    fundamental,
    market,
    sector,
    sector_neutral,
    distress,
    sector_rotation,
    macro_interaction,
)

from ml.features._pipeline import FeatureEngineer

FEATURE_COLUMNS = get_all_feature_columns()
