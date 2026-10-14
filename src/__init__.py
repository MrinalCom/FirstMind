"""FirstMind Enterprise Platform - Source Package"""

# Expose primary modules and classes at package root
try:
    from src.environment.reward_engine import RewardEngine
    from src.environment.rag_simulator import RAGSimulator
    from src.environment.geo_env import GeoBrandEnv
except ImportError:
    from src.reward_engine import RewardEngine
    from src.rag_simulator import RAGSimulator
    from src.geo_env import GeoBrandEnv

try:
    from src.evaluator import GEOEvaluator
except ImportError:
    GEOEvaluator = None

__all__ = [
    "RewardEngine",
    "RAGSimulator",
    "GeoBrandEnv",
    "GEOEvaluator",
]
