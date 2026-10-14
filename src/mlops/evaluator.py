r"""GEO Benchmark Evaluator for Generative Engine Optimization.

This module evaluates policy performance and baseline agents against holdout enterprise
buyer search queries. It quantifies AI search visibility using industry-standard
GEO metrics: Answer Presence Rate (APR), Share of Model (SoM), and Average Positional Rank.

================================================================================
MATHEMATICAL FORMULATIONS
================================================================================
1. Answer Presence Rate (APR):
   Measures the empirical probability that the brand is cited or recommended
   in the AI search engine's zero-click synthesized answer:

       APR = (1 / N) * \sum_{i=1}^N \mathbb{I}(\text{brand} \in \text{response}_i)

   where N is the total number of evaluation test queries and \mathbb{I} is the
   indicator function.

2. Share of Model (SoM):
   Analogous to Share of Voice (SoV) in traditional search marketing, SoM quantifies
   the proportional share of brand citations relative to all direct competitors:

       SoM = \frac{\text{Count}(\text{brand})}{\text{Count}(\text{brand}) + \sum_{c \in \mathcal{C}} \text{Count}(c)}

   where \mathcal{C} represents the set of recognized enterprise competitors
   (e.g., Vanta, Drata, Secureframe).

3. Average Rank Position (\overline{Rank}):
   The mean positional ranking across queries where the brand is recommended:

       \overline{\text{Rank}} = \frac{1}{|M|} \sum_{i \in M} \text{Rank}_i(\text{brand})

   where M = \{i \mid \text{brand} \in \text{response}_i\} is the subset of responses
   containing the brand.
================================================================================
"""

from __future__ import annotations

import logging
import random
import re
import statistics
from typing import Any, Callable, Dict, List, Optional

try:
    import numpy as np
except ImportError:
    np = None  # Handled gracefully if numpy is not installed

from src.environment.rag_simulator import RAGSimulator
from src.environment.reward_engine import RewardEngine
from src.mlops.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class GEOEvaluator:
    """Benchmark evaluator for FirstMind policies and search optimization baselines."""

    DEFAULT_COMPETITORS = ["Vanta", "Drata", "Secureframe"]

    def __init__(
        self,
        rag_simulator: RAGSimulator,
        reward_engine: RewardEngine,
        brand_name: str = "SecureFlow",
        competitors: Optional[List[str]] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            rag_simulator: RAGSimulator instance.
            reward_engine: RewardEngine instance for scoring and rank detection.
            brand_name: Target brand name (default: 'SecureFlow').
            competitors: List of competitor brand names.
        """
        self.rag_simulator = rag_simulator
        self.reward_engine = reward_engine
        self.brand_name = brand_name
        self.competitors = competitors or list(self.DEFAULT_COMPETITORS)

    def compute_apr(self, responses: List[str], brand_name: Optional[str] = None) -> float:
        r"""Compute Answer Presence Rate (APR).

        Formula:
            APR = (1 / N) * \sum_{i=1}^N \mathbb{I}(brand \in response_i)

        Args:
            responses: List of synthesized LLM response strings.
            brand_name: Target brand name (default: self.brand_name).

        Returns:
            Float in [0.0, 1.0] representing presence rate.
        """
        if not responses:
            return 0.0

        target = brand_name or self.brand_name
        pattern = rf"\b{re.escape(target)}\b"
        mentions = sum(
            1 for resp in responses if bool(re.search(pattern, resp, re.IGNORECASE))
        )
        return float(mentions / len(responses))

    def compute_som(
        self,
        responses: List[str],
        brand_name: Optional[str] = None,
        competitor_names: Optional[List[str]] = None,
    ) -> float:
        r"""Compute Share of Model (SoM).

        Formula:
            SoM = Count(brand) / (Count(brand) + \sum_{c \in C} Count(c))

        Args:
            responses: List of synthesized LLM response strings.
            brand_name: Target brand name.
            competitor_names: List of competitor brand names.

        Returns:
            Float in [0.0, 1.0] representing share of citations.
        """
        if not responses:
            return 0.0

        target = brand_name or self.brand_name
        comps = competitor_names or self.competitors

        target_pattern = rf"\b{re.escape(target)}\b"
        target_count = 0
        comp_count = 0

        for resp in responses:
            target_count += len(re.findall(target_pattern, resp, re.IGNORECASE))
            for comp in comps:
                comp_pattern = rf"\b{re.escape(comp)}\b"
                comp_count += len(re.findall(comp_pattern, resp, re.IGNORECASE))

        total_mentions = target_count + comp_count
        if total_mentions == 0:
            return 0.0

        return float(target_count / total_mentions)

    def compute_avg_rank(
        self,
        responses: List[str],
        brand_name: Optional[str] = None,
    ) -> float:
        r"""Compute Average Positional Rank across queries where brand appears.

        Formula:
            \overline{Rank} = (1 / |M|) * \sum_{i \in M} Rank_i(brand)

        Args:
            responses: List of synthesized LLM response strings.
            brand_name: Target brand name.

        Returns:
            Float representing mean rank (1.0 = best possible).
            Returns 0.0 if never ranked.
        """
        target = brand_name or self.brand_name
        ranks: List[int] = []

        for resp in responses:
            rank = self.reward_engine._detect_rank(resp, target)
            if rank > 0:
                ranks.append(rank)

        if not ranks:
            return 0.0

        if np is not None:
            return float(np.mean(ranks))
        return float(sum(ranks) / len(ranks))

    def evaluate_random_agent(
        self,
        queries: List[str],
        brand_profile: Dict[str, Any],
        n_episodes: int = 50,
    ) -> Dict[str, Any]:
        """Evaluate a random baseline agent across evaluation queries.

        Args:
            queries: Holdout benchmark test queries.
            brand_profile: Base brand profile dictionary.
            n_episodes: Number of episodes to run (default: 50).

        Returns:
            Dictionary with aggregated benchmark metrics.
        """
        from src.environment.geo_env import GeoBrandEnv

        competitor_profiles = [
            {"brand": c, "content": f"{c} enterprise SOC 2 compliance automation platform.", "url": f"https://{c.lower()}.com"}
            for c in self.competitors
        ]

        env = GeoBrandEnv(
            brand_profile=brand_profile,
            competitor_profiles=competitor_profiles,
            queries=queries,
            rag_simulator=self.rag_simulator,
            reward_engine=self.reward_engine,
        )

        responses: List[str] = []
        episode_rewards: List[float] = []
        terminal_ranks: List[int] = []
        rank_1_successes = 0

        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            total_ep_reward = 0.0

            while not done:
                if np is not None:
                    random_action = {
                        "content_type": int(np.random.randint(0, 5)),
                        "tone_adjustment": int(np.random.randint(0, 3)),
                        "target_entity": int(np.random.randint(0, 4)),
                    }
                else:
                    random_action = {
                        "content_type": random.randint(0, 4),
                        "tone_adjustment": random.randint(0, 2),
                        "target_entity": random.randint(0, 3),
                    }
                obs, reward, terminated, truncated, step_info = env.step(random_action)
                total_ep_reward += reward
                done = terminated or truncated
                if done:
                    responses.append(step_info["llm_response"])
                    terminal_ranks.append(step_info["current_rank"])
                    if step_info["current_rank"] == 1:
                        rank_1_successes += 1

            episode_rewards.append(total_ep_reward)

        apr = self.compute_apr(responses)
        som = self.compute_som(responses)
        avg_rank = self.compute_avg_rank(responses)
        win_rate = float(rank_1_successes / max(n_episodes, 1))

        if np is not None:
            mean_r = float(np.mean(episode_rewards))
            std_r = float(np.std(episode_rewards))
        else:
            mean_r = float(sum(episode_rewards) / max(len(episode_rewards), 1))
            std_r = float(statistics.stdev(episode_rewards)) if len(episode_rewards) > 1 else 0.0

        return {
            "agent_type": "random_baseline",
            "episodes": n_episodes,
            "apr": round(apr, 4),
            "som": round(som, 4),
            "avg_rank": round(avg_rank, 2),
            "win_rate_rank_1": round(win_rate, 4),
            "mean_reward": round(mean_r, 4),
            "std_reward": round(std_r, 4),
        }

    def evaluate_policy(
        self,
        policy_fn: Callable[[Dict[str, Any]], Dict[str, int]],
        queries: List[str],
        brand_profile: Dict[str, Any],
        n_episodes: int = 50,
    ) -> Dict[str, Any]:
        """Evaluate a trained RL policy function across evaluation queries.

        Args:
            policy_fn: Callable mapping observation dictionary to discrete action dict.
            queries: Holdout benchmark test queries.
            brand_profile: Base brand profile dictionary.
            n_episodes: Number of episodes to run (default: 50).

        Returns:
            Dictionary with aggregated benchmark metrics.
        """
        from src.environment.geo_env import GeoBrandEnv

        competitor_profiles = [
            {"brand": c, "content": f"{c} enterprise SOC 2 compliance automation platform.", "url": f"https://{c.lower()}.com"}
            for c in self.competitors
        ]

        env = GeoBrandEnv(
            brand_profile=brand_profile,
            competitor_profiles=competitor_profiles,
            queries=queries,
            rag_simulator=self.rag_simulator,
            reward_engine=self.reward_engine,
        )

        responses: List[str] = []
        episode_rewards: List[float] = []
        terminal_ranks: List[int] = []
        rank_1_successes = 0

        for _ in range(n_episodes):
            obs, info = env.reset()
            done = False
            total_ep_reward = 0.0

            while not done:
                action = policy_fn(obs)
                obs, reward, terminated, truncated, step_info = env.step(action)
                total_ep_reward += reward
                done = terminated or truncated
                if done:
                    responses.append(step_info["llm_response"])
                    terminal_ranks.append(step_info["current_rank"])
                    if step_info["current_rank"] == 1:
                        rank_1_successes += 1

            episode_rewards.append(total_ep_reward)

        apr = self.compute_apr(responses)
        som = self.compute_som(responses)
        avg_rank = self.compute_avg_rank(responses)
        win_rate = float(rank_1_successes / max(n_episodes, 1))

        if np is not None:
            mean_r = float(np.mean(episode_rewards))
            std_r = float(np.std(episode_rewards))
        else:
            mean_r = float(sum(episode_rewards) / max(len(episode_rewards), 1))
            std_r = float(statistics.stdev(episode_rewards)) if len(episode_rewards) > 1 else 0.0

        return {
            "agent_type": "trained_geo_policy",
            "episodes": n_episodes,
            "apr": round(apr, 4),
            "som": round(som, 4),
            "avg_rank": round(avg_rank, 2),
            "win_rate_rank_1": round(win_rate, 4),
            "mean_reward": round(mean_r, 4),
            "std_reward": round(std_r, 4),
        }

    def generate_report(
        self,
        results: Dict[str, Any],
        tracker: Optional[ExperimentTracker] = None,
    ) -> str:
        """Format and print an executive benchmark evaluation report and log to tracker.

        Args:
            results: Evaluation metrics dictionary.
            tracker: Optional ExperimentTracker instance.

        Returns:
            Formatted report string.
        """
        report_lines = [
            "\n" + "=" * 60,
            f"  GEO BENCHMARK REPORT: {results.get('agent_type', 'Agent').upper()}",
            "=" * 60,
            f"  Evaluated Episodes:         {results.get('episodes', 0)}",
            f"  Answer Presence Rate (APR): {results.get('apr', 0.0) * 100:.2f}%",
            f"  Share of Model (SoM):       {results.get('som', 0.0) * 100:.2f}%",
            f"  Average Rank Position:      #{results.get('avg_rank', 0.0):.2f}",
            f"  Rank #1 Win Rate:           {results.get('win_rate_rank_1', 0.0) * 100:.2f}%",
            f"  Mean Cumulative Reward:     {results.get('mean_reward', 0.0):.4f} (+/- {results.get('std_reward', 0.0):.4f})",
            "=" * 60 + "\n",
        ]
        report_str = "\n".join(report_lines)
        print(report_str)

        if tracker is not None:
            metric_payload = {
                f"eval/{k}": v
                for k, v in results.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            tracker.log_metrics(metric_payload, step=0)

        return report_str
