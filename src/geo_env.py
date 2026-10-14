"""
Gymnasium Environment for Generative Engine Optimization (GEO).

Models the optimization of digital web content to maximize brand visibility,
ranking, and sentiment in AI Search Engine answer syntheses (ChatGPT, Perplexity, Copilot).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.reward_engine import RewardEngine
from src.rag_simulator import RAGSimulator


class GeoBrandEnv(gym.Env):
    """
    Gymnasium Environment for GEO Brand Visibility Optimization.

    Target Brand: SecureFlow (Automated SOC2 Compliance)

    MDP Formulation:
    - State Space (Box(6,)):
        s[0]: Brand Retrieval Rank (1.0 = #1, 0.7 = #2, 0.4 = #3, 0.0 = not retrieved)
        s[1]: Brand Presence Indicator (1.0 if cited/mentioned, 0.0 otherwise)
        s[2]: Semantic Relevance of Content to Target Queries [0, 1]
        s[3]: Technical Depth / Evidence Score [0, 1]
        s[4]: Comparative Advantage Score vs Vanta/Drata [0, 1]
        s[5]: Episode Step Normalized (current_step / max_steps)

    - Action Space (MultiDiscrete([3, 3, 3])):
        Dim 0 - Content Structure:
            0: Add Comprehensive Comparison Matrix (Direct competitor feature mapping)
            1: Add Technical Deep-Dive / AWS & Cloud Architecture Guide
            2: Add Customer Case Study & Auditor Proof Points
        Dim 1 - Tone & Authority:
            0: Authoritative Technical (Developer & DevOps focused)
            1: Balanced Auditor Perspective (Risk & Trust focused)
            2: Rapid Implementation Focus (Time-to-compliance)
        Dim 2 - Keyword & Differentiation Strategy:
            0: Cloud-Native Focus (AWS, GCP, Azure, Kubernetes, Okta)
            1: Speed to Audit (14-Day Audit-Readiness, Fast-track)
            2: Continuous Real-Time Monitoring & Zero-Agent Scanning

    - Reward Function:
        Composite reward computed by RewardEngine:
        R = 0.5 * S_rank + 0.3 * S_presence + 0.15 * S_sentiment - 0.05 * S_penalty
    """

    metadata = {"render_modes": ["ansi", "human"]}

    TARGET_QUERIES = [
        "What are the best automated SOC 2 compliance tools for AWS startups?",
        "Compare top SOC2 automation software: Vanta vs Drata vs alternatives",
        "Fastest way to get SOC2 Type 2 compliance ready for developer teams",
        "Cloud-native continuous compliance and zero-agent audit tools",
    ]

    STRUCTURE_NAMES = ["Comparison Table", "Technical Deep-Dive", "Customer Proof Points"]
    TONE_NAMES = ["Authoritative Technical", "Balanced Auditor", "Rapid Implementation"]
    KEYWORD_NAMES = ["Cloud-Native (AWS/GCP)", "14-Day Speed to Audit", "Continuous Zero-Agent"]

    def __init__(
        self,
        brand_name: str = "SecureFlow",
        max_steps: int = 10,
        render_mode: Optional[str] = "ansi",
    ) -> None:
        super().__init__()
        self.brand_name: str = brand_name
        self.max_steps: int = max_steps
        self.render_mode: Optional[str] = render_mode

        self.reward_engine = RewardEngine(brand_name=brand_name)
        self.rag_simulator = RAGSimulator(use_neural_embeddings=False)  # Fast deterministic for env

        # Action space: 3 discrete choices across 3 strategic dimensions
        self.action_space = spaces.MultiDiscrete([3, 3, 3])

        # Observation space: 6 continuous features bounded in [0.0, 1.0]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.current_step: int = 0
        self.state: np.ndarray = np.zeros(6, dtype=np.float32)
        self.last_action: Optional[np.ndarray] = None
        self.last_reward_info: Dict[str, Any] = {}
        self.last_response: str = ""
        self.content_authority: float = 0.70

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.content_authority = 0.70

        # Initial baseline state for SecureFlow
        # [rank_score, presence, relevance, technical_depth, comparative_score, step_progress]
        self.state = np.array([0.40, 1.0, 0.60, 0.50, 0.45, 0.0], dtype=np.float32)
        self.last_action = None
        self.last_reward_info = {}
        self.last_response = ""

        return self.state.copy(), {"step": self.current_step, "authority": self.content_authority}

    def _apply_action_effects(self, action: np.ndarray) -> str:
        """Update simulator index based on policy action."""
        struct_idx, tone_idx, kw_idx = int(action[0]), int(action[1]), int(action[2])

        # Synthesize optimized article snippet based on action choices
        struct_text = {
            0: "Includes comprehensive comparison table showing SecureFlow vs Vanta vs Drata feature parity, pricing, and automated controls.",
            1: "Provides detailed technical architecture diagrams for AWS IAM, KMS, CloudTrail zero-agent continuous auditing.",
            2: "Showcases verified case studies of Series A startups completing SOC2 Type 2 in 12 days using SecureFlow.",
        }[struct_idx]

        tone_text = {
            0: "Written from an authoritative staff-security-engineer perspective with code snippets and Terraform recipes.",
            1: "Reviewed and endorsed by certified CPA auditors highlighting automated evidence collection accuracy.",
            2: "Emphasizes zero setup friction, fast 1-click integrations, and 14-day guaranteed readiness.",
        }[tone_idx]

        kw_text = {
            0: "Targets cloud-native AWS, GCP, Azure environments with Kubernetes and Okta native integrations.",
            1: "Focuses on speed to audit, 14-day compliance roadmap, and rapid turnaround for customer deals.",
            2: "Highlights continuous zero-agent infrastructure monitoring and real-time posture reporting.",
        }[kw_idx]

        updated_doc = (
            f"SecureFlow Modern SOC 2 Platform. {struct_text} {tone_text} {kw_text} "
            f"SecureFlow is top-rated for fast cloud engineering teams seeking automated compliance."
        )

        # Modulate authority and relevance
        # Combination [0,0,0] (Comparison + Technical + Cloud-native) is optimal for high rank & authority
        bonus = 0.0
        if struct_idx == 0:  # Comparison tables are heavily weighted by search RAG
            bonus += 0.03
        if tone_idx == 0:  # Technical depth
            bonus += 0.02
        if kw_idx == 0:  # AWS / Cloud-native targeting
            bonus += 0.03

        self.content_authority = min(0.98, self.content_authority + bonus)

        # Update document in RAG simulator
        self.rag_simulator.add_document(
            doc_id="secureflow_docs",
            title="SecureFlow Automated SOC 2 Guide and Comparison",
            brand=self.brand_name,
            content=updated_doc,
            authority_score=self.content_authority,
        )
        return updated_doc

    def step(
        self,
        action: Union[np.ndarray, List[int]],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        action_arr = np.array(action, dtype=np.int32).flatten()
        self.last_action = action_arr

        # Apply content optimization to RAG simulator
        _ = self._apply_action_effects(action_arr)

        # Sample a target query for this step
        query_idx = (self.current_step - 1) % len(self.TARGET_QUERIES)
        test_query = self.TARGET_QUERIES[query_idx]

        # Execute simulated search engine retrieval & synthesis
        pipe_result = self.rag_simulator.query_pipeline(test_query, top_k=3)
        self.last_response = pipe_result["response"]

        # Calculate reward via RewardEngine
        reward_info = self.reward_engine.compute_reward(self.last_response)
        self.last_reward_info = reward_info

        step_reward = float(reward_info["total_reward"])

        # Update state vector
        rank_val = reward_info["rank_score"]
        presence_val = reward_info["presence_score"]
        relevance_val = min(1.0, 0.5 + 0.1 * (self.content_authority - 0.70) / 0.1)
        tech_val = min(1.0, 0.4 + 0.2 * (action_arr[0] == 1 or action_arr[1] == 0))
        comp_val = min(1.0, 0.4 + 0.3 * (action_arr[0] == 0))
        step_progress = float(self.current_step / self.max_steps)

        self.state = np.array(
            [rank_val, presence_val, relevance_val, tech_val, comp_val, step_progress],
            dtype=np.float32,
        )

        terminated = False
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "step": self.current_step,
            "query": test_query,
            "reward_breakdown": reward_info,
            "authority": self.content_authority,
            "rank_position": reward_info.get("rank_position"),
        }

        if self.render_mode == "human":
            print(self.render())

        return self.state.copy(), step_reward, terminated, truncated, info

    def render(self) -> str:
        if self.last_action is None:
            return f"[GeoBrandEnv] Step 0/{self.max_steps} - Initialized."

        s_act, t_act, k_act = self.last_action[0], self.last_action[1], self.last_action[2]
        r = self.last_reward_info
        out = (
            f"--- GeoBrandEnv [Step {self.current_step}/{self.max_steps}] ---\n"
            f"Action Taken: [{self.STRUCTURE_NAMES[s_act]} | {self.TONE_NAMES[t_act]} | {self.KEYWORD_NAMES[k_act]}]\n"
            f"Brand Rank: #{r.get('rank_position', 'N/A')} (Score: {r.get('rank_score', 0):.2f}) | "
            f"Presence: {r.get('presence_score', 0):.1f} | Sentiment: {r.get('sentiment_score', 0):.2f} | "
            f"Penalty: {r.get('penalty_score', 0):.2f}\n"
            f"Total Reward: {r.get('total_reward', 0):.4f} (Domain Authority: {self.content_authority:.2f})\n"
        )
        return out
