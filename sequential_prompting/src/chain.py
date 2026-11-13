from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class ChainState:
    prompt: str
    steps: List[str] = field(default_factory=list)
    score: float = 0.0


class PromptChain:
    """Greedy/beam sequential prompt chaining controller."""

    def __init__(self, model_call: Callable[[str], str], scorer: Callable[[str], float], max_steps: int = 6):
        self.model_call = model_call
        self.scorer = scorer
        self.max_steps = max_steps

    def run(self, initial_prompt: str) -> ChainState:
        state = ChainState(prompt=initial_prompt)
        current = initial_prompt
        for step in range(self.max_steps):
            response = self.model_call(current)
            state.steps.append(response)
            state.score = self.scorer(response)
            current = f"{current}
{response}"
            if state.score >= 0.92:
                break
        return state
