from dataclasses import dataclass, field
from typing import Callable, Dict


@dataclass
class ModelSpec:
    # identifier for the model
    name: str
    # number of parameters in billions
    num_params: float
    # Function that formats the prompt
    prompt_formatter: Callable
    # model_id for each framework and dtype
    model_ids: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # whether this model uses a thinking/reasoning chat template
    thinking: bool = False

    def has_model_id(self, framework: str, dtype: str) -> bool:
        """Check if we have a model_id for the framework and dtype."""
        return framework in self.model_ids and dtype in self.model_ids[framework]
