from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np

__all__ = [
    "LlmBenchmarkMeasurement",
    "Measurements",
]


@dataclass
class LlmBenchmarkMeasurement:
    """Measurement for LLM benchmarks, for one prompt."""

    # generated text
    response: str

    # time needed to parse prompt, init kv cache
    prompt_time_sec: float
    prompt_tps: float  # tokens/sec

    # time needed for generation
    generation_time_sec: float
    generation_tps: float  # tokens/sec

    # number of tokens
    num_prompt_tokens: int
    num_generated_tokens: int

    # peak memory usage
    peak_memory_gib: float

    def to_dict(self, include_reponse: bool = False) -> Dict:
        dictionary = asdict(self)
        if not include_reponse:
            dictionary.pop("response")
        return dictionary


class Measurements:
    """Container class for measurements."""

    def __init__(self):
        self._measurements = None
        self._num_measurements = 0

    @property
    def keys(self) -> List[str]:
        if self._measurements is None:
            return []
        else:
            return list(self._measurements.keys())

    def add(self, measurement: Dict):
        # initialize container if needed
        if self._measurements is None:
            self._measurements = dict()
            for key in measurement:
                self._measurements[key] = []

        # add values to the container
        for key in self._measurements:
            if key not in measurement:
                raise KeyError(f"Key {key} not found in measurements")

            self._measurements[key].append(measurement[key])

        self._num_measurements += 1
        return

    def get_mean(self, key: str) -> float:
        if key in self._measurements:
            return np.mean(self._measurements[key])
        else:
            raise KeyError(
                f"Key {key} not found in measurements, must be one of {list(self._measurements.keys())}"
            )

    def get_std(self, key: str) -> float:
        if key in self._measurements:
            return np.std(self._measurements[key])
        else:
            raise KeyError(
                f"Key {key} not found in measurements, must be one of {list(self._measurements.keys())}"
            )

    def get_means(self) -> Dict[str, float]:
        return {key: self.get_mean(key) for key in self._measurements}

    def reset(self):
        self._measurements = None
        self._num_measurements = 0

    def __repr__(self) -> str:
        tostring = (
            f"{self.__class__.__name__}("
            f"\n  num_measurements={self._num_measurements},"
        )
        for key in self._measurements:
            tostring += f"\n  {key}={self.get_mean(key):.4f},"
        tostring += "\n)"
        return tostring
