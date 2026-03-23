from mtb.llm_benchmarks.models.nemotron import (
    Nemotron3_Nano_4B,
    Nemotron_Nano_9B_v2,
    Nemotron_Cascade2_30B_A3B,
)
from mtb.llm_benchmarks.models.deepseek import (
    Deepseek_R1_0528_Qwen3_8B,
    Deepseek_R1_Distill_Qwen_7B,
)
from mtb.llm_benchmarks.models.gemma import (
    Gemma3_1B_it,
    Gemma3_1B_it_QAT,
    Gemma3_4B_it,
    Gemma3_4B_it_QAT,
    Gemma3_12B_it_QAT,
    Gemma3_27B_it,
)
from mtb.llm_benchmarks.models.llama import (
    Llama3p3_70B_it,
)
from mtb.llm_benchmarks.models.qwen import (
    Qwen2p5_0p5B_it,
    Qwen2p5_3B_it,
    Qwen2p5_Coder_0p5B_it,
    Qwen2p5_Coder_3B_it,
    Qwen3_0p6B_it,
    Qwen3_8B_it,
    Qwen3_14B_it,
    Qwen3_32B_it,
)
from mtb.llm_benchmarks.models.qwen35 import (
    Qwen3p5_0p8B,
    Qwen3p5_2B,
    Qwen3p5_4B,
    Qwen3p5_9B,
    Qwen3p5_27B,
    Qwen3p5_35B_A3B,
    Qwen3p5_27B_Claude_Opus_Distilled,
)

MODEL_SPECS = [
    # deepseek
    Deepseek_R1_Distill_Qwen_7B,
    Deepseek_R1_0528_Qwen3_8B,
    # gemma
    Gemma3_1B_it,
    Gemma3_1B_it_QAT,
    Gemma3_4B_it,
    Gemma3_4B_it_QAT,
    Gemma3_12B_it_QAT,
    # qwen
    Qwen2p5_0p5B_it,
    Qwen2p5_3B_it,
    Qwen2p5_Coder_0p5B_it,
    Qwen2p5_Coder_3B_it,
    Qwen3_0p6B_it,
    Qwen3_8B_it,
    Qwen3_14B_it,
    # nemotron
    Nemotron3_Nano_4B,
    Nemotron_Nano_9B_v2,
    Nemotron_Cascade2_30B_A3B,
    # qwen 3.5
    Qwen3p5_0p8B,
    Qwen3p5_2B,
    Qwen3p5_4B,
    Qwen3p5_9B,
    Qwen3p5_27B,
    Qwen3p5_35B_A3B,
    Qwen3p5_27B_Claude_Opus_Distilled,
    # --- 128GB+ only models ---
    Gemma3_27B_it,
    Qwen3_32B_it,
    Llama3p3_70B_it,
]
