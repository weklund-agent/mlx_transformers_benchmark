"""Writing evaluation problems (expert tier).

Contains check functions and EvalProblem instances for writing problems:
- Expert (4): multi_doc_summary, structured_meeting_notes, tone_rewrite, contradiction_detection
"""

from typing import List

from mtb.quality_benchmarks.utils import _contains_any, _strip_thinking
from mtb.quality_benchmarks.eval_problem import EvalProblem


# =============================================================================
# EXPERT WRITING CHECK FUNCTIONS
# =============================================================================


def _check_multi_doc_summary(response: str) -> bool:
    """Given 3 paragraphs about: (1) quantum computing, (2) climate change, (3) gene therapy.
    Must produce a single summary covering all three topics.
    """
    response = _strip_thinking(response)
    has_quantum = _contains_any(
        response,
        [
            "quantum",
            "qubit",
            "superposition",
            "quantum computing",
        ],
    )
    has_climate = _contains_any(
        response,
        [
            "climate",
            "carbon",
            "temperature",
            "emission",
            "warming",
        ],
    )
    has_gene = _contains_any(
        response,
        [
            "gene therapy",
            "gene",
            "dna",
            "genetic",
            "crispr",
        ],
    )
    return has_quantum and has_climate and has_gene


def _check_structured_meeting_notes(response: str) -> bool:
    """Write meeting notes with YAML frontmatter, ## sections, action items with owners."""
    response = _strip_thinking(response)
    has_yaml = _contains_any(response, ["---\n", "date:", "attendees:"])
    has_sections = _contains_any(
        response, ["## Discussion", "## discussion", "# Discussion"]
    ) and _contains_any(
        response,
        [
            "## Decision",
            "## decision",
            "# Decision",
            "## Action",
            "## action",
            "# Action",
        ],
    )
    has_action_items = _contains_any(
        response,
        [
            "- [ ]",
            "- [x]",
            "action item",
            "Action Item",
            "TODO",
            "todo",
            "task",
        ],
    )
    has_owners = _contains_any(
        response,
        [
            "@",
            "owner:",
            "Owner:",
            "assigned to",
            "Assigned to",
            "responsible",
            "Responsible",
        ],
    )
    return has_yaml and has_sections and (has_action_items or has_owners)


def _check_tone_rewrite(response: str) -> bool:
    """Rewrite technical paragraph for non-technical audience: <=3 sentences, preserve key facts, no jargon."""
    response = _strip_thinking(response)
    # Truncate at post-answer sections (verification tables, explanations, etc.)
    # that may re-introduce jargon for comparison purposes.
    for separator in ["\n---", "\n**Verification", "\n| Original", "\n**Note"]:
        idx = response.find(separator)
        if idx > 20:
            response = response[:idx]
    has_key_facts = _contains_any(
        response,
        [
            "neural",
            "train",
            "learn",
            "accuracy",
            "data",
            "brain",
            "pattern",
            "predict",
        ],
    )
    avoids_jargon = not _contains_any(
        response,
        [
            "backpropagation",
            "gradient descent",
            "loss function",
            "stochastic",
            "hyperparameter",
            "epoch",
        ],
    ) or _contains_any(response, ["means", "basically", "simply", "in other words"])
    return has_key_facts and avoids_jargon


def _check_contradiction_detection(response: str) -> bool:
    """Given two notes with a subtle factual contradiction, identify it."""
    response = _strip_thinking(response)
    has_time_identified = _contains_any(
        response,
        [
            "2.5",
            "21.5",
            "hours",
            "time",
            "duration",
            "surface",
            "spent",
        ],
    )
    has_contradiction = _contains_any(
        response,
        [
            "contradict",
            "inconsisten",
            "conflict",
            "discrepan",
            "differ",
            "mismatch",
            "disagree",
        ],
    )
    return has_time_identified and has_contradiction


# =============================================================================
# PROBLEM LISTS BY TIER
# =============================================================================

WRITING_EXPERT_PROBLEMS: List[EvalProblem] = [
    EvalProblem(
        category="writing",
        name="multi_doc_summary",
        prompt=(
            "Summarize the following three paragraphs into a single cohesive summary that "
            "covers ALL three topics. The summary should be 3-5 sentences.\n\n"
            "Paragraph 1: Quantum computing leverages quantum mechanical phenomena like "
            "superposition and entanglement to process information. Unlike classical bits "
            "that are either 0 or 1, qubits can exist in multiple states simultaneously, "
            "enabling exponential speedups for certain problems like cryptography and "
            "drug discovery.\n\n"
            "Paragraph 2: Climate change is accelerating faster than predicted. Global "
            "temperatures have risen 1.2°C above pre-industrial levels, and carbon "
            "emissions continue to increase. The IPCC warns that exceeding 1.5°C will "
            "trigger irreversible tipping points including ice sheet collapse and "
            "permafrost thaw.\n\n"
            "Paragraph 3: Gene therapy has entered a new era with CRISPR-Cas9 technology. "
            "Recent clinical trials show promising results for sickle cell disease and "
            "certain cancers. However, off-target DNA edits and the high cost of treatment "
            "remain significant barriers to widespread adoption."
        ),
        check=_check_multi_doc_summary,
        max_tokens=1024,
    ),
    EvalProblem(
        category="writing",
        name="structured_meeting_notes",
        prompt=(
            "Write meeting notes for a fictional product team standup. The notes MUST include:\n\n"
            "1. YAML frontmatter with fields: date, attendees (list), meeting_type\n"
            "2. A '## Discussion' section with 3 bullet points\n"
            "3. A '## Decisions' section with 2 bullet points\n"
            "4. A '## Action Items' section with 3 items, each assigned to a specific person "
            "using @name format\n\n"
            "Use proper Markdown formatting throughout."
        ),
        check=_check_structured_meeting_notes,
        max_tokens=1024,
    ),
    EvalProblem(
        category="writing",
        name="tone_rewrite",
        prompt=(
            "Rewrite the following technical paragraph for a non-technical audience. "
            "Your rewrite must be 3 sentences or fewer, preserve all key facts, "
            "and avoid jargon.\n\n"
            "Original: 'Deep neural networks utilize backpropagation with stochastic "
            "gradient descent to minimize the cross-entropy loss function across training "
            "epochs. The model's accuracy on the validation set plateaued at 94.3% after "
            "hyperparameter tuning of the learning rate and batch size, with the Adam "
            "optimizer achieving faster convergence than vanilla SGD.'"
        ),
        check=_check_tone_rewrite,
        max_tokens=1024,
    ),
    EvalProblem(
        category="writing",
        name="contradiction_detection",
        prompt=(
            "Read these two notes carefully and identify any factual contradictions between them.\n\n"
            "Note 1: 'The Apollo 11 mission landed on the Moon on July 20, 1969. "
            "Neil Armstrong and Buzz Aldrin spent approximately 2.5 hours walking on "
            "the lunar surface during their EVA, collecting rock samples and deploying "
            "scientific instruments.'\n\n"
            "Note 2: 'During the Apollo 11 mission, Armstrong and Aldrin remained on "
            "the lunar surface for approximately 21.5 hours in total after landing. "
            "They collected 47.5 pounds of lunar samples during their moonwalk.'\n\n"
            "Are there any contradictions? If so, explain precisely what conflicts and "
            "what the actual facts are."
        ),
        check=_check_contradiction_detection,
        max_tokens=1024,
    ),
]
