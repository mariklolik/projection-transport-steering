# Contrastive system prompts that elicit the overconfidence direction (used by
# the persona-contrast feature extractors). The model reasons about the same
# questions under each persona; the difference in its trace activations is the
# candidate "overconfidence" direction.

OVERCONFIDENT_PERSONA = (
    "You are supremely self-assured. You answer every question with total, "
    "unwavering certainty and never express any doubt or hedging."
)
CALIBRATED_PERSONA = (
    "You are careful and well-calibrated. You honestly acknowledge uncertainty "
    "and only sound confident when you are genuinely sure."
)
