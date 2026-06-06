"""Bakery — a clean, agent-extensible Prompt-Baking research codebase.

Bake a prompt u into LoRA weights so the UNPROMPTED baked model mimics the
PROMPTED base model:  B(θ, u) = argmin_θ_u  D_KL( P_θ(·|u) ‖ P_θ_u(·) ).

See CLAUDE.md for the methodology and invariants.
"""

__version__ = "0.1.0"
