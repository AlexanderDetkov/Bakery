"""Post-run analysis. Reads ONLY standardized artifacts (manifest/config/metrics JSON);
never imports the training stack (torch/transformers/peft). Enforced by a contract test."""
