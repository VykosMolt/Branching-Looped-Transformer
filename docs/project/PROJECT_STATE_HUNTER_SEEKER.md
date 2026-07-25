# Ouro Project — Hunter-Seeker / Ouro Core State

This file is now the navigation index for the chunked Hunter-Seeker/Ouro state.

Full preserved state lives in [`../hunter_seeker_state/`](../hunter_seeker_state/). The old monolithic state body was split losslessly on 2026-05-14; merged memo sources and checksums are recorded in [`../hunter_seeker_state/PRESERVATION_MANIFEST.md`](../hunter_seeker_state/PRESERVATION_MANIFEST.md).

## Start Here

- [Hunter-Seeker state directory](../hunter_seeker_state/README.md)
- [Preservation manifest](../hunter_seeker_state/PRESERVATION_MANIFEST.md)
- [Local-agent wrapper state](../local_agent/PROJECT_STATE_LOCAL_AGENT.md)
- [Evaluator/domain-transfer docs](../evaluator/README.md)

## State Chunks

- [Preamble and Current Snapshot](../hunter_seeker_state/chunks/00-preamble-current-snapshot.md)
- [Observation Policy, Topology Trio, and Refactor Updates](../hunter_seeker_state/chunks/01-observation-policy-refactor-2026-05.md)
- [Canonical Synthesis and Pre/Post Ladder Detail](../hunter_seeker_state/chunks/02-canonical-synthesis-ladder-2026-04-29.md)
- [Architecture, Roadmap, V17 History, and Encoder Drift](../hunter_seeker_state/chunks/03-architecture-roadmap-v17-history.md)
- [Engram Memory and Hazard Arbitration Handoff](../hunter_seeker_state/chunks/04-engram-memory-hazard-arbitration.md)
- [Risk Recovery, Phase Continuity, Cleanup, and Seeded Ladder](../hunter_seeker_state/chunks/05-risk-recovery-cleanup-2026-05-03-04.md)
- [RLTT Weights, Evaluator, Probes, and Acquisition Hardening](../hunter_seeker_state/chunks/06-rltt-evaluator-probes-acquisition.md)
- [Architecture Audit, Latest Patches, Naming, and Refactor Notes](../hunter_seeker_state/chunks/07-audit-latest-patches-refactor-2026-05.md)

## Merged Context

The following old standalone Markdown files were merged into `docs/hunter_seeker_state/imported/` and removed from their prior locations:

- [`PROJECT_ARCHITECTURE_MAP.md`](../hunter_seeker_state/imported/root/PROJECT_ARCHITECTURE_MAP.md)
- [`PROJECT_COMPONENT_REFERENCE.md`](../hunter_seeker_state/imported/root/PROJECT_COMPONENT_REFERENCE.md)
- [`SESSION_CONTEXT_2026-05-10.md`](../hunter_seeker_state/imported/root/SESSION_CONTEXT_2026-05-10.md)
- [`es_integration_plan.md`](../hunter_seeker_state/imported/root/es_integration_plan.md)
- [`hunter_seeker_additional_components.md`](../hunter_seeker_state/imported/root/hunter_seeker_additional_components.md)
- [`hybrid_engram_memory_notes.md`](../hunter_seeker_state/imported/root/hybrid_engram_memory_notes.md)
- [`looped_ssa_research_memo.md`](../hunter_seeker_state/imported/root/looped_ssa_research_memo.md)
- [`ouro_chat_sft_plan.md`](../hunter_seeker_state/imported/root/ouro_chat_sft_plan.md)
- [`pairwise_evaluator_locus_memo_2026-05-11.md`](../hunter_seeker_state/imported/root/pairwise_evaluator_locus_memo_2026-05-11.md)
- [`claude_sandbox/CLAUDE_SESSION_SUMMARY.md`](../hunter_seeker_state/imported/claude_sandbox/CLAUDE_SESSION_SUMMARY.md)
- [`claude_sandbox/SESSION_SUMMARY.md`](../hunter_seeker_state/imported/claude_sandbox/SESSION_SUMMARY.md)
- [`claude_sandbox/README.md`](../hunter_seeker_state/imported/claude_sandbox/README.md)
- [`claude_sandbox/gptopinion2.md`](../hunter_seeker_state/imported/claude_sandbox/gptopinion2.md)
- [`claude_sandbox/gptopinion3.md`](../hunter_seeker_state/imported/claude_sandbox/gptopinion3.md)
- [`claude_sandbox/gptsopinion.md`](../hunter_seeker_state/imported/claude_sandbox/gptsopinion.md)
- [`claude_sandbox/ouro_hunter_seeker_ladder_anchor_notes_2026-04-27.md`](../hunter_seeker_state/imported/claude_sandbox/ouro_hunter_seeker_ladder_anchor_notes_2026-04-27.md)
- [`claude_sandbox/post_quick_ladder_cleanup_plan.md`](../hunter_seeker_state/imported/claude_sandbox/post_quick_ladder_cleanup_plan.md)
- [`claude_sandbox/post_quick_ladder_cleanup_plan_v4.md`](../hunter_seeker_state/imported/claude_sandbox/post_quick_ladder_cleanup_plan_v4.md)
- [`claude_sandbox/pre_ladder_audit_backlog_updated.md`](../hunter_seeker_state/imported/claude_sandbox/pre_ladder_audit_backlog_updated.md)
- [`claude_sandbox/terminal_predframe_context_8run_ladder_post.md`](../hunter_seeker_state/imported/claude_sandbox/terminal_predframe_context_8run_ladder_post.md)
- [`claude_sandbox/design/ablation_ladder.md`](../hunter_seeker_state/imported/claude_sandbox/design/ablation_ladder.md)
- [`claude_sandbox/design/sprint_11_self_model.md`](../hunter_seeker_state/imported/claude_sandbox/design/sprint_11_self_model.md)
- [`claude_sandbox/hunter_seeker/README.md`](../hunter_seeker_state/imported/claude_sandbox/hunter_seeker/README.md)
- [`docs/root_notes_20260429_143517/README_transfer.md`](../hunter_seeker_state/imported/docs/root_notes_20260429_143517/README_transfer.md)
- [`docs/root_notes_20260429_143517/hunter_seeker_terminal_memory_handoff_codex.md`](../hunter_seeker_state/imported/docs/root_notes_20260429_143517/hunter_seeker_terminal_memory_handoff_codex.md)
- [`docs/root_notes_20260429_143517/ouro_hunter_seeker_handoff_notes_2026-04-28.md`](../hunter_seeker_state/imported/docs/root_notes_20260429_143517/ouro_hunter_seeker_handoff_notes_2026-04-28.md)
- [`docs/root_notes_20260429_143517/ouro_project_state.md`](../hunter_seeker_state/imported/docs/root_notes_20260429_143517/ouro_project_state.md)
- [`docs/root_notes_20260429_143517/pre_ladder_audit_backlog_final.md`](../hunter_seeker_state/imported/docs/root_notes_20260429_143517/pre_ladder_audit_backlog_final.md)

## Notes

- Historical paths inside older state chunks may point at deleted run artifacts or merged Markdown files. Use this index and the preservation manifest for current navigation.
- Future Hunter-Seeker/Ouro state updates should either edit a focused chunk under `docs/hunter_seeker_state/chunks/` or add a new dated chunk there, then update this index.
