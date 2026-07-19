# Agent Notes Index

Purpose: keep app memory lightweight, factual, and easy to scan between tasks.

## Files
- `SESSION_NOTES.md`: current app state and key links.
- `MODULE_TRACKER.md`: durable app module status and next steps.
- `DECISION_LOG.md`: durable runtime and app decisions.
- `CROSS_REPO_CONTEXT.md`: shared contract with the modeling repo.
- `Session_Notes/YYYY-MM.md`: append-only chronological task log.
- `../research/README.md`: index for standalone research studies.

## Routine
1. Start with `SESSION_NOTES.md`.
2. Check the latest monthly note.
3. Use `MODULE_TRACKER.md` and `DECISION_LOG.md` for durable state.
4. Keep large experiment outputs in `research/`, not `app/`.
5. End meaningful tasks with a short monthly-log entry.

## Guardrails
- Keep notes concise and implementation-focused.
- Do not store secrets, credentials, or raw private data.
- If notes conflict with the latest user request, the latest user request wins.
