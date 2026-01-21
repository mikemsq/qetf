# Handoffs Directory

This directory contains task handoffs, completion records, architecture docs, and research findings.

## Directory Structure

```
handoffs/
├── README.md               # This file
├── TEMPLATE-COMPLETION.md  # Standard completion report template
├── architecture/           # Long-lived architectural documents
├── research/               # Research findings and analysis
├── tasks/                  # Active task handoffs
└── completions/            # Task completion records
```

## Subdirectories

### architecture/
Long-lived architectural and planning documents that remain relevant across multiple tasks.
- Data Access Layer architecture and plans
- Phase implementation plans
- System design documents

### research/
Research findings, analysis reports, and strategy exploration results.
- Regime hypothesis research
- Strategy optimization findings
- Research agendas

### tasks/
Active task handoff files for work in progress or ready to pick up.
- Named: `handoff-TASK-ID.md` (e.g., `handoff-IMPL-019.md`)
- Delete after creating the completion file

### completions/
Task completion records documenting what was implemented.
- Named: `completion-TASK-ID.md` (e.g., `completion-IMPL-019.md`)
- Use `TEMPLATE-COMPLETION.md` as a starting point

## Workflow

1. **Task ready:** Handoff file exists in `tasks/`
2. **Task picked up:** Agent reads handoff, implements task
3. **Task completed:** Agent creates completion file in `completions/`
4. **Cleanup:** Consider deleting the handoff file (optional)

## See Also

- [TASKS.md](../TASKS.md) - Task queue and status
- [CLAUDE_CONTEXT.md](../CLAUDE_CONTEXT.md) - Documentation standards
