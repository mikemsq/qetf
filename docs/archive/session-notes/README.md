# Session Notes

This folder contains detailed notes from individual Claude coding sessions for the QuantETF project.

-----

## Purpose

Session notes provide:

- **Detailed record** of what was built and why
- **Handoff context** for continuing work across sessions
- **Decision documentation** for future reference
- **Learning capture** - mistakes and patterns discovered
- **Historical context** - understanding how the codebase evolved

-----

## Naming Convention

Files must follow this format: `YYYY-MM-DD-description.md`

**Examples:**

- `2026-01-06-project-setup.md`
- `2026-01-07-data-fetching-api.md`
- `2026-01-10-portfolio-calculator.md`
- `2026-01-15-bug-fix-rate-limiting.md`

**Guidelines:**

- Use lowercase with hyphens
- Keep descriptions brief but clear
- Use dates for easy chronological sorting
- One session = one file (unless session spans multiple days)

-----

## Template

Use the session note template located at: `TEMPLATE.md`

**To create a new session note:**

1. Copy `TEMPLATE.md`
1. Rename to today’s date + description
1. Fill in as you work
1. Commit when session is complete

-----

## Session Index

### Week 1: January 6-12, 2026

|Date      |Session                                       |Focus Area                      |Status    |
|----------|----------------------------------------------|--------------------------------|----------|
|2026-01-06|[Project Setup](./2026-01-06-project-setup.md)|Infrastructure & workflow design|✅ Complete|
|          |                                              |                                |          |

### Week 2: January 13-19, 2026

|Date|Session|Focus Area|Status|
|----|-------|----------|------|
|    |       |          |      |

-----

## Quick Links

**Current session:** [Link to most recent session note]

**Templates:**

- [Session Note Template](./TEMPLATE.md)
- [Session Starter Templates](../docs/session-starters.md) *(if you save them)*

**Project docs:**

- [CLAUDE_CONTEXT.md](../CLAUDE_CONTEXT.md)
- [PROGRESS_LOG.md](../PROGRESS_LOG.md)
- [PROJECT_BRIEF.md](../PROJECT_BRIEF.md)

-----

## Session Types

Common session types and their focus:

**🏗️ Setup/Infrastructure**

- Initial project setup
- Configuration and tooling
- Workflow establishment

**📋 Planning**

- Feature planning and design
- Architecture decisions
- Task breakdown

**💻 Implementation**

- Feature development
- Code writing
- Integration work

**🐛 Bug Fixes**

- Debugging sessions
- Error resolution
- Issue tracking

**🧪 Testing**

- Test creation
- Quality assurance
- Verification

**📝 Documentation**

- README updates
- Code documentation
- Decision records

**🔍 Code Review**

- Reviewing generated code
- Refactoring
- Quality improvement

**🔄 Refactoring**

- Code cleanup
- Performance optimization
- Technical debt reduction

-----

## Best Practices

### During Sessions

- ✅ Start each session by loading context (CLAUDE_CONTEXT.md + PROGRESS_LOG.md)
- ✅ Document decisions as you make them
- ✅ Note challenges and solutions
- ✅ Track time spent on different activities
- ✅ Update files in real-time when possible

### After Sessions

- ✅ Complete the session note before ending
- ✅ Update PROGRESS_LOG.md daily entry
- ✅ Create clear handoff notes for next session
- ✅ Identify patterns/mistakes to add to CLAUDE_CONTEXT.md
- ✅ Link related sessions together

### Weekly

- ✅ Review all session notes from the week
- ✅ Update the index table above
- ✅ Identify recurring patterns
- ✅ Extract learnings for CLAUDE_CONTEXT.md
- ✅ Archive or consolidate if needed

-----

## What to Include

### Always Document

- Session goal and outcome
- Files created/modified
- Technical decisions made
- Testing performed
- Next steps for handoff

### Sometimes Document

- Code snippets (for complex/important code)
- Error messages encountered
- External resources used
- Time breakdowns
- Screenshots/visual artifacts

### Rarely Document

- Trivial changes
- Routine operations
- Duplicate information already in git commits

-----

## File Organization

As the project grows, consider organizing by:

```
session-notes/
├── README.md (this file)
├── TEMPLATE.md
├── 2026-01/
│   ├── 2026-01-06-project-setup.md
│   ├── 2026-01-07-feature-x.md
│   └── ...
├── 2026-02/
│   └── ...
└── archive/
    └── [older sessions if needed]
```

For now, keep all files in root `/session-notes/` directory.

-----

## Using Session Notes for Handoffs

**When starting a new session:**

1. Read CLAUDE_CONTEXT.md (coding standards)
1. Read PROGRESS_LOG.md (current status)
1. Read the most recent session note for context
1. Check “Next Session Should Focus On” section

**When ending a session:**

1. Fill out “Session Handoff” section completely
1. Be specific about what’s done vs. in-progress
1. Note any blockers clearly
1. Provide context for the next person/session

-----

## Metrics to Track

Session notes help track:

- **Velocity:** Features completed per week
- **Session efficiency:** % of sessions that met their goals
- **Common challenges:** Recurring issues to address
- **Time allocation:** Where time is spent
- **Quality:** Bugs introduced vs. fixed

Review these metrics weekly in PROGRESS_LOG.md.

-----

## Tips for Mobile Sessions

**Mobile-friendly approach:**

- Use shorter session notes for quick mobile sessions
- Focus on: Goal, Completed, Next Steps
- Expand details later from desktop if needed
- Take photos of whiteboard/sketches and reference in notes

**Quick mobile template:**

```markdown
# [Date-description]
Goal: [one sentence]
Completed: [bullet list]
Next: [what to do next]
```

-----

## Archive Policy

**Keep session notes if:**

- Major features or decisions
- Difficult bugs solved
- Important learnings captured
- Referenced by other sessions

**Consider archiving if:**

- Trivial/routine work
- Superseded by later sessions
- Over 6 months old with no references

For now, keep everything. Archive only when folder becomes unwieldy.

-----

## Questions?

If unsure what to document, ask:

- Will future me need to know this?
- Could another Claude session benefit from this context?
- Does this explain a “why” that’s not obvious from code?
- Is this a pattern worth remembering?

When in doubt, document it. You can always clean up later.

-----

**Last Updated:** January 6, 2026