# Agent Workflow - QuantETF

**Last Updated:** January 9, 2026

This document describes the multi-agent workflow for developing QuantETF. Instead of relying on single long-running sessions, we use specialized agents working in parallel from shared documentation and a task queue.

## Overview

The project uses a **distributed agentic workflow** inspired by Boris Cherny's approach, adapted for parallel agent execution:

```
Planning Agent → Task Queue → Specialized Coding Agents → Review Agent
       ↓                              ↓
   TASKS.md                     Implementation
       ↓                              ↓
Documentation ← ─────────────── PROGRESS_LOG.md
```

## Agent Roles

### 1. Planning Agent
- **Responsibility:** Break down features into specific, independent tasks
- **Input:** PROJECT_BRIEF.md, PROGRESS_LOG.md, user requirements
- **Output:** TASKS.md with detailed task specifications
- **When to use:** Start of each phase, when adding new features

### 2. Coding Agent (multiple instances)
- **Responsibility:** Implement specific tasks from the queue
- **Input:** Single task from TASKS.md, CLAUDE_CONTEXT.md
- **Output:** Working code + tests + task completion note
- **When to use:** Pick up any task marked "ready"

### 3. Review Agent
- **Responsibility:** Verify implementations, run tests, update docs
- **Input:** Completed task, test results
- **Output:** Updated PROGRESS_LOG.md, merged code
- **When to use:** After coding agent completes a task

### 4. Scheduling Agent
- **Responsibility:** Maintain task queue, identify blockers, prioritize
- **Input:** TASKS.md, PROGRESS_LOG.md
- **Output:** Updated task priorities and dependencies
- **When to use:** Daily, or when tasks are blocked

## File Structure

### Core Documentation (Always Read)
- **CLAUDE_CONTEXT.md** - Coding standards, patterns, principles
- **PROJECT_BRIEF.md** - Overall goals, phases, architecture
- **PROGRESS_LOG.md** - Daily updates, decisions, status

### Agent Workflow Files (New)
- **TASKS.md** - Task queue with statuses
- **HANDOFFS/** - Directory for agent-to-agent handoffs
  - `handoff-TASKID.md` - Detailed task context
  - `completion-TASKID.md` - Task completion summary

## Task Lifecycle

### 1. Task Creation (Planning Agent)

```yaml
Task: IMPL-001-momentum-alpha
Status: ready
Priority: high
Assigned: (none - available for pickup)
Estimated: 1-2 hours
Dependencies: []

Description: |
  Implement MomentumAlpha class in src/quantetf/alpha/momentum.py

Acceptance Criteria:
  - Implements AlphaModel base class
  - Uses 252-day lookback by default
  - Strict T-1 data access (no lookahead)
  - Includes docstrings and type hints
  - Has unit tests with synthetic data
  - Has integration test with real snapshot

Context:
  - Read: CLAUDE_CONTEXT.md, src/quantetf/alpha/base.py
  - Pattern: See existing YFinanceProvider for data access
  - Testing: See tests/test_yfinance_provider.py for test patterns

Handoff File: handoffs/handoff-IMPL-001.md
```

### 2. Task Pickup (Coding Agent)

Agent updates TASKS.md:
```yaml
Status: in_progress
Assigned: Agent-Alpha-001
Started: 2026-01-09 10:30
```

### 3. Implementation

Agent works on the task, following CLAUDE_CONTEXT.md standards.

### 4. Completion

Agent:
1. Runs tests (`pytest tests/`)
2. Creates completion note: `handoffs/completion-IMPL-001.md`
3. Updates TASKS.md status to `completed`
4. Commits code with clear message

### 5. Review (Review Agent)

Agent:
1. Verifies tests pass
2. Checks code quality
3. Updates PROGRESS_LOG.md
4. Marks task as `merged`

## Creating Task Handoffs

### Handoff File Template

```markdown
# Task Handoff: IMPL-001-momentum-alpha

## Quick Context
You are implementing the MomentumAlpha class. This is part of Phase 2
(Backtest Engine) which started on Jan 9, 2026.

## What You Need to Know
- We use T-1 data access (no lookahead bias)
- All price data is in MultiIndex format (Ticker, Price_Field)
- Momentum = (current_price / price_N_days_ago) - 1.0

## Files to Read First
1. /workspaces/qetf/CLAUDE_CONTEXT.md - Coding standards
2. /workspaces/qetf/src/quantetf/alpha/base.py - Base class
3. /workspaces/qetf/src/quantetf/data/snapshot_store.py - Data access

## Implementation Steps
1. Create MomentumAlpha class inheriting from AlphaModel
2. Implement __init__ with lookback_days parameter
3. Implement score() method with T-1 data access
4. Add comprehensive docstrings
5. Create tests in tests/test_momentum_alpha.py
6. Run: pytest tests/test_momentum_alpha.py

## Acceptance Criteria
- [ ] Class implements AlphaModel interface correctly
- [ ] Uses store.get_close_prices(as_of=...) for T-1 data
- [ ] Handles missing data gracefully (returns NaN scores)
- [ ] All tests pass
- [ ] Code follows CLAUDE_CONTEXT.md standards

## Success Looks Like
```python
# Can be used like this:
alpha = MomentumAlpha(lookback_days=252)
scores = alpha.score(
    as_of=pd.Timestamp("2023-12-31"),
    universe=universe,
    features=features,  # Not used, but required by interface
    store=store
)
# Returns AlphaScores with momentum for each ticker
```

## Questions? Issues?
If blocked or unclear:
1. Check CLAUDE_CONTEXT.md for patterns
2. Look at similar implementations (e.g., YFinanceProvider)
3. Document the blocker in completion note
```

## Parallel Work Strategy

### Example: Phase 2 Parallelization

**Planning Agent creates 5 tasks:**
1. `IMPL-001` - MomentumAlpha model
2. `IMPL-002` - EqualWeightTopN constructor
3. `IMPL-003` - FlatTransactionCost model
4. `IMPL-004` - SimpleBacktestEngine
5. `TEST-001` - Integration tests

**Execution:**
- Agent-Alpha-001 picks up IMPL-001 (momentum)
- Agent-Portfolio-001 picks up IMPL-002 (constructor)
- Agent-Cost-001 picks up IMPL-003 (cost model)
- All work in parallel
- When all 3 complete, Agent-Engine-001 picks up IMPL-004
- Finally Agent-Test-001 picks up TEST-001

**Time saved:** 3x speedup vs sequential

## Task Dependencies

Tasks specify dependencies:
```yaml
Task: IMPL-004-backtest-engine
Dependencies: [IMPL-001, IMPL-002, IMPL-003]
Status: blocked
```

Scheduling agent updates to `ready` when dependencies complete.

## Benefits of This Approach

1. **No context loss** - Each agent starts fresh
2. **Parallel execution** - Multiple tasks simultaneously
3. **Clear handoffs** - Explicit task specifications
4. **Better testing** - Each component tested independently
5. **Easier debugging** - Small, focused changes
6. **Scalable** - Add more agents as needed
7. **Resilient** - Agent failure doesn't lose all progress

## Best Practices

### For Planning Agent
- Break tasks into 1-3 hour chunks
- Make tasks as independent as possible
- Provide clear acceptance criteria
- Include code examples in handoffs

### For Coding Agents
- Read handoff file completely first
- Follow CLAUDE_CONTEXT.md standards
- Write tests before marking complete
- Update completion notes with any learnings

### For Review Agent
- Verify tests pass
- Check for CLAUDE_CONTEXT.md compliance
- Update PROGRESS_LOG.md with what was completed
- Identify patterns for CLAUDE_CONTEXT.md

### For Scheduling Agent
- Check daily for blocked tasks
- Identify new dependencies as they emerge
- Prioritize critical path tasks
- Balance agent workload

## Migration from Current State

**Current:** Single agent session with todo list
**Target:** Multiple agents with task queue

**Steps:**
1. Create TASKS.md from current todos
2. Create handoff files for each task
3. Launch specialized agents to pick up tasks
4. Review agent consolidates results

## Example Session Starters

### Planning Agent
```
Read:
- /workspaces/qetf/PROJECT_BRIEF.md
- /workspaces/qetf/PROGRESS_LOG.md

Your role: Planning Agent
Task: Break down Phase 2 (Backtest Engine) into 5-10 independent tasks

Create:
- TASKS.md with task specifications
- handoffs/ directory with handoff files for each task

Make tasks parallelizable where possible.
```

### Coding Agent
```
Read:
- /workspaces/qetf/CLAUDE_CONTEXT.md
- /workspaces/qetf/handoffs/handoff-IMPL-001.md

Your role: Coding Agent
Task: Implement the task specified in handoff-IMPL-001.md

Deliver:
- Working implementation
- Tests (all passing)
- handoffs/completion-IMPL-001.md

Update TASKS.md status when done.
```

## Tools

### Task Status Check
```bash
# See what tasks are available
grep "Status: ready" TASKS.md

# See what's in progress
grep "Status: in_progress" TASKS.md

# See completed tasks
grep "Status: completed" TASKS.md
```

### Quick Agent Launch
```bash
# For next session: point agent directly to task
claude "Read handoffs/handoff-IMPL-001.md and implement it"
```

## References

- Boris Cherny's agentic workflow: https://threadreaderapp.com/thread/2007179832300581177.html
- Adapted for parallel multi-agent execution
- Inspired by kanban / agile task management
