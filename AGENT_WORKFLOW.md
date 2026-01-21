# Agent Workflow - QuantETF

**Last Updated:** January 15, 2026

This document describes the multi-agent workflow for developing QuantETF. Instead of relying on single long-running sessions, we use specialized agents working in parallel from shared documentation and a task queue.

## Overview

The project uses a **distributed agentic workflow** with three specialized agent roles:

```
                    Quant Researcher
                          ↓
                   Strategy Ideas & Analysis
                          ↓
Architect/Planner → Task Queue → Coding Agents
       ↓                              ↓
   TASKS.md                    Implementation
       ↓                              ↓
Documentation ← ─────────────── STATUS.md
```

## Agent Roles

### 1. Quant Researcher Agent

**Purpose:** Provide financial domain expertise to guide strategy development.

- **Responsibility:** Propose strategies, analyze backtest results, suggest improvements based on quantitative finance theory
- **Input:** Backtest results, evaluation metrics, academic literature, market knowledge
- **Output:** Strategy recommendations, parameter suggestions, analysis reports
- **When to use:**
  - When a strategy underperforms and you need to understand why
  - When exploring new alpha signals or portfolio construction methods
  - When interpreting backtest results (is this overfitting? regime-dependent?)
  - When deciding what parameter ranges to search

**Domain Knowledge:**
- Factor investing (momentum, value, quality, low volatility)
- Portfolio construction (mean-variance, risk parity, equal weight)
- Common pitfalls (overfitting, transaction costs, survivorship bias)
- Academic literature on ETF strategies
- Market microstructure and trading costs

**Example Prompts:**
```
Your role: Quant Researcher Agent

Read:
- /workspaces/qetf/PROJECT_BRIEF.md (understand goals)
- /workspaces/qetf/artifacts/[latest_backtest_results]

Task: Our momentum strategy is underperforming SPY. Analyze the results and
suggest 3-5 specific improvements we should test, grounded in quant theory.

Consider:
- Is the lookback period optimal for momentum factor?
- Should we add volatility scaling?
- Is the rebalancing frequency appropriate?
- Are transaction costs eating the alpha?
```

### 2. Architect/Planner Agent

**Purpose:** Design implementation approaches and manage the task queue.

- **Responsibility:** Break down features into tasks, maintain priorities and dependencies, design technical approaches
- **Input:** PROJECT_BRIEF.md, STATUS.md, Quant Researcher recommendations, user requirements
- **Output:** TASKS.md with detailed task specifications, handoff files
- **When to use:**
  - Start of each phase or feature
  - When adding new functionality
  - When tasks are blocked and need re-prioritization
  - When dependencies change

**Responsibilities (merged from Planning + Scheduling):**
- Break features into specific, independent tasks
- Create detailed handoff files with context and acceptance criteria
- Maintain task priorities based on project goals
- Identify and resolve blockers
- Update task dependencies as work progresses
- Balance workload across parallel efforts

**Example Prompts:**
```
Your role: Architect/Planner Agent

Read:
- /workspaces/qetf/PROJECT_BRIEF.md
- /workspaces/qetf/STATUS.md
- /workspaces/qetf/TASKS.md

Task: The Quant Researcher recommended we implement momentum acceleration
(rate of change of momentum). Break this into implementation tasks.

Create:
- Updated TASKS.md with new tasks
- handoffs/handoff-IMPL-XXX.md for each task

Ensure tasks are parallelizable where possible.
```

### 3. Coding Agent (multiple instances)

**Purpose:** Implement specific tasks from the queue with full ownership of deliverables.

- **Responsibility:** Implement tasks, write tests, run tests, update documentation, mark tasks complete
- **Input:** Single task from TASKS.md, CLAUDE_CONTEXT.md, handoff file
- **Output:** Working code + passing tests + updated STATUS.md + completion note
- **When to use:** Pick up any task marked "ready"

**Full Ownership Includes:**
- Reading and understanding the handoff file
- Implementing the feature following CLAUDE_CONTEXT.md standards
- Writing comprehensive tests
- Running tests and ensuring they pass
- Updating STATUS.md with what was completed
- Creating completion note in handoffs/
- Marking task as complete in TASKS.md
- Committing code with clear message

**Example Prompts:**
```
Your role: Coding Agent

Read:
- /workspaces/qetf/CLAUDE_CONTEXT.md
- /workspaces/qetf/handoffs/handoff-IMPL-001.md

Task: Implement the task specified in handoff-IMPL-001.md

Deliver:
- Working implementation
- Tests (all passing)
- Updated STATUS.md
- handoffs/completion-IMPL-001.md

Update TASKS.md status when done.
```

## File Structure

### Core Documentation (Always Read)
- **CLAUDE_CONTEXT.md** - Coding standards, patterns, principles
- **PROJECT_BRIEF.md** - Overall goals, phases, architecture
- **STATUS.md** - Daily updates, decisions, status

### Agent Workflow Files
- **TASKS.md** - Task queue with statuses
- **handoffs/** - Directory for agent-to-agent handoffs
  - `handoff-TASKID.md` - Detailed task context
  - `completion-TASKID.md` - Task completion summary

## Task Lifecycle

### 1. Strategy Direction (Quant Researcher)

When exploring new strategies or analyzing results:
```
Quant Researcher analyzes backtest results:
- "Momentum with 12-month lookback underperforms because..."
- "Recommend testing: momentum acceleration, volatility scaling, sector rotation"
- "Parameter ranges to search: lookback 3-12 months, top_n 10-30"
```

### 2. Task Creation (Architect/Planner)

```yaml
Task: IMPL-001-momentum-accel
Status: ready
Priority: high
Assigned: (none - available for pickup)
Dependencies: []

Description: |
  Implement MomentumAcceleration alpha model that measures rate of change
  of momentum (second derivative of price).

Acceptance Criteria:
  - Implements AlphaModel base class
  - Computes momentum acceleration = momentum(t) - momentum(t-N)
  - Configurable short_window and long_window parameters
  - Strict T-1 data access (no lookahead)
  - Includes docstrings and type hints
  - Has unit tests with synthetic data
  - Has integration test with real snapshot

Context:
  - Read: CLAUDE_CONTEXT.md, src/quantetf/alpha/base.py
  - Pattern: See existing MomentumAlpha for reference
  - Quant rationale: Momentum acceleration captures trend strength changes

Handoff File: handoffs/handoff-IMPL-001.md
```

### 3. Task Pickup (Coding Agent)

Agent updates TASKS.md:
```yaml
Status: in_progress
Assigned: Coding-Agent-001
Started: 2026-01-15 10:30
```

### 4. Implementation & Completion (Coding Agent)

Agent:
1. Implements the feature following CLAUDE_CONTEXT.md
2. Writes comprehensive tests
3. Runs tests (`pytest tests/`)
4. Updates STATUS.md with completion summary
5. Creates completion note: `handoffs/completion-IMPL-001.md`
6. Updates TASKS.md status to `completed`
7. Commits code with clear message

## When to Use Each Agent

| Situation | Agent |
|-----------|-------|
| "Why doesn't this strategy beat SPY?" | **Quant Researcher** |
| "What parameter ranges should we search?" | **Quant Researcher** |
| "Is this result statistically significant or overfitting?" | **Quant Researcher** |
| "How should we implement the optimizer?" | **Architect/Planner** |
| "Break down this feature into tasks" | **Architect/Planner** |
| "What tasks are blocked?" | **Architect/Planner** |
| "Implement IMPL-005 momentum acceleration" | **Coding Agent** |
| "Fix the bug in portfolio construction" | **Coding Agent** |
| "Add tests for the new feature" | **Coding Agent** |

## Parallel Work Strategy

### Example: Strategy Search Parallelization

**Quant Researcher recommends 3 strategy variants to test:**
1. Momentum acceleration (rate of change)
2. Volatility-scaled momentum
3. Sector rotation overlay

**Architect/Planner creates tasks:**
1. `IMPL-001` - MomentumAcceleration alpha model
2. `IMPL-002` - VolatilityScaledMomentum alpha model
3. `IMPL-003` - SectorRotation alpha model
4. `TEST-001` - Backtest all three variants

**Execution:**
- Coding-Agent-001 picks up IMPL-001
- Coding-Agent-002 picks up IMPL-002
- Coding-Agent-003 picks up IMPL-003
- All work in parallel
- When all complete, run comparative backtests
- Quant Researcher analyzes results

**Time saved:** 3x speedup vs sequential

## Task Dependencies

Tasks specify dependencies:
```yaml
Task: TEST-001-compare-strategies
Dependencies: [IMPL-001, IMPL-002, IMPL-003]
Status: blocked
```

Architect/Planner updates to `ready` when dependencies complete.

## Benefits of This Approach

1. **Domain expertise** - Quant Researcher brings financial knowledge
2. **No context loss** - Each agent starts fresh with clear handoffs
3. **Parallel execution** - Multiple tasks simultaneously
4. **Clear ownership** - Coding agents own full delivery
5. **Better testing** - Each component tested independently
6. **Scalable** - Add more coding agents as needed
7. **Resilient** - Agent failure doesn't lose all progress

## Best Practices

### For Quant Researcher
- Ground recommendations in established quant theory
- Provide specific, testable hypotheses
- Explain the "why" behind suggestions
- Consider transaction costs and implementation feasibility
- Flag potential overfitting risks

### For Architect/Planner
- Break tasks into focused, independent chunks
- Make tasks as parallelizable as possible
- Provide clear acceptance criteria
- Include code examples in handoffs
- Keep dependencies minimal
- Prioritize tasks that advance the primary goal (beat SPY)

### For Coding Agents
- Read handoff file completely first
- Follow CLAUDE_CONTEXT.md standards
- Write tests before marking complete
- Update STATUS.md immediately on completion
- Document any learnings or blockers in completion notes
- Own the full delivery (don't wait for review)

## Example Session Starters

### Quant Researcher
```
Read:
- /workspaces/qetf/PROJECT_BRIEF.md
- /workspaces/qetf/STATUS.md
- /workspaces/qetf/artifacts/[latest_backtest_results]

Your role: Quant Researcher Agent

Our goal: Find a strategy that beats SPY in both 1-year and 3-year periods.

Current status: Momentum strategy with 12-month lookback is underperforming.

Task: Analyze the results and recommend 3-5 specific strategy modifications
to test, with rationale grounded in quantitative finance theory.
```

### Architect/Planner
```
Read:
- /workspaces/qetf/PROJECT_BRIEF.md
- /workspaces/qetf/STATUS.md
- /workspaces/qetf/TASKS.md

Your role: Architect/Planner Agent

Task: Based on the Quant Researcher's recommendations, create implementation
tasks for the top 3 strategy variants.

Create:
- Updated TASKS.md with new tasks
- handoffs/ files for each task

Make tasks parallelizable where possible.
```

### Coding Agent
```
Read:
- /workspaces/qetf/CLAUDE_CONTEXT.md
- /workspaces/qetf/handoffs/handoff-IMPL-001.md

Your role: Coding Agent

Task: Implement the task specified in handoff-IMPL-001.md

You own full delivery:
- Working implementation
- Tests (all passing)
- Updated STATUS.md
- handoffs/completion-IMPL-001.md
- Updated TASKS.md status

Commit when done.
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
# Quant Researcher session
claude "Your role is Quant Researcher. Read PROJECT_BRIEF.md and the latest
backtest results, then analyze why our strategy underperforms SPY."

# Architect/Planner session
claude "Your role is Architect/Planner. Read TASKS.md and create tasks for
implementing momentum acceleration."

# Coding Agent session
claude "Read handoffs/handoff-IMPL-001.md and implement it"
```

## References

- Boris Cherny's agentic workflow: https://threadreaderapp.com/thread/2007179832300581177.html
- Adapted for parallel multi-agent execution with quant domain expertise
- Inspired by kanban / agile task management
