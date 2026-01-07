# CLAUDE_CONTEXT.md

**Last Updated:** January 6, 2026  
**Project:** QuantETF

-----

## How to Use This File

This file contains instructions and context for Claude across all sessions. Read this file at the start of every session by fetching:

```
https://raw.githubusercontent.com/mikemsq/QuantETF/main/CLAUDE_CONTEXT.md
```

Always check PROGRESS_LOG.md for current status.

-----

## Project Overview

**What we’re building:**  
QuantETF - [Add one sentence description of what QuantETF does]

**Current phase:**  
Planning / Initial Development

**Tech stack:**

- Frontend: [e.g., React, Vue, or vanilla JS]
- Backend: [e.g., Python/Flask, Node.js, or serverless]
- Database: [e.g., PostgreSQL, MongoDB, or Firebase]
- Data Sources: [e.g., APIs for ETF/market data]
- Hosting: [e.g., Vercel, AWS, Heroku]

-----

## Project Files Structure

```
QuantETF/
├── PROJECT_BRIEF.md          # Strategic overview and goals
├── CLAUDE_CONTEXT.md         # This file - read first
├── PROGRESS_LOG.md           # Daily updates - check current status
├── README.md                 # Public-facing documentation
├── /docs                     # Additional documentation
│   └── /decisions            # Architecture decision records
├── /session-notes            # Notes from each Claude session
├── /src                      # Source code
│   ├── /components           # UI components
│   ├── /services             # Business logic/API calls
│   └── /utils                # Helper functions
└── /tests                    # Test files
```

-----

## Core Principles

### 1. Always verify before implementing

- Read the full requirements before starting
- Ask clarifying questions if anything is ambiguous
- Propose a plan before writing code
- Consider data accuracy and financial calculation precision

### 2. Keep it simple

- Use the simplest solution that works
- Avoid over-engineering
- Write clear, readable code over clever code
- Financial calculations should be transparent and verifiable

### 3. Document as you go

- Update PROGRESS_LOG.md after completing tasks
- Add comments for complex financial logic
- Document decisions in /docs/decisions/
- Explain data sources and calculation methods

### 4. Test your work

- Verify financial calculations with known test cases
- Test edge cases (market closures, missing data, etc.)
- [Add specific testing commands once established]
- Validate data accuracy against source

-----

## Coding Standards

### Style Guide

- **Language:** [Specify: JavaScript/TypeScript/Python]
- **Formatting:** 2 spaces, semicolons (if JS), single quotes
- **Naming conventions:**
  - Variables: camelCase (e.g., `portfolioValue`)
  - Functions: camelCase (e.g., `calculateReturns`)
  - Components: PascalCase (e.g., `ETFDashboard`)
  - Constants: UPPER_SNAKE_CASE (e.g., `API_KEY`, `MAX_RETRIES`)
  - Files: kebab-case.js (e.g., `portfolio-calculator.js`)

### Code Organization

- One component/function per file when reasonable
- Keep files under 200 lines when possible
- Group related functionality together
- Separate data fetching from calculation logic
- Keep UI separate from business logic

### Comments

- Write self-documenting code first
- Add comments for “why”, not “what”
- **Always explain financial formulas and calculations**
- Document data source expectations and formats
- Use JSDoc/docstrings for functions

-----

## Things to ALWAYS Do

✅ Read PROGRESS_LOG.md before starting work  
✅ Check if similar code exists before creating new patterns  
✅ Run tests before marking work complete  
✅ Update PROGRESS_LOG.md when finishing a task  
✅ Create small, focused commits with clear messages  
✅ Add error handling for API calls and data fetching  
✅ Validate financial data before calculations  
✅ Consider edge cases (missing data, market holidays, API failures)  
✅ Use precise number types for financial calculations (avoid floating point errors)  
✅ Document data sources and assumptions

-----

## Things to NEVER Do

❌ Don’t modify files outside the current task scope  
❌ Don’t delete or comment out code without discussion  
❌ Don’t introduce new dependencies without noting in PROGRESS_LOG.md  
❌ Don’t commit broken/non-functional code  
❌ Don’t ignore errors or warnings from APIs  
❌ Don’t hardcode API keys or sensitive credentials  
❌ Don’t skip data validation steps  
❌ Don’t use imprecise number types for money/percentages  
❌ Don’t make financial calculations without documenting the formula  
❌ Don’t cache stale financial data without timestamps

-----

## Common Patterns

### Pattern 1: API Data Fetching

```javascript
// Example of how we handle ETF/market data API calls
async function fetchETFData(ticker) {
  try {
    const response = await fetch(`${API_BASE_URL}/etf/${ticker}`);
    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`);
    }
    const data = await response.json();
    
    // Validate required fields
    if (!data.price || !data.timestamp) {
      throw new Error('Invalid data format');
    }
    
    return data;
  } catch (error) {
    console.error(`Error fetching ETF data for ${ticker}:`, error);
    // Handle gracefully - show cached data or user message
    throw error;
  }
}
```

### Pattern 2: Financial Calculations

```javascript
// Always document formulas and handle precision
/**
 * Calculate portfolio total return
 * Formula: (Current Value - Initial Value) / Initial Value * 100
 * @param {number} currentValue - Current portfolio value
 * @param {number} initialValue - Initial investment
 * @returns {number} Return percentage (e.g., 15.5 for 15.5%)
 */
function calculateReturn(currentValue, initialValue) {
  if (initialValue === 0) {
    throw new Error('Initial value cannot be zero');
  }
  
  // Use precise decimal library if needed for production
  const returnValue = ((currentValue - initialValue) / initialValue) * 100;
  return Number(returnValue.toFixed(2));
}
```

### Pattern 3: Error Handling

```javascript
// Show user-friendly messages for financial data errors
function handleDataError(error, context) {
  const userMessage = {
    'API_UNAVAILABLE': 'Market data temporarily unavailable. Please try again.',
    'INVALID_TICKER': 'ETF ticker not found. Please check the symbol.',
    'RATE_LIMIT': 'Too many requests. Please wait a moment.',
    'STALE_DATA': 'Data may be delayed. Last updated: [timestamp]'
  };
  
  console.error(`Error in ${context}:`, error);
  return userMessage[error.code] || 'An error occurred. Please try again.';
}
```

-----

## Common Mistakes (Learn from these!)

### Issue: [Issues will be added as they occur]

**What happened:** [Brief description]  
**Why it happened:** [Root cause]  
**Solution:** [How to avoid it]  
**Date added:** [Date]

-----

## Financial Data Guidelines

### Data Accuracy

- Always validate data from external APIs
- Cross-reference critical calculations
- Handle missing data gracefully (show “N/A” not “0”)
- Display timestamps with all financial data
- Indicate when data is delayed or estimated

### Number Precision

- Use appropriate decimal precision (2 decimals for currency, 4 for percentages)
- Be aware of JavaScript floating point limitations
- Consider using a decimal library for critical calculations
- Always round appropriately for display

### Date/Time Handling

- Use consistent timezone (preferably UTC for storage, local for display)
- Handle market hours and holidays
- Consider international markets if applicable
- Store timestamps with all time-sensitive data

-----

## Session Workflow

When starting a new session:

1. **Load context:**
- Read this file (CLAUDE_CONTEXT.md)
- Read PROGRESS_LOG.md for current status
- Check any relevant session notes from /session-notes/
1. **Understand the task:**
- Review the goal clearly
- Ask clarifying questions about requirements
- Confirm data sources and calculation methods
- Propose a plan
1. **Implement:**
- Follow coding standards above
- Write clean, tested code
- Document financial logic clearly
- Handle errors gracefully
1. **Verify:**
- Test calculations with known values
- Check edge cases
- Verify data accuracy
- Ensure requirements are met
1. **Document:**
- Update PROGRESS_LOG.md
- Create session note in /session-notes/
- Note any learnings for CLAUDE_CONTEXT.md
- Document any new patterns or mistakes

-----

## External Resources

### Documentation

- [Link to financial data API documentation]
- [Link to framework/library documentation]
- [Link to ETF reference materials]

### Data Sources

- [List approved data sources for ETF information]
- [API endpoints an