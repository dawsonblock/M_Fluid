```markdown
# M_Fluid Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and conventions used in the M_Fluid Python codebase. You will learn how to structure files, write and organize code, follow commit message conventions, and understand the repository's approach to testing and workflows. This knowledge will help you contribute code that is consistent, maintainable, and easy for others to review.

## Coding Conventions

### File Naming
- Use **snake_case** for all Python files.
  - Example: `fluid_solver.py`, `boundary_conditions.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import compute_pressure
    from . import constants
    ```

### Export Style
- Use **named exports** (explicitly define what is exported).
  - Example:
    ```python
    __all__ = ['FluidSolver', 'BoundaryCondition']
    ```

### Commit Messages
- Use the `feat` prefix for new features.
- Commit messages are concise, averaging around 59 characters.
  - Example:
    ```
    feat: add pressure solver for incompressible flow
    ```

## Workflows

### Adding a New Feature
**Trigger:** When implementing a new capability or module  
**Command:** `/add-feature`

1. Create a new Python file using snake_case naming.
2. Implement the feature using relative imports as needed.
3. Define `__all__` to specify exported classes/functions.
4. Write or update tests for the new feature.
5. Commit changes with a message starting with `feat:`.
6. Submit a pull request for review.

### Refactoring Existing Code
**Trigger:** When improving code structure or readability  
**Command:** `/refactor-code`

1. Identify the code to refactor.
2. Update code while preserving existing functionality.
3. Ensure imports remain relative and exports are named.
4. Run all relevant tests to confirm no regressions.
5. Commit changes with a descriptive message.
6. Submit for review.

## Testing Patterns

- **Framework:** Unknown (not explicitly detected).
- **File Pattern:** Test files are named with the pattern `*.test.ts`, suggesting some TypeScript-based tests may exist, possibly for frontend or API validation.
- **Best Practice:** Ensure each feature or module has a corresponding test file. Place tests in a dedicated directory or alongside the module.
  - Example test file: `fluid_solver.test.ts`

## Commands

| Command         | Purpose                                 |
|-----------------|-----------------------------------------|
| /add-feature    | Start the workflow for adding a feature |
| /refactor-code  | Begin refactoring existing code         |
```
