```markdown
# M_Fluid Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you the core development patterns and conventions used in the **M_Fluid** TypeScript codebase. You'll learn how to structure files, write imports/exports, follow commit message guidelines, and organize tests. These patterns help maintain consistency and readability throughout the project.

## Coding Conventions

### File Naming
- Use **snake_case** for all file names.

**Example:**
```plaintext
fluid_solver.ts
vector_utils.ts
```

### Import Style
- Use **relative imports** for referencing other files.

**Example:**
```typescript
import { computePressure } from './pressure_solver';
import { Vector2D } from '../math/vector2d';
```

### Export Style
- Use **named exports** for functions, classes, and constants.

**Example:**
```typescript
// In vector_utils.ts
export function addVectors(a: Vector2D, b: Vector2D): Vector2D { ... }

export const ZERO_VECTOR: Vector2D = { x: 0, y: 0 };
```

### Commit Messages
- Follow **conventional commits** style.
- Use the `chore` prefix for routine changes.
- Keep commit messages concise (average ~56 characters).

**Example:**
```plaintext
chore: update dependencies to latest versions
chore: fix typo in fluid_solver.ts
```

## Workflows

### Code Update
**Trigger:** When making changes to the codebase  
**Command:** `/code-update`

1. Make your code changes following the coding conventions.
2. Write a commit message using the conventional commit format (e.g., `chore: ...`).
3. Commit your changes.

### Add a Test
**Trigger:** When adding or updating features  
**Command:** `/add-test`

1. Create a test file using the `*.test.*` pattern (e.g., `fluid_solver.test.ts`).
2. Write tests for your new or updated code.
3. Run the test suite (framework unknown; see project documentation or scripts).

## Testing Patterns

- Test files use the `*.test.*` naming pattern (e.g., `vector_utils.test.ts`).
- The test framework is not explicitly detected; check project scripts or documentation for details.
- Place tests alongside the code or in a dedicated test directory as per project structure.

**Example:**
```typescript
// vector_utils.test.ts
import { addVectors } from './vector_utils';

test('addVectors adds two vectors correctly', () => {
  expect(addVectors({ x: 1, y: 2 }, { x: 3, y: 4 })).toEqual({ x: 4, y: 6 });
});
```

## Commands
| Command        | Purpose                                   |
|----------------|-------------------------------------------|
| /code-update   | Make and commit code changes               |
| /add-test      | Add or update a test file                  |
```
