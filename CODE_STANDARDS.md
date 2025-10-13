# Code Standards and Guidelines

This document outlines the code standards and best practices for this project.

## Line Endings

**Standard**: Unix-style line endings (LF, `\n`) for all files.

**Why**: Ensures consistency across different operating systems and prevents `^M` (CRLF) characters from appearing in files.

**Enforcement**:
- `.editorconfig` automatically configures IDEs to use LF line endings
- `.pre-commit-config.yaml` with `mixed-line-ending` hook checks and fixes line endings before commits

**Manual Fix** (if needed):
```bash
# Fix line endings in a file
sed -i '' 's/\r$//' filename.py

# Check file line endings
file filename.py
```

## Python Code Style

### Formatting
- **Indentation**: 4 spaces
- **Line Length**: Maximum 120 characters
- **Formatter**: Black (automatically formats code)
- **Import Sorting**: isort with Black profile

### Testing
- **Framework**: pytest with pytest-mock
- **Test Location**: `backend/tests/`
- **Fixtures**: Shared fixtures in `backend/tests/conftest.py`
- **Naming**: Test files must start with `test_`

### Test Best Practices
1. **Use Shared Fixtures**: Avoid duplicate mock setup code
   ```python
   # Good: Use fixtures from conftest.py
   def test_something(self, ai_generator_factory, mock_vector_store):
       generator = ai_generator_factory()
       # ... test code

   # Bad: Duplicate setup in each test
   def test_something(self):
       generator = AIGenerator(
           endpoint="...",
           api_key="...",
           # ... repeated parameters
       )
   ```

2. **Factory Fixtures for Mocked Dependencies**: Use factory fixtures when tests use `@patch` decorators
   ```python
   @pytest.fixture
   def ai_generator_factory():
       """Factory to create AIGenerator instances"""
       def _create():
           return AIGenerator(endpoint="...", api_key="...")
       return _create
   ```

3. **Centralize Helper Functions**: Put reusable helpers in `conftest.py`
   ```python
   def create_mock_response(content, tool_calls=None):
       """Helper to create a mock Azure OpenAI response"""
       # Implementation
   ```

## Editor Configuration

### EditorConfig Support
The `.editorconfig` file automatically configures your IDE with project standards:
- Line endings (LF)
- Character encoding (UTF-8)
- Indentation (spaces vs tabs, size)
- Trailing whitespace removal

**Supported IDEs**: VS Code, PyCharm, IntelliJ, Sublime Text, Atom, and more.

**Setup**: Most modern IDEs have built-in EditorConfig support or plugins available.

## Pre-commit Hooks

### Installation
```bash
# Install pre-commit
uv add pre-commit --dev

# Install the git hooks
pre-commit install
```

### What Gets Checked
1. **File Checks**:
   - Trailing whitespace removal
   - End-of-file newlines
   - Line ending consistency (enforce LF)
   - Large file detection (>1MB)
   - YAML/JSON syntax validation
   - Merge conflict markers

2. **Python Checks**:
   - Black formatting
   - isort import sorting
   - Flake8 linting (max line length: 120)

### Running Manually
```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run
```

## Package Management

**Tool**: Always use `uv`, never `pip` directly.

```bash
# Install dependencies
uv sync

# Add a package
uv add <package-name>

# Add a dev dependency
uv add <package-name> --dev

# Remove a package
uv remove <package-name>
```

## Project Structure

```
starting-ragchatbot-codebase/
├── .editorconfig              # IDE configuration
├── .pre-commit-config.yaml    # Pre-commit hooks
├── CODE_STANDARDS.md          # This file
├── CLAUDE.md                  # Claude Code instructions
├── backend/
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py        # Shared fixtures and helpers
│   │   ├── test_search_tools.py
│   │   ├── test_ai_generator.py
│   │   ├── test_rag_system.py
│   │   ├── ANALYSIS_REPORT.md
│   │   └── REFACTORING_SUMMARY.md
│   ├── config.py
│   ├── rag_system.py
│   ├── search_tools.py
│   ├── ai_generator.py
│   └── ...
└── ...
```

## Testing Workflow

1. **Write Tests**: Add tests to appropriate file in `backend/tests/`
2. **Use Fixtures**: Leverage shared fixtures from `conftest.py`
3. **Run Tests**: `uv run pytest tests/ -v`
4. **Check Coverage**: `uv run pytest tests/ --cov=. --cov-report=html`
5. **Format Code**: Pre-commit hooks handle this automatically
6. **Commit**: Git hooks enforce standards before commit

## Common Issues

### Issue: Tests fail with import errors
**Solution**: Ensure `sys.path` is set correctly in test files:
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
```

### Issue: CRLF line endings appear
**Solution**:
1. Install and activate pre-commit hooks: `pre-commit install`
2. Run manually: `pre-commit run --all-files`
3. Or fix manually: `sed -i '' 's/\r$//' filename`

### Issue: Pre-commit hooks fail
**Solution**:
1. Review the error messages
2. Fix formatting issues: `black backend/`
3. Fix import order: `isort backend/`
4. Run hooks again: `pre-commit run --all-files`

## Best Practices Summary

✅ **DO**:
- Use Unix line endings (LF)
- Use shared fixtures from `conftest.py`
- Use factory fixtures for mocked dependencies
- Use `uv` for package management
- Run tests before committing
- Let pre-commit hooks auto-format code

❌ **DON'T**:
- Use Windows line endings (CRLF)
- Duplicate mock setup code in tests
- Use `pip` directly (use `uv` instead)
- Commit without running tests
- Manually format code (let Black do it)

## References

- **EditorConfig**: https://editorconfig.org/
- **Pre-commit**: https://pre-commit.com/
- **Black**: https://black.readthedocs.io/
- **isort**: https://pycqa.github.io/isort/
- **pytest**: https://docs.pytest.org/
- **uv**: https://github.com/astral-sh/uv
