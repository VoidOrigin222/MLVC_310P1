# C++ Style

The project follows the Google C++ Style Guide through the repository
`.clang-format` file:

- C++17
- two-space indentation
- 100-column limit
- left-aligned pointers
- regrouped, case-sensitive includes

Format all project C++ files with:

```bash
./scripts/format_cpp.sh
```

For CI or a pre-commit check, use the same file list with
`clang-format --dry-run --Werror`. Formatting must not be mixed with behavior
changes in the same logical edit. Resource ownership uses RAII and standard
library smart pointers; worker threads must have an explicit shutdown and join
path.
