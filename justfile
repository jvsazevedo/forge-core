# Forge Core justfile
# Run `just --list` to see all available recipes.

default:
    @just --list

# Fast type-check without linking
check:
    cargo check

# Build all workspace members
build:
    cargo build

# Build in release mode (produces libforge_core.so)
build-release:
    cargo build --release

# Run all tests
test:
    cargo test

# Run only smoke tests (forge-core)
smoke:
    cargo test -p forge-core -- --nocapture

# Run the MLP proof-of-concept
run-mlp:
    cargo run -p mlp-poc

# Clean all build artifacts
clean:
    cargo clean
