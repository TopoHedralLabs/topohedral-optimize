# Building Docs Locally

The repository’s `build-docs.sh` script builds Rust API documentation, deploys
the MkDocs pages to a temporary local `mike` branch, injects the API docs, and
serves the result.

```bash
./build-docs.sh
./build-docs.sh v0.0.0
```

The site is served at `http://localhost:8000`. The script uses a temporary git
worktree and cleans it up when it exits. Rust documentation is generated with
`cargo doc --no-deps`; MkDocs dependencies are listed in
`docs/website/requirements.txt`.
