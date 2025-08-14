#!/bin/zsh
export RUSTDOCFLAGS="--html-in-header $(pwd)/docs/html/custom-header.html --document-private-items"
export TOPO_LOG=cg=debug
