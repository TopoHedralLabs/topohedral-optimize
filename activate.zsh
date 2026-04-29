#!/bin/zsh
export RUSTDOCFLAGS="--html-in-header $(pwd)/docs/html/custom-header.html --document-private-items"
export TOPO_LOG='aug=trace,qn=trace,cg=trace,topohedral_optimize::constrained::augmented_lagrangian2=trace'
