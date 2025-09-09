#!/usr/bin/bash

for dir in workload_1 workload_2 workload_3 workload_4; do
    if [ -d "$dir" ] && [ -x "$dir/test_expansions.sh" ]; then
        (cd "$dir" && ./test_expansions.sh)
    fi
done