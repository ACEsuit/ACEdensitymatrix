#!/bin/bash

for order in 1 3; do
    for degree in $(seq 6 10); do
        julia --project=.. batchfit.jl --degree $degree --order $order > out.$degree.$order 2>&1 &
    done
done

wait