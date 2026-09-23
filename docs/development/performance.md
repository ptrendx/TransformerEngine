## Performance Changes

Use the relevant program under `benchmarks/` and report its exact command. A
performance result must identify the selected TE backend or kernel, GPU model
and count, software versions, shapes, layouts, dtypes, recipes, warm-up,
synchronization, input distribution, and measurement statistic. Separate
compile or autotune time, Python launch overhead, communication, and
steady-state kernel time where they materially affect the result.

Compare on equivalent hardware and software configurations and confirm
numerical correctness before accepting a faster result. Do not place one-off
performance measurements in correctness tests.
