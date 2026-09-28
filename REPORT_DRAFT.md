# Reusing aggregation indices and preserving projection types in TQP-Inmem

Lyon — starter research report draft. Fresh-checkout verification and final
submission commit are pending; this draft does not claim either is complete.

## Motivation and approach

This work implements two Improvement tasks: Aggregation Performance and Data
Materialization Overhead. The objective is to reduce redundant tensor work
without changing the intended query answers. The changes are independent,
opt-in proof-of-concept paths, allowing original, aggregation-only,
materialization-only and combined configurations to be compared.

In `operators/aggregate.py`, TQP assigns rows to groups and evaluates aggregate
expressions. When rows/groups is at least 65,536, the existing implementation
uses a two-stage reduction. Its helper builds a subgroup number for every row
using `arange` and remainder, then computes a flat group/subgroup index. Before
this change, each reduction reconstructed those same indices. Q1 has several
sums and averages, including extra reductions for average denominators.

The aggregation change passes an optional dictionary through recursive
`evaluate()` calls. One dictionary is created per aggregation invocation and
stores flat indices by subgroup count. It is never shared across queries or
different grouping inputs. Reuse is enabled with `TQP_AGG_REUSE=1`; default zero
retains the original path. The hypothesis is that repeated reductions over the
same grouping benefit, while the single-stage path receives little benefit.

In `operators/project.py`, the original implementation remembers the expression
result type, converts the result to float64, copies it into pool storage, then
converts it back. For non-float64 results this can create redundant conversions
and temporary tensors. `TQP_PROJECT_NATIVE=1` skips the forced float64 conversion.
The existing `malloc_like` pool copy uses the actual result dtype; the final
conversion to that same dtype no longer requires changing types. The pool copy
is retained, avoiding a wider change to storage ownership and reuse lifetimes.

Initial testing exposed a non-contiguous int8 substring result. The old cast
had incidentally produced a compact copy; without it, `malloc_like` failed at
`subject_tensor.view(-1).numel()`. Flattening was unnecessary for sizing, so this
was replaced with `subject_tensor.numel()`. The allocated destination remains
compact and the existing assignment copies the strided source into it. This
failure and correction are part of the experiment, not omitted negative results.

## Experimental setting

Tests ran on an NVIDIA RTX 5080 (16303 MiB), driver 590.48.01, with an Intel Xeon
w5-2455X (12 cores/24 threads), Python 3.12.14 and PyTorch 2.14.0+cu130 (CUDA
runtime 13.0). The GPU was selected by UUID. The original cu126 installation
did not support this GPU; GPU arithmetic was verified after installing cu130.
The upstream 40 GB GPU pool was reduced to 8 GB for this machine, equally for
all modes. Data were generated at SF1 with CPU PyTorch 2.6.0 in a separate
environment, avoiding the mismatch between the installed CUDA compiler and
GPU PyTorch runtime. Exact source commits and setup commands are in REPRODUCE.md.

Each query timing configuration used five separate processes, each with two
whole-suite SF1 warmups before the measured queries. Profiling was disabled.
TQP's reported CSV time subtracts scan and output time; it is not total program
latency. The final four-way experiment rotated configuration order. Results
below are medians, not best times. Shared-machine utilization was not captured
systematically, so timing variability cannot be assigned to a specific cause.

## Results

An initial alternating comparison found Q1 median time decreased from 14.776 ms
to 10.933 ms with aggregation reuse (26.0% less time). Q13 and Q18 changed by
less than 1%. A separate materialization comparison found Q22 decreased from
6.328 ms to 6.127 ms (3.2%); Q1 was essentially unchanged. These are separate
batches and should not be pooled without accounting for their conditions.

The final four-way batch produced these median times in milliseconds:

| Query | Neither | Materialization | Aggregation | Both |
|---|---:|---:|---:|---:|
| Q1 | 16.842 | 16.796 | 12.930 | 11.133 |
| Q13 | 28.507 | 27.563 | 27.073 | 25.735 |
| Q18 | 20.988 | 18.211 | 19.688 | 16.076 |
| Q22 | 10.335 | 6.735 | 10.215 | 6.745 |

The final batch was noisy: original Q22 ranged from 6.953 to 11.570 ms.
Consequently the apparent large Q22 median difference is not a reliable precise
effect size. Likewise, improvements on other queries in this batch alone do
not establish causality. Repeated Q1 improvements and reduced operation counts
provide stronger evidence for aggregation reuse than these noisy timings alone.

Diagnostic Q1 profiles showed aggregation multiplication calls decreasing from
18 to 8 and addition calls from 12 to 2. These reductions are consistent with
eliminating ten reconstructions of flat subgroup indices. The top-five tables
do not establish every operation's exact count. Profile timings include
instrumentation and cold-start effects and are not used for the runtime claims.

A separate synthetic benchmark computes sum/average twice on resident GPU
tensors. It uses three warmups per mode and eight timed invocations, alternating
mode order with synchronization around host-clock timing. Input generation and
reference checks are outside timing. At four groups, 262,144 rows activates the
two-stage path: median time fell from 0.271941 to 0.183646 ms (32.5%). At
1,048,576 rows and four groups, it fell from 0.421779 to 0.257982 ms (38.8%).
At 65,536 rows with four or 64 groups, the path is inactive and median times
were nearly unchanged. At 262,144/64 and 1,048,576/64, differences were small
and mixed. There were individual outliers. This supports a rows-per-group
explanation, not an assertion that speedup grows with all increases in data
size. These are operator-scale experiments, not larger TPC-H scale factors.

Whole-query PyTorch peak allocated memory was essentially unchanged. Q1 used
8,493,014,528 bytes without reuse and 8,493,012,480 with reuse; the 2,048-byte
difference is not a meaningful saving. Q13 and Q18 were identical across modes
at 8,175,665,664 and 8,205,578,752 bytes. Q22 with/without native projection used
8,076,139,008 bytes in all three repetitions. These figures include the 8 GB
pool and exclude memory not accounted for by PyTorch. Avoided temporary
allocations do not necessarily lower the maximum reached elsewhere in a query.

## Correctness, limitations and next steps

The repository's SF1 test passed for aggregation-only, materialization-only and
combined modes. It compares formatted leading rows, not every query output row.
Additional aggregation tests performed 110 full-vector comparisons spanning
integer/float data, threshold boundaries, skew, null counts and index reuse.
Ten materialization checks covered sliced allocation/copy, substring projection,
integer/decimal/Boolean outputs and large-integer preservation in native mode.
Synthetic scaling outputs also matched direct reference sums and means.

These checks are not a full SQL correctness proof: empty inputs/groups,
negative-value min/max semantics, broader null semantics and hash collisions
remain outside coverage. Native typing can change historical rounding behavior
for integers beyond float64's exact range; the focused native test verifies
preservation rather than claiming equivalence to an inaccurate round trip.

Reuse keeps an index tensor live longer and offers little benefit outside the
two-stage path. Native projection does not benefit expressions already in
float64 and retains the pool copy. Future work should measure isolated
projection conversion traffic, expand full-output correctness checks, improve
empty-result dtype handling, repeat on a controlled idle machine, and test
larger TPC-H scales. The initial evidence supports a targeted aggregation
optimization and a smaller materialization proof of concept, not a universal
speedup or reduction in peak GPU memory.

## Reproduction and delivery status

REPRODUCE.md contains environment/data setup, both switches, correctness,
profiling, memory, scaling and four-way runtime commands. Both switches default
off. Raw logs/CSVs should accompany the final branch. Before delivery, record
the submitted commit, capture the final environment, verify the documented
fresh-checkout workflow, and render this draft into the requested 2–4 pages.
