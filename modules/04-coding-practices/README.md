# Module 04: Coding with AI

## Objectives
By the end of this module you will be able to:
- Turn a coding problem into a reproducible prompt an AI assistant can
  actually solve correctly
- Recognize a common category of AI-introduced bug in scientific Python
  (accidentally forcing eager computation) and know the fix
- Require a test alongside any AI-generated fix, not just the fix itself
- Write a personal or lab coding style guide so AI-generated code matches
  your conventions instead of generic defaults
- Apply the same reproducibility principles in MATLAB and Bash, not just
  Python

## Why this matters in a research context
AI-generated code can be syntactically perfect and scientifically wrong —
quietly changing indexing, units, staggering, or memory behavior in ways
that don't throw an error. The fix isn't to avoid AI for code; it's to
prompt for reproducibility and verify with tests, every time.

## Core concept

### Make coding problems reproducible
A good coding prompt is usually a paragraph, not a sentence. Include:
- **Language, version, environment, and dependencies** — for HPC problems,
  this means modules loaded, compiler flags, allocation, and the exact
  launch command, not a narrative description.
- **The smallest complete code and the exact error** — logs and exact
  commands are far more useful to the model than a description of the
  problem.
- **Expected vs. observed behavior.**
- **Constraints that cannot change** — APIs, performance requirements,
  style, files that must stay untouched.
- **A request for tests, edge cases, and the commands to verify the
  result.**

If you find yourself writing all of this out repeatedly, that's a signal
to capture it once in a **design document** — a file describing your
project's conventions and constraints that you attach to every coding
session. (This overlaps directly with skill-writing from Module 03 — a
coding design document *is* a skill.)

### Worked example: a "bottleneck" that isn't what it looks like
Given this function and asked to analyze why it might be slow:

```python
import xarray as xr

def monthly_mean(path):
    ds = xr.open_mfdataset(
        path,
        chunks={"time": 30, "lat": 100, "lon": 100},
    )
    temperature = ds["temperature"].values  # !!
    return temperature.mean(axis=0)
```

The bug is the `.values` call. `xarray` + `dask` is designed to keep data
**lazy** — chunked and not actually loaded into memory — until you
explicitly ask for the computed result. `.values` converts the
`DataArray` to a NumPy array immediately, which can force the entire
(possibly huge) dataset into memory at once.

**Fix — preserve laziness until the caller explicitly wants it:**
```python
import xarray as xr

def monthly_mean(path):
    ds = xr.open_mfdataset(
        path,
        chunks={"time": 30, "lat": 100, "lon": 100},
    )
    # Returns a lazy, Dask-backed xarray.DataArray.
    return ds["temperature"].mean(dim="time")

result = monthly_mean("data/*.nc")
computed_result = result.compute()  # computation happens here, on demand
```

**Rule of thumb:** avoid `.values` (and similar eager conversions) inside
a function meant to return a lazy object — it silently changes the memory
behavior of everything downstream.

### Test what you ship
A fix without a test is a claim, not a verified result. For the example
above:

```python
# test_monthly_mean.py
import numpy as np
import xarray as xr
from dask.base import is_dask_collection
from my_module import monthly_mean

def test_monthly_mean_remains_lazy(tmp_path):
    # Create two small NetCDF files with distinct time coordinates.
    for i in range(2):
        ds = xr.Dataset(
            {"temperature": (("time", "lat", "lon"), np.full((2, 3, 4), i + 1.0))},
            coords={
                "time": np.arange(i * 2, i * 2 + 2),
                "lat": np.arange(3),
                "lon": np.arange(4),
            },
        )
        ds.to_netcdf(tmp_path / f"temperature_{i}.nc")

    result = monthly_mean(str(tmp_path / "temperature_*.nc"))

    # The underlying data should still be a lazy Dask array.
    assert is_dask_collection(result.data)

    computed = result.compute()

    # After computation, the underlying data should be a NumPy array.
    assert isinstance(computed.data, np.ndarray)
    np.testing.assert_allclose(computed.values, 1.5)
```
Run with `pytest -q`. Ask your AI assistant to generate this kind of test
*alongside* any fix — "also write a test that verifies X stays true" is a
one-line addition to your prompt.

### The long game: write a style/design guide
If you notice yourself repeating the same conventions in every prompt,
that's the signal to write a personal (or lab) coding design document —
the code equivalent of the skill files in Module 03. A reasonable set of
starting principles for data-intensive scientific code:

- Keep configuration explicit and centralized
- Normalize metadata and coordinates before analysis
- Keep data operations readable and reproducible
- Use small, reusable helper functions
- Document code for readability and reproducibility
- Write modular code that can be shared across scripts and functions
- Make outputs consistent and interpretable for both humans and
  downstream tools

Questions worth answering explicitly in your own guide: how do you want
functions to look? How many comments is enough? Does every function get a
test? (For that last one — yes, it should.) Once written, attach this
document to coding sessions the same way you'd attach a skill — it's what
keeps AI-generated code looking like *your* code instead of generic
output. For larger efforts, a design document (the plan) and an
implementation skill (the detailed how-to) can work together — the design
might be a few pages, the implementation skill much longer, covering
edge cases accumulated over real use.

### Beyond Python: the same principles hold

**MATLAB** — scientific code has invariants that "looks correct" can
silently break: indexing convention, nodal vs. cell-centered locations,
periodicity, boundary conditions. State them explicitly.

*Prompt:* "Write me a MATLAB function that takes five matrices as input.
The first two arguments are matrices of X and Y coordinates, the third
matrix is the values. The fourth and fifth arguments are different-sized
X and Y matrices of coordinates. The function is an interpolation
function." — paired with an explicit constraint like *fourth-order
accuracy required, public interface must not change* and a request to
*test dimensions, endpoints, and an analytic function; verify convergence
order.* A MATLAB style guide (naming, header/doc format, argument
validation conventions) helps here exactly the way a Python design
document does.

**Bash** — reproducibility constraints matter just as much for scripts.
A prompt for a script comparing two netCDF files should state: the tools
available (e.g. CDO, NCO, `ncdump`), that the script must be read-only
(must not modify either input file), and what "different" means (missing
variables vs. size mismatches vs. value differences) — otherwise the
model has to guess which kind of "diff" you want.

## Hands-on exercise
**Task:** Take a function you've written (or the `monthly_mean` example
above) and:
1. Write a reproducible prompt asking an AI assistant to analyze it for a
   specific concern (performance, memory, correctness) — include
   language/version/environment, the smallest complete code, and expected
   vs. observed behavior.
2. Ask for the fix **and** a test that verifies the fix.
3. Run the test yourself before accepting the change.

Do this in [exercise.ipynb](exercise.ipynb).

**Success looks like:** you have a passing test that specifically checks
the property you were worried about (not just "the function runs without
error") and you can explain in one sentence why the original code had the
problem.

## Key takeaways
- A good coding prompt is a paragraph: language/version/environment,
  minimal repro + exact error, expected vs. observed behavior, fixed
  constraints, and a request for tests
- In scientific Python, watch for operations (like `.values`) that force
  eager computation and silently change memory behavior
- Always pair an AI-generated fix with a test — a fix without a test is a
  claim, not a verified result
- Write a personal/lab coding design document once, then attach it to
  every session — it's a skill (Module 03) applied to code style
- The same reproducibility discipline (explicit environment, exact
  constraints, verification step) applies in MATLAB, Bash, or any other
  language — not just Python
