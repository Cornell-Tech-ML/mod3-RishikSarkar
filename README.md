# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## 3.1 and 3.2 NUMBA Diagnostics Output

### MAP
```
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
D:\Cornell\Academic\Fall 2024\Machine Learning
Engineering\mod3-RishikSarkar\minitorch\fast_ops.py (163)
================================================================================

Parallel loop listing for  Function tensor_map.<locals>._map, D:\Cornell\Academic\Fall 2024\Machine Learning Engineering\mod3-RishikSarkar\minitorch\fast_ops.py (163)
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        in_storage: Storage,                                                 |
        in_shape: Shape,                                                     |
        in_strides: Strides,                                                 |
    ) -> None:                                                               |
        # Optimized path for identical shapes and strides                    |
        out_size = np.prod(out_shape)----------------------------------------| #2
        if np.array_equal(in_shape, out_shape) and np.array_equal(           |
            in_strides, out_strides                                          |
        ):                                                                   |
            for i in prange(out_size):---------------------------------------| #3
                out[i] = fn(in_storage[i])                                   |
        else:                                                                |
            for i in prange(out_size):---------------------------------------| #4
                out_index = np.zeros(len(out_shape), dtype=np.int32)---------| #0
                in_index = np.zeros(len(in_shape), dtype=np.int32)-----------| #1
                                                                             |
                to_index(i, out_shape, out_index)                            |
                broadcast_index(out_index, out_shape, in_shape, in_index)    |
                                                                             |
                position = index_to_position(in_index, in_strides)           |
                out[i] = fn(in_storage[position])                            |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #2, #3, #4, #0, #1).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--4 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--4 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--4 (parallel)
   +--0 (serial)
   +--1 (serial)

Parallel region 0 (loop #4) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#4).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at D:\Cornell\Academic\Fall
2024\Machine Learning Engineering\mod3-RishikSarkar\minitorch\fast_ops.py (180)
is hoisted out of the parallel loop labelled #4 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at D:\Cornell\Academic\Fall
2024\Machine Learning Engineering\mod3-RishikSarkar\minitorch\fast_ops.py (181)
is hoisted out of the parallel loop labelled #4 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: in_index = np.zeros(len(in_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
```

### ZIP
```
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
D:\Cornell\Academic\Fall 2024\Machine Learning
Engineering\mod3-RishikSarkar\minitorch\fast_ops.py (215)
================================================================================

Parallel loop listing for  Function tensor_zip.<locals>._zip, D:\Cornell\Academic\Fall 2024\Machine Learning Engineering\mod3-RishikSarkar\minitorch\fast_ops.py (215)
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              |
        out: Storage,                                                      |
        out_shape: Shape,                                                  |
        out_strides: Strides,                                              |
        a_storage: Storage,                                                |
        a_shape: Shape,                                                    |
        a_strides: Strides,                                                |
        b_storage: Storage,                                                |
        b_shape: Shape,                                                    |
        b_strides: Strides,                                                |
    ) -> None:                                                             |
        # Optimized path if all shapes and strides match                   |
        if (                                                               |
            np.array_equal(a_strides, out_strides)                         |
            and np.array_equal(b_strides, out_strides)                     |
            and np.array_equal(a_shape, out_shape)                         |
            and np.array_equal(b_shape, out_shape)                         |
        ):                                                                 |
            for i in prange(len(out)):-------------------------------------| #8
                out[i] = fn(a_storage[i], b_storage[i])                    |
        else:                                                              |
            for i in prange(len(out)):-------------------------------------| #9
                a_index = np.zeros(len(a_shape), dtype=np.int32)-----------| #5
                b_index = np.zeros(len(b_shape), dtype=np.int32)-----------| #6
                out_index = np.zeros(len(out_shape), dtype=np.int32)-------| #7
                                                                           |
                to_index(i, out_shape, out_index)                          |
                broadcast_index(out_index, out_shape, a_shape, a_index)    |
                broadcast_index(out_index, out_shape, b_shape, b_index)    |
                                                                           |
                a_pos = index_to_position(a_index, a_strides)              |
                b_pos = index_to_position(b_index, b_strides)              |
                                                                           |
                out[i] = fn(a_storage[a_pos], b_storage[b_pos])            |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #8, #9, #5, #6, #7).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--9 is a parallel loop
   +--5 --> rewritten as a serial loop
   +--6 --> rewritten as a serial loop
   +--7 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--9 (parallel)
   +--5 (parallel)
   +--6 (parallel)
   +--7 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--9 (parallel)
   +--5 (serial)
   +--6 (serial)
   +--7 (serial)

Parallel region 0 (loop #9) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#9).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at D:\Cornell\Academic\Fall
2024\Machine Learning Engineering\mod3-RishikSarkar\minitorch\fast_ops.py (237)
is hoisted out of the parallel loop labelled #9 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: a_index = np.zeros(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at D:\Cornell\Academic\Fall
2024\Machine Learning Engineering\mod3-RishikSarkar\minitorch\fast_ops.py (238)
is hoisted out of the parallel loop labelled #9 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: b_index = np.zeros(len(b_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at D:\Cornell\Academic\Fall
2024\Machine Learning Engineering\mod3-RishikSarkar\minitorch\fast_ops.py (239)
is hoisted out of the parallel loop labelled #9 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
```

### REDUCE
```
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
D:\Cornell\Academic\Fall 2024\Machine Learning
Engineering\mod3-RishikSarkar\minitorch\fast_ops.py (274)
================================================================================

Parallel loop listing for  Function tensor_reduce.<locals>._reduce, D:\Cornell\Academic\Fall 2024\Machine Learning Engineering\mod3-RishikSarkar\minitorch\fast_ops.py (274)
-----------------------------------------------------------------------|loop #ID
    def _reduce(                                                       |
        out: Storage,                                                  |
        out_shape: Shape,                                              |
        out_strides: Strides,                                          |
        a_storage: Storage,                                            |
        a_shape: Shape,                                                |
        a_strides: Strides,                                            |
        reduce_dim: int,                                               |
    ) -> None:                                                         |
        dim_len = a_shape[reduce_dim]                                  |
        start = out[0]                                                 |
                                                                       |
        for i in prange(np.prod(out_shape)):---------------------------| #13, 12
            out_index = np.zeros(len(out_shape), dtype=np.int32)-------| #10
            a_index = np.zeros(len(a_shape), dtype=np.int32)-----------| #11
                                                                       |
            to_index(i, out_shape, out_index)                          |
                                                                       |
            # Reduce across dimension                                  |
            acc = start                                                |
            for j in range(dim_len):                                   |
                for k in range(len(out_shape)):                        |
                    a_index[k] = out_index[k]                          |
                a_index[reduce_dim] = j                                |
                a_pos = index_to_position(a_index, a_strides)          |
                acc = fn(acc, a_storage[a_pos])                        |
                                                                       |
            out[i] = acc                                               |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #12, #13, #10, #11).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--13 is a parallel loop
   +--10 --> rewritten as a serial loop
   +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--10 (parallel)
   +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--10 (serial)
   +--11 (serial)

Parallel region 0 (loop #13) had 0 loop(s) fused and 2 loop(s) serialized as
part of the larger parallel loop (#13).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at D:\Cornell\Academic\Fall
2024\Machine Learning Engineering\mod3-RishikSarkar\minitorch\fast_ops.py (287)
is hoisted out of the parallel loop labelled #13 (it will be performed before
the loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at D:\Cornell\Academic\Fall
2024\Machine Learning Engineering\mod3-RishikSarkar\minitorch\fast_ops.py (288)
is hoisted out of the parallel loop labelled #13 (it will be performed before
the loop is executed and reused inside the loop):
   Allocation:: a_index = np.zeros(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
```

### MATRIX MULTIPLY
```
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
D:\Cornell\Academic\Fall 2024\Machine Learning
Engineering\mod3-RishikSarkar\minitorch\fast_ops.py (306)
================================================================================

Parallel loop listing for  Function _tensor_matrix_multiply, D:\Cornell\Academic\Fall 2024\Machine Learning Engineering\mod3-RishikSarkar\minitorch\fast_ops.py (306)
------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                            |
    out: Storage,                                                       |
    out_shape: Shape,                                                   |
    out_strides: Strides,                                               |
    a_storage: Storage,                                                 |
    a_shape: Shape,                                                     |
    a_strides: Strides,                                                 |
    b_storage: Storage,                                                 |
    b_shape: Shape,                                                     |
    b_strides: Strides,                                                 |
) -> None:                                                              |
    """NUMBA tensor matrix multiply function.                           |
                                                                        |
    Should work for any tensor shapes that broadcast as long as         |
                                                                        |
    ```                                                                 |
    assert a_shape[-1] == b_shape[-2]                                   |
    ```                                                                 |
                                                                        |
    Optimizations:                                                      |
                                                                        |
    * Outer loop in parallel                                            |
    * No index buffers or function calls                                |
    * Inner loop should have no global writes, 1 multiply.              |
                                                                        |
                                                                        |
    Args:                                                               |
    ----                                                                |
        out (Storage): storage for `out` tensor                         |
        out_shape (Shape): shape for `out` tensor                       |
        out_strides (Strides): strides for `out` tensor                 |
        a_storage (Storage): storage for `a` tensor                     |
        a_shape (Shape): shape for `a` tensor                           |
        a_strides (Strides): strides for `a` tensor                     |
        b_storage (Storage): storage for `b` tensor                     |
        b_shape (Shape): shape for `b` tensor                           |
        b_strides (Strides): strides for `b` tensor                     |
                                                                        |
    Returns:                                                            |
    -------                                                             |
        None : Fills in `out`                                           |
                                                                        |
    """                                                                 |
    assert a_shape[-1] == b_shape[-2], "Shapes do not match!"           |
                                                                        |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0              |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0              |
                                                                        |
    for batch in prange(out_shape[0]):----------------------------------| #14
        for row in range(out_shape[-2]):                                |
            for col in range(out_shape[-1]):                            |
                # Calculate base positions once                         |
                a_pos = batch * a_batch_stride + row * a_strides[-2]    |
                b_pos = batch * b_batch_stride + col * b_strides[-1]    |
                                                                        |
                accumulator = 0.0                                       |
                for k in range(a_shape[-1]):                            |
                    accumulator += (                                    |
                        a_storage[a_pos + k * a_strides[-1]]            |
                        * b_storage[b_pos + k * b_strides[-2]]          |
                    )                                                   |
                                                                        |
                out_pos = (                                             |
                    batch * out_strides[0]                              |
                    + row * out_strides[-2]                             |
                    + col * out_strides[-1]                             |
                )                                                       |
                out[out_pos] = accumulator                              |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #14).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

---

## 3.3 + 3.4 CUDA Tests

### 3.3 Tests
```console
======================================= test session starts ========================================
platform linux -- Python 3.12.7, pytest-8.3.2, pluggy-1.5.0
rootdir: /content/mod3-RishikSarkar
configfile: pyproject.toml
plugins: hypothesis-6.54.0, env-1.1.4
collected 117 items / 60 deselected / 57 selected

tests/test_tensor_general.py .........................................................       [100%]

========================================= warnings summary =========================================
tests/test_tensor_general.py: 16 warnings
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py: 4268 warnings
  /usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py: 11 warnings
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_args[cuda-fn7]
tests/test_tensor_general.py::test_two_args[cuda-fn5]
tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
tests/test_tensor_general.py::test_one_derivative[cuda-fn6]
tests/test_tensor_general.py::test_sum_practice2
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 3 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 6 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 18 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 27 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn0]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 9 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn1]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn3]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 12 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_one_derivative[cuda-fn4]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_sum_practice_other_dims
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=================== 57 passed, 60 deselected, 4309 warnings in 185.52s (0:03:05) ===================
```

### 3.4 Tests
```console
======================================= test session starts ========================================
platform linux -- Python 3.12.7, pytest-8.3.2, pluggy-1.5.0
rootdir: /content/mod3-RishikSarkar
configfile: pyproject.toml
plugins: hypothesis-6.54.0, env-1.1.4
collected 117 items / 110 deselected / 7 selected

tests/test_tensor_general.py .......                                                         [100%]

========================================= warnings summary =========================================
tests/test_tensor_general.py::test_mul_practice1
tests/test_tensor_general.py::test_mul_practice3
tests/test_tensor_general.py::test_mul_practice3
tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py: 111 warnings
  /usr/local/lib/python3.12/dist-packages/numba/cuda/cudadrv/devicearray.py:888: NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_mul_practice4
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 35 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_mul_practice4
tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 4 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_mul_practice5
tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 8 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 3 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 24 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 12 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 36 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 18 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 27 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 64 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 5 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 6 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

tests/test_tensor_general.py::test_bmm[cuda]
  /usr/local/lib/python3.12/dist-packages/numba/cuda/dispatcher.py:536: NumbaPerformanceWarning: Grid size 48 will likely result in GPU under-utilization due to low occupancy.
    warn(NumbaPerformanceWarning(msg))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========================= 7 passed, 110 deselected, 140 warnings in 12.31s =========================
```

### 3.4 Matrix Multiplication Timing Results (using timing.py)
```
Running size 64
{'fast': np.float64(0.004395484924316406), 'gpu': np.float64(0.013965765635172525)}
Running size 128
{'fast': np.float64(0.017716646194458008), 'gpu': np.float64(0.023818174997965496)}
Running size 256
{'fast': np.float64(0.10421435038248698), 'gpu': np.float64(0.06492956479390462)}
Running size 512
{'fast': np.float64(0.4929521083831787), 'gpu': np.float64(0.21951794624328613)}
Running size 1024
{'fast': np.float64(4.474050760269165), 'gpu': np.float64(0.8294498125712076)}

Timing summary
Size: 64
    fast: 0.00440
{'fast': np.float64(0.10421435038248698), 'gpu': np.float64(0.06492956479390462)}
Running size 512
{'fast': np.float64(0.4929521083831787), 'gpu': np.float64(0.21951794624328613)}
Running size 1024
{'fast': np.float64(4.474050760269165), 'gpu': np.float64(0.8294498125712076)}

Timing summary
Size: 64
    fast: 0.00440
{'fast': np.float64(0.4929521083831787), 'gpu': np.float64(0.21951794624328613)}
Running size 1024
{'fast': np.float64(4.474050760269165), 'gpu': np.float64(0.8294498125712076)}

Timing summary
Size: 64
    fast: 0.00440
{'fast': np.float64(4.474050760269165), 'gpu': np.float64(0.8294498125712076)}

Timing summary
Size: 64
    fast: 0.00440
    gpu: 0.01397
Size: 128
Timing summary
Size: 64
    fast: 0.00440
    gpu: 0.01397
Size: 128
    fast: 0.01772
Size: 64
    fast: 0.00440
    gpu: 0.01397
Size: 128
    fast: 0.01772
    gpu: 0.02382
Size: 256
    gpu: 0.01397
Size: 128
    fast: 0.01772
    gpu: 0.02382
Size: 256
    fast: 0.10421
    gpu: 0.02382
Size: 256
    fast: 0.10421
    gpu: 0.06493
    fast: 0.10421
    gpu: 0.06493
    gpu: 0.06493
Size: 512
    fast: 0.49295
    gpu: 0.21952
Size: 1024
    fast: 4.47405
    gpu: 0.82945
```

## Training Results

### XOR Dataset - Small Model (CPU)
Parameters:
* Hidden Layers: 100
* Learning Rate: 0.05
* Backend: CPU
* Dataset Size: 50 points

Training Progress:
```
Epoch    0 | Loss 5.97502 | Correct   34 | Time 19.7640s
Epoch   10 | Loss 7.04915 | Correct   43 | Time 0.0766s
Epoch   20 | Loss 2.67970 | Correct   45 | Time 0.0880s
Epoch   30 | Loss 3.28085 | Correct   41 | Time 0.1213s
Epoch   40 | Loss 4.45794 | Correct   45 | Time 0.1350s
Epoch   50 | Loss 3.99205 | Correct   43 | Time 0.0775s
Epoch   60 | Loss 5.67755 | Correct   43 | Time 0.0784s
Epoch   70 | Loss 3.00926 | Correct   46 | Time 0.0789s
Epoch   80 | Loss 3.35669 | Correct   46 | Time 0.0785s
Epoch   90 | Loss 1.92815 | Correct   43 | Time 0.0769s
Epoch  100 | Loss 1.40904 | Correct   46 | Time 0.0751s
Epoch  110 | Loss 4.36400 | Correct   47 | Time 0.0851s
Epoch  120 | Loss 1.36593 | Correct   46 | Time 0.0775s
Epoch  130 | Loss 2.29552 | Correct   46 | Time 0.0757s
Epoch  140 | Loss 5.14656 | Correct   44 | Time 0.0760s
Epoch  150 | Loss 1.60225 | Correct   46 | Time 0.0897s
Epoch  160 | Loss 1.24502 | Correct   46 | Time 0.0783s
Epoch  170 | Loss 2.46542 | Correct   46 | Time 0.0781s
Epoch  180 | Loss 2.58880 | Correct   46 | Time 0.1369s
Epoch  190 | Loss 2.06091 | Correct   46 | Time 0.1511s
Epoch  200 | Loss 2.08967 | Correct   47 | Time 0.0786s
Epoch  210 | Loss 2.30680 | Correct   47 | Time 0.0762s
Epoch  220 | Loss 1.00663 | Correct   46 | Time 0.0781s
Epoch  230 | Loss 2.64572 | Correct   48 | Time 0.0771s
Epoch  240 | Loss 0.83267 | Correct   46 | Time 0.0897s
Epoch  250 | Loss 1.23253 | Correct   46 | Time 0.0779s
Epoch  260 | Loss 1.75260 | Correct   48 | Time 0.0773s
Epoch  270 | Loss 0.88677 | Correct   50 | Time 0.0785s
Epoch  280 | Loss 1.35076 | Correct   48 | Time 0.0783s
Epoch  290 | Loss 1.27302 | Correct   48 | Time 0.0779s
Epoch  300 | Loss 1.12339 | Correct   49 | Time 0.0773s
Epoch  310 | Loss 1.54354 | Correct   49 | Time 0.0796s
Epoch  320 | Loss 1.13248 | Correct   50 | Time 0.1688s
Epoch  330 | Loss 0.77947 | Correct   47 | Time 0.1197s
Epoch  340 | Loss 1.79479 | Correct   50 | Time 0.0782s
Epoch  350 | Loss 0.91626 | Correct   50 | Time 0.0780s
Epoch  360 | Loss 1.94995 | Correct   50 | Time 0.0793s
Epoch  370 | Loss 0.15056 | Correct   50 | Time 0.0796s
Epoch  380 | Loss 0.49022 | Correct   50 | Time 0.0779s
Epoch  390 | Loss 0.74923 | Correct   50 | Time 0.0796s
Epoch  400 | Loss 0.43654 | Correct   49 | Time 0.0762s
Epoch  410 | Loss 0.38834 | Correct   50 | Time 0.0771s
Epoch  420 | Loss 0.25350 | Correct   50 | Time 0.0858s
Epoch  430 | Loss 0.43659 | Correct   50 | Time 0.0782s
Epoch  440 | Loss 0.35813 | Correct   50 | Time 0.0778s
Epoch  450 | Loss 0.22344 | Correct   50 | Time 0.0764s
Epoch  460 | Loss 0.80293 | Correct   50 | Time 0.1192s
Epoch  470 | Loss 0.12446 | Correct   50 | Time 0.1616s
Epoch  480 | Loss 0.21404 | Correct   50 | Time 0.0772s
Epoch  490 | Loss 0.45139 | Correct   50 | Time 0.0786s
Epoch  499 | Loss 0.08523 | Correct   50 | Time 0.0797s
```

Training Statistics:
* Total Training Time: 63.91s
* Average Epoch Time: 0.1278s
* Final Accuracy: 100.0% (50/50 correct)
* Final Loss: 0.08523

### XOR Dataset - Large Model (CPU)
Parameters:
* Hidden Layers: 200
* Learning Rate: 0.05
* Backend: CPU
* Dataset Size: 50 points

Training Progress:
```
Epoch    0 | Loss 7.35411 | Correct   22 | Time 15.9736s
Epoch   10 | Loss 6.36068 | Correct   45 | Time 0.1252s
Epoch   20 | Loss 3.12276 | Correct   45 | Time 0.1252s
Epoch   30 | Loss 2.39683 | Correct   44 | Time 0.1414s
Epoch   40 | Loss 2.30914 | Correct   46 | Time 0.1254s
Epoch   50 | Loss 2.16758 | Correct   47 | Time 0.1254s
Epoch   60 | Loss 0.67399 | Correct   47 | Time 0.1254s
Epoch   70 | Loss 2.08351 | Correct   48 | Time 0.1410s
Epoch   80 | Loss 1.69033 | Correct   48 | Time 0.1201s
Epoch   90 | Loss 0.56669 | Correct   49 | Time 0.1254s
Epoch  100 | Loss 1.13112 | Correct   49 | Time 0.1350s
Epoch  110 | Loss 1.68987 | Correct   47 | Time 0.1410s
Epoch  120 | Loss 1.52260 | Correct   50 | Time 0.1315s
Epoch  130 | Loss 1.35780 | Correct   47 | Time 0.1410s
Epoch  140 | Loss 0.41241 | Correct   48 | Time 0.1197s
Epoch  150 | Loss 1.02668 | Correct   48 | Time 0.1420s
Epoch  160 | Loss 2.62987 | Correct   49 | Time 0.1340s
Epoch  170 | Loss 1.26236 | Correct   48 | Time 0.1254s
Epoch  180 | Loss 1.98002 | Correct   46 | Time 0.1201s
Epoch  190 | Loss 0.30144 | Correct   49 | Time 0.1410s
Epoch  200 | Loss 0.07679 | Correct   48 | Time 0.1370s
Epoch  210 | Loss 0.22996 | Correct   48 | Time 0.1408s
Epoch  220 | Loss 0.39481 | Correct   50 | Time 0.1522s
Epoch  230 | Loss 1.65221 | Correct   50 | Time 0.1417s
Epoch  240 | Loss 2.00888 | Correct   48 | Time 0.1340s
Epoch  250 | Loss 0.17513 | Correct   48 | Time 0.1316s
Epoch  260 | Loss 0.60533 | Correct   50 | Time 0.1414s
Epoch  270 | Loss 1.12541 | Correct   49 | Time 0.1410s
Epoch  280 | Loss 0.99916 | Correct   49 | Time 0.1570s
Epoch  290 | Loss 0.19896 | Correct   49 | Time 0.1726s
Epoch  300 | Loss 0.35495 | Correct   48 | Time 0.2039s
Epoch  310 | Loss 0.08584 | Correct   48 | Time 0.1510s
Epoch  320 | Loss 0.24161 | Correct   49 | Time 0.1636s
Epoch  330 | Loss 0.45654 | Correct   50 | Time 0.1570s
Epoch  340 | Loss 0.68180 | Correct   48 | Time 0.1414s
Epoch  350 | Loss 1.24315 | Correct   50 | Time 0.1410s
Epoch  360 | Loss 1.59617 | Correct   48 | Time 0.1410s
Epoch  370 | Loss 1.43335 | Correct   49 | Time 0.1410s
Epoch  380 | Loss 1.82391 | Correct   47 | Time 0.1566s
Epoch  390 | Loss 1.33152 | Correct   49 | Time 0.1410s
Epoch  400 | Loss 1.10339 | Correct   48 | Time 0.1570s
Epoch  410 | Loss 0.56039 | Correct   49 | Time 0.1515s
Epoch  420 | Loss 0.20256 | Correct   49 | Time 0.1570s
Epoch  430 | Loss 0.33725 | Correct   50 | Time 0.1505s
Epoch  440 | Loss 0.14105 | Correct   50 | Time 0.1414s
Epoch  450 | Loss 1.18033 | Correct   49 | Time 0.1414s
Epoch  460 | Loss 0.07708 | Correct   50 | Time 0.1414s
Epoch  470 | Loss 0.85474 | Correct   49 | Time 0.1520s
Epoch  480 | Loss 1.21148 | Correct   49 | Time 0.1570s
Epoch  490 | Loss 0.17861 | Correct   49 | Time 0.1414s
Epoch  499 | Loss 0.60974 | Correct   49 | Time 0.1566s
```

Training Statistics:
* Total Training Time: 88.73s
* Average Epoch Time: 0.1775s
* Final Accuracy: 98.0% (49/50 correct)
* Final Loss: 0.60974

---

### Simple Dataset - Small Model (CPU)
Parameters:
* Hidden Layers: 100
* Learning Rate: 0.05
* Backend: CPU
* Dataset Size: 50 points

Training Progress:
```
Epoch    0 | Loss 4.45742 | Correct   42 | Time 20.6153s
Epoch   10 | Loss 1.53997 | Correct   50 | Time 0.0792s
Epoch   20 | Loss 0.88063 | Correct   50 | Time 0.0769s
Epoch   30 | Loss 0.57703 | Correct   50 | Time 0.0791s
Epoch   40 | Loss 0.55526 | Correct   50 | Time 0.0791s
Epoch   50 | Loss 0.06133 | Correct   50 | Time 0.0772s
Epoch   60 | Loss 0.06000 | Correct   50 | Time 0.0864s
Epoch   70 | Loss 0.74051 | Correct   50 | Time 0.1725s
Epoch   80 | Loss 0.38206 | Correct   50 | Time 0.0786s
Epoch   90 | Loss 0.56463 | Correct   50 | Time 0.0761s
Epoch  100 | Loss 0.06906 | Correct   50 | Time 0.0776s
Epoch  110 | Loss 0.32536 | Correct   50 | Time 0.0772s
Epoch  120 | Loss 0.25343 | Correct   50 | Time 0.0777s
Epoch  130 | Loss 0.13292 | Correct   50 | Time 0.0768s
Epoch  140 | Loss 0.28324 | Correct   50 | Time 0.0756s
Epoch  150 | Loss 0.07069 | Correct   50 | Time 0.0905s
Epoch  160 | Loss 0.02626 | Correct   50 | Time 0.0774s
Epoch  170 | Loss 0.35083 | Correct   50 | Time 0.0765s
Epoch  180 | Loss 0.00185 | Correct   50 | Time 0.0776s
Epoch  190 | Loss 0.16453 | Correct   50 | Time 0.0794s
Epoch  200 | Loss 0.08810 | Correct   50 | Time 0.1414s
Epoch  210 | Loss 0.06900 | Correct   50 | Time 0.1505s
Epoch  220 | Loss 0.08425 | Correct   50 | Time 0.0777s
Epoch  230 | Loss 0.24454 | Correct   50 | Time 0.0789s
Epoch  240 | Loss 0.16437 | Correct   50 | Time 0.0914s
Epoch  250 | Loss 0.20540 | Correct   50 | Time 0.0765s
Epoch  260 | Loss 0.09131 | Correct   50 | Time 0.0918s
Epoch  270 | Loss 0.06251 | Correct   50 | Time 0.0761s
Epoch  280 | Loss 0.12354 | Correct   50 | Time 0.0774s
Epoch  290 | Loss 0.11067 | Correct   50 | Time 0.0776s
Epoch  300 | Loss 0.07027 | Correct   50 | Time 0.0786s
Epoch  310 | Loss 0.22358 | Correct   50 | Time 0.0894s
Epoch  320 | Loss 0.06146 | Correct   50 | Time 0.1887s
Epoch  330 | Loss 0.27318 | Correct   50 | Time 0.0760s
Epoch  340 | Loss 0.40789 | Correct   50 | Time 0.0772s
Epoch  350 | Loss 0.67686 | Correct   50 | Time 0.0783s
Epoch  360 | Loss 0.42108 | Correct   50 | Time 0.0778s
Epoch  370 | Loss 0.56009 | Correct   50 | Time 0.0794s
Epoch  380 | Loss 0.01390 | Correct   50 | Time 0.0769s
Epoch  390 | Loss 0.39220 | Correct   50 | Time 0.0791s
Epoch  400 | Loss 0.01260 | Correct   50 | Time 0.0802s
Epoch  410 | Loss 0.29561 | Correct   50 | Time 0.0779s
Epoch  420 | Loss 0.57358 | Correct   50 | Time 0.0778s
Epoch  430 | Loss 0.52167 | Correct   50 | Time 0.0770s
Epoch  440 | Loss 0.56615 | Correct   50 | Time 0.0772s
Epoch  450 | Loss 0.58075 | Correct   50 | Time 0.1272s
Epoch  460 | Loss 0.00863 | Correct   50 | Time 0.1145s
Epoch  470 | Loss 0.60918 | Correct   50 | Time 0.0780s
Epoch  480 | Loss 0.50905 | Correct   50 | Time 0.0785s
Epoch  490 | Loss 0.51520 | Correct   50 | Time 0.0771s
Epoch  499 | Loss 0.29050 | Correct   50 | Time 0.0793s
```

Training Statistics:
* Total Training Time: 64.97s
* Average Epoch Time: 0.1299s
* Final Accuracy: 100.0% (50/50 correct)
* Final Loss: 0.01975

### Simple Dataset - Large Model (CPU)
Parameters:
* Hidden Layers: 200
* Learning Rate: 0.05
* Backend: CPU
* Dataset Size: 50 points

Training Progress:
```
Epoch    0 | Loss 4.72131 | Correct   35 | Time 17.2107s
Epoch   10 | Loss 1.81449 | Correct   48 | Time 0.1379s
Epoch   20 | Loss 1.24055 | Correct   46 | Time 0.1374s
Epoch   30 | Loss 0.81414 | Correct   47 | Time 0.1349s
Epoch   40 | Loss 0.13297 | Correct   50 | Time 0.1388s
Epoch   50 | Loss 0.73870 | Correct   50 | Time 0.1389s
Epoch   60 | Loss 0.23575 | Correct   50 | Time 0.1402s
Epoch   70 | Loss 0.42018 | Correct   50 | Time 0.1418s
Epoch   80 | Loss 0.24045 | Correct   49 | Time 0.1415s
Epoch   90 | Loss 1.02745 | Correct   49 | Time 0.1370s
Epoch  100 | Loss 0.06666 | Correct   49 | Time 0.1328s
Epoch  110 | Loss 0.23824 | Correct   50 | Time 0.1465s
Epoch  120 | Loss 0.16987 | Correct   50 | Time 0.1344s
Epoch  130 | Loss 0.24667 | Correct   50 | Time 0.1392s
Epoch  140 | Loss 0.25215 | Correct   50 | Time 0.1336s
Epoch  150 | Loss 0.60321 | Correct   50 | Time 0.1389s
Epoch  160 | Loss 0.70694 | Correct   50 | Time 0.1197s
Epoch  170 | Loss 0.07200 | Correct   50 | Time 0.1289s
Epoch  180 | Loss 0.26895 | Correct   50 | Time 0.1406s
Epoch  190 | Loss 0.44239 | Correct   50 | Time 0.1393s
Epoch  200 | Loss 0.78716 | Correct   50 | Time 0.1350s
Epoch  210 | Loss 0.22729 | Correct   50 | Time 0.1388s
Epoch  220 | Loss 0.55085 | Correct   50 | Time 0.1274s
Epoch  230 | Loss 0.18525 | Correct   50 | Time 0.1404s
Epoch  240 | Loss 0.32403 | Correct   50 | Time 0.1419s
Epoch  250 | Loss 0.14362 | Correct   50 | Time 0.1316s
Epoch  260 | Loss 0.60755 | Correct   50 | Time 0.1284s
Epoch  270 | Loss 0.39499 | Correct   50 | Time 0.1316s
Epoch  280 | Loss 0.84113 | Correct   50 | Time 0.1397s
Epoch  290 | Loss 0.36983 | Correct   50 | Time 0.1275s
Epoch  300 | Loss 0.06398 | Correct   50 | Time 0.1397s
Epoch  310 | Loss 0.01540 | Correct   50 | Time 0.1397s
Epoch  320 | Loss 0.91708 | Correct   49 | Time 0.1442s
Epoch  330 | Loss 0.04303 | Correct   50 | Time 0.1345s
Epoch  340 | Loss 0.25116 | Correct   50 | Time 0.1437s
Epoch  350 | Loss 0.13365 | Correct   50 | Time 0.1302s
Epoch  360 | Loss 0.60759 | Correct   50 | Time 0.1366s
Epoch  370 | Loss 0.03344 | Correct   50 | Time 0.1330s
Epoch  380 | Loss 0.01238 | Correct   50 | Time 0.1366s
Epoch  390 | Loss 0.19600 | Correct   50 | Time 0.1399s
Epoch  400 | Loss 0.22687 | Correct   50 | Time 0.1327s
Epoch  410 | Loss 0.42601 | Correct   50 | Time 0.1362s
Epoch  420 | Loss 0.25716 | Correct   50 | Time 0.1293s
Epoch  430 | Loss 0.43060 | Correct   50 | Time 0.1391s
Epoch  440 | Loss 0.36203 | Correct   50 | Time 0.1296s
Epoch  450 | Loss 0.17395 | Correct   50 | Time 0.1500s
Epoch  460 | Loss 0.01737 | Correct   50 | Time 0.1443s
Epoch  470 | Loss 0.17979 | Correct   50 | Time 0.1483s
Epoch  480 | Loss 0.15790 | Correct   50 | Time 0.1472s
Epoch  490 | Loss 0.33208 | Correct   50 | Time 0.1435s
Epoch  499 | Loss 0.05355 | Correct   50 | Time 0.1474s
```

Training Statistics:
* Total Training Time: 87.21s
* Average Epoch Time: 0.1744s
* Final Accuracy: 100.0% (50/50 correct)
* Final Loss: 0.05355

---

### Split Dataset - Small Model (CPU)
Parameters:
* Hidden Layers: 100
* Learning Rate: 0.05
* Backend: CPU
* Dataset Size: 50 points

Training Progress:
```
Epoch    0 | Loss 4.62409 | Correct   34 | Time 19.7474s
Epoch   10 | Loss 7.46764 | Correct   40 | Time 0.0780s
Epoch   20 | Loss 4.00682 | Correct   39 | Time 0.0980s
Epoch   30 | Loss 5.70571 | Correct   40 | Time 0.1834s
Epoch   40 | Loss 4.86282 | Correct   45 | Time 0.0774s
Epoch   50 | Loss 3.33000 | Correct   42 | Time 0.0763s
Epoch   60 | Loss 1.78046 | Correct   42 | Time 0.0778s
Epoch   70 | Loss 1.96509 | Correct   49 | Time 0.0760s
Epoch   80 | Loss 2.86069 | Correct   49 | Time 0.0751s
Epoch   90 | Loss 1.04152 | Correct   50 | Time 0.0786s
Epoch  100 | Loss 1.86045 | Correct   46 | Time 0.0762s
Epoch  110 | Loss 2.05591 | Correct   49 | Time 0.0759s
Epoch  120 | Loss 1.55844 | Correct   50 | Time 0.0758s
Epoch  130 | Loss 0.87196 | Correct   49 | Time 0.0763s
Epoch  140 | Loss 0.94036 | Correct   50 | Time 0.0769s
Epoch  150 | Loss 0.70849 | Correct   49 | Time 0.0778s
Epoch  160 | Loss 0.82492 | Correct   50 | Time 0.1226s
Epoch  170 | Loss 0.58113 | Correct   50 | Time 0.1091s
Epoch  180 | Loss 0.07695 | Correct   49 | Time 0.0772s
Epoch  190 | Loss 0.77315 | Correct   50 | Time 0.0812s
Epoch  200 | Loss 0.72960 | Correct   50 | Time 0.0791s
Epoch  210 | Loss 0.15449 | Correct   50 | Time 0.0773s
Epoch  220 | Loss 0.50639 | Correct   50 | Time 0.0780s
Epoch  230 | Loss 0.60734 | Correct   50 | Time 0.0785s
Epoch  240 | Loss 0.42044 | Correct   50 | Time 0.0779s
Epoch  250 | Loss 0.06462 | Correct   50 | Time 0.0788s
Epoch  260 | Loss 0.67198 | Correct   50 | Time 0.0884s
Epoch  270 | Loss 0.01186 | Correct   50 | Time 0.0781s
Epoch  280 | Loss 1.35441 | Correct   50 | Time 0.0840s
Epoch  290 | Loss 0.71836 | Correct   50 | Time 0.0773s
Epoch  300 | Loss 0.53341 | Correct   50 | Time 0.0781s
Epoch  310 | Loss 0.28494 | Correct   50 | Time 0.0894s
Epoch  320 | Loss 0.56080 | Correct   50 | Time 0.1887s
Epoch  330 | Loss 0.27318 | Correct   50 | Time 0.0760s
Epoch  340 | Loss 0.40789 | Correct   50 | Time 0.0772s
Epoch  350 | Loss 0.67686 | Correct   50 | Time 0.0783s
Epoch  360 | Loss 0.42108 | Correct   50 | Time 0.0778s
Epoch  370 | Loss 0.56009 | Correct   50 | Time 0.0794s
Epoch  380 | Loss 0.01390 | Correct   50 | Time 0.0769s
Epoch  390 | Loss 0.39220 | Correct   50 | Time 0.0791s
Epoch  400 | Loss 0.01260 | Correct   50 | Time 0.0802s
Epoch  410 | Loss 0.29561 | Correct   50 | Time 0.0779s
Epoch  420 | Loss 0.57358 | Correct   50 | Time 0.0778s
Epoch  430 | Loss 0.52167 | Correct   50 | Time 0.0770s
Epoch  440 | Loss 0.56615 | Correct   50 | Time 0.0772s
Epoch  450 | Loss 0.58075 | Correct   50 | Time 0.1272s
Epoch  460 | Loss 0.00863 | Correct   50 | Time 0.1145s
Epoch  470 | Loss 0.60918 | Correct   50 | Time 0.0780s
Epoch  480 | Loss 0.50905 | Correct   50 | Time 0.0785s
Epoch  490 | Loss 0.51520 | Correct   50 | Time 0.0771s
Epoch  499 | Loss 0.29050 | Correct   50 | Time 0.0793s
```

Training Statistics:
* Total Training Time: 63.68s
* Average Epoch Time: 0.1273s
* Final Accuracy: 100.0% (50/50 correct)
* Final Loss: 0.29050

### Split Dataset - Large Model (CPU)
Parameters:
* Hidden Layers: 200
* Learning Rate: 0.05
* Backend: CPU
* Dataset Size: 50 points

Training Progress:
```
Epoch    0 | Loss 10.65115 | Correct   30 | Time 15.8332s
Epoch   10 | Loss 4.57919 | Correct   38 | Time 0.1357s
Epoch   20 | Loss 4.80803 | Correct   45 | Time 0.1254s
Epoch   30 | Loss 1.13931 | Correct   49 | Time 0.1254s
Epoch   40 | Loss 1.58802 | Correct   47 | Time 0.1309s
Epoch   50 | Loss 0.85437 | Correct   50 | Time 0.1254s
Epoch   60 | Loss 1.41152 | Correct   49 | Time 0.1130s
Epoch   70 | Loss 2.18499 | Correct   49 | Time 0.1254s
Epoch   80 | Loss 1.39820 | Correct   50 | Time 0.1201s
Epoch   90 | Loss 1.46362 | Correct   49 | Time 0.1254s
Epoch  100 | Loss 0.59004 | Correct   49 | Time 0.1201s
Epoch  110 | Loss 0.82394 | Correct   50 | Time 0.1254s
Epoch  120 | Loss 0.55208 | Correct   49 | Time 0.1254s
Epoch  130 | Loss 1.52150 | Correct   49 | Time 0.1410s
Epoch  140 | Loss 1.75472 | Correct   49 | Time 0.1254s
Epoch  150 | Loss 0.40686 | Correct   50 | Time 0.1410s
Epoch  160 | Loss 1.01210 | Correct   50 | Time 0.1410s
Epoch  170 | Loss 1.12577 | Correct   50 | Time 0.1414s
Epoch  180 | Loss 0.48538 | Correct   50 | Time 0.1410s
Epoch  190 | Loss 0.44171 | Correct   50 | Time 0.1254s
Epoch  200 | Loss 1.02653 | Correct   49 | Time 0.1409s
Epoch  210 | Loss 0.02377 | Correct   50 | Time 0.1410s
Epoch  220 | Loss 1.22453 | Correct   49 | Time 0.1410s
Epoch  230 | Loss 0.84228 | Correct   49 | Time 0.1258s
Epoch  240 | Loss 0.76376 | Correct   49 | Time 0.1254s
Epoch  250 | Loss 0.48151 | Correct   50 | Time 0.1410s
Epoch  260 | Loss 0.63526 | Correct   50 | Time 0.1311s
Epoch  270 | Loss 1.01951 | Correct   49 | Time 0.1254s
Epoch  280 | Loss 0.10683 | Correct   49 | Time 0.1254s
Epoch  290 | Loss 0.20784 | Correct   49 | Time 0.1455s
Epoch  300 | Loss 0.21012 | Correct   50 | Time 0.1254s
Epoch  310 | Loss 0.82647 | Correct   49 | Time 0.1353s
Epoch  320 | Loss 1.01652 | Correct   49 | Time 0.1410s
Epoch  330 | Loss 0.31660 | Correct   50 | Time 0.1414s
Epoch  340 | Loss 0.74347 | Correct   50 | Time 0.1357s
Epoch  350 | Loss 0.26680 | Correct   50 | Time 0.1410s
Epoch  360 | Loss 0.85119 | Correct   49 | Time 0.1410s
Epoch  370 | Loss 0.03983 | Correct   49 | Time 0.1417s
Epoch  380 | Loss 0.04241 | Correct   49 | Time 0.1410s
Epoch  390 | Loss 0.25191 | Correct   50 | Time 0.1414s
Epoch  400 | Loss 0.15942 | Correct   49 | Time 0.1513s
Epoch  410 | Loss 0.51777 | Correct   50 | Time 0.1259s
Epoch  420 | Loss 0.09673 | Correct   50 | Time 0.1409s
Epoch  430 | Loss 0.14567 | Correct   49 | Time 0.1361s
Epoch  440 | Loss 0.52826 | Correct   50 | Time 0.1414s
Epoch  450 | Loss 0.57619 | Correct   50 | Time 0.1410s
Epoch  460 | Loss 0.54476 | Correct   50 | Time 0.1509s
Epoch  470 | Loss 0.00389 | Correct   50 | Time 0.1410s
Epoch  480 | Loss 0.05340 | Correct   50 | Time 0.1414s
Epoch  490 | Loss 0.38370 | Correct   50 | Time 0.1339s
Epoch  499 | Loss 0.06553 | Correct   50 | Time 0.1414s
```

Training Statistics:
* Total Training Time: 83.67s
* Average Epoch Time: 0.1673s
* Final Accuracy: 100.0% (50/50 correct)
* Final Loss: 0.06553

---

### XOR Dataset - Small Model (GPU)
Parameters:
* Hidden Layers: 100
* Learning Rate: 0.05
* Backend: GPU
* Dataset Size: 50 points

Training Progress:
```
Epoch    0 | Loss 6.10812 | Correct   31 | Time 4.4940s
Epoch   10 | Loss 4.78282 | Correct   43 | Time 1.1462s
Epoch   20 | Loss 4.19695 | Correct   44 | Time 1.1820s
Epoch   30 | Loss 3.33396 | Correct   45 | Time 1.1681s
Epoch   40 | Loss 2.21444 | Correct   45 | Time 1.1528s
Epoch   50 | Loss 3.46232 | Correct   45 | Time 1.1490s
Epoch   60 | Loss 1.18688 | Correct   44 | Time 1.1896s
Epoch   70 | Loss 2.84870 | Correct   45 | Time 1.2144s
Epoch   80 | Loss 1.72358 | Correct   46 | Time 1.1624s
Epoch   90 | Loss 2.56431 | Correct   48 | Time 1.1643s
Epoch  100 | Loss 3.22844 | Correct   47 | Time 1.1698s
Epoch  110 | Loss 1.59254 | Correct   47 | Time 1.1550s
Epoch  120 | Loss 1.23674 | Correct   49 | Time 1.1555s
Epoch  130 | Loss 2.63734 | Correct   48 | Time 1.1580s
Epoch  140 | Loss 1.34030 | Correct   48 | Time 1.2281s
Epoch  150 | Loss 0.28163 | Correct   49 | Time 1.1549s
Epoch  160 | Loss 1.93069 | Correct   49 | Time 1.1413s
Epoch  170 | Loss 1.97253 | Correct   49 | Time 1.1473s
Epoch  180 | Loss 1.11590 | Correct   47 | Time 1.1611s
Epoch  190 | Loss 0.94126 | Correct   48 | Time 1.1908s
Epoch  200 | Loss 1.56340 | Correct   48 | Time 1.1434s
Epoch  210 | Loss 0.36101 | Correct   50 | Time 1.2284s
Epoch  220 | Loss 0.15163 | Correct   50 | Time 1.1494s
Epoch  230 | Loss 0.56328 | Correct   50 | Time 1.1495s
Epoch  240 | Loss 1.18985 | Correct   49 | Time 1.1480s
Epoch  250 | Loss 0.50150 | Correct   50 | Time 1.1481s
Epoch  260 | Loss 0.49930 | Correct   49 | Time 1.1582s
Epoch  270 | Loss 0.87420 | Correct   50 | Time 1.1571s
Epoch  280 | Loss 0.31756 | Correct   50 | Time 1.2049s
Epoch  290 | Loss 1.09817 | Correct   50 | Time 1.1536s
Epoch  300 | Loss 0.06957 | Correct   50 | Time 1.2442s
Epoch  310 | Loss 0.67900 | Correct   50 | Time 1.3256s
Epoch  320 | Loss 1.71390 | Correct   49 | Time 1.3834s
Epoch  330 | Loss 0.26682 | Correct   50 | Time 1.4480s
Epoch  340 | Loss 0.26268 | Correct   50 | Time 1.4986s
Epoch  350 | Loss 0.68632 | Correct   50 | Time 1.5336s
Epoch  360 | Loss 0.60919 | Correct   50 | Time 1.5313s
Epoch  370 | Loss 0.17119 | Correct   50 | Time 1.5263s
Epoch  380 | Loss 0.88038 | Correct   50 | Time 1.5738s
Epoch  390 | Loss 0.90701 | Correct   49 | Time 1.7067s
Epoch  400 | Loss 0.78891 | Correct   49 | Time 1.7098s
Epoch  410 | Loss 0.32119 | Correct   49 | Time 1.7101s
Epoch  420 | Loss 0.44307 | Correct   49 | Time 1.8173s
Epoch  430 | Loss 0.98327 | Correct   49 | Time 1.7224s
Epoch  440 | Loss 0.02847 | Correct   49 | Time 1.7159s
Epoch  450 | Loss 0.83818 | Correct   50 | Time 1.6987s
Epoch  460 | Loss 0.19674 | Correct   49 | Time 1.6343s
Epoch  470 | Loss 0.61539 | Correct   49 | Time 1.5857s
Epoch  480 | Loss 0.61510 | Correct   50 | Time 1.5304s
Epoch  490 | Loss 0.29364 | Correct   50 | Time 1.5591s
Epoch  499 | Loss 1.02490 | Correct   49 | Time 1.6909s
```

Training Statistics:
* Total Training Time: 629.44s
* Average Epoch Time: 1.2589s
* Final Accuracy: 98.0% (49/50 correct)
* Final Loss: 1.02490

### XOR Dataset - Large Model (GPU)
Parameters:
* Hidden Layers: 200
* Learning Rate: 0.05
* Backend: GPU
* Dataset Size: 50 points

Training Progress:
```
Epoch    0 | Loss 9.80118 | Correct   26 | Time 4.2749s
Epoch   10 | Loss 3.54844 | Correct   45 | Time 1.4280s
Epoch   20 | Loss 3.28800 | Correct   47 | Time 1.4932s
Epoch   30 | Loss 1.47814 | Correct   50 | Time 1.4350s
Epoch   40 | Loss 2.03668 | Correct   50 | Time 1.5100s
Epoch   50 | Loss 1.20216 | Correct   50 | Time 1.5448s
Epoch   60 | Loss 0.26705 | Correct   48 | Time 1.5148s
Epoch   70 | Loss 1.42270 | Correct   48 | Time 1.5107s
Epoch   80 | Loss 1.33907 | Correct   50 | Time 1.5476s
Epoch   90 | Loss 1.47358 | Correct   50 | Time 1.5250s
Epoch  100 | Loss 0.88841 | Correct   49 | Time 1.4995s
Epoch  110 | Loss 1.56250 | Correct   50 | Time 1.4674s
Epoch  120 | Loss 0.94244 | Correct   50 | Time 1.5135s
Epoch  130 | Loss 2.24722 | Correct   46 | Time 1.5143s
Epoch  140 | Loss 0.23606 | Correct   50 | Time 1.5958s
Epoch  150 | Loss 0.67038 | Correct   50 | Time 1.5635s
Epoch  160 | Loss 1.10976 | Correct   50 | Time 1.5522s
Epoch  170 | Loss 2.11180 | Correct   50 | Time 1.4415s
Epoch  180 | Loss 0.79962 | Correct   50 | Time 1.4401s
Epoch  190 | Loss 0.63255 | Correct   50 | Time 1.4476s
Epoch  200 | Loss 0.09700 | Correct   50 | Time 1.4301s
Epoch  210 | Loss 0.97863 | Correct   50 | Time 1.3987s
Epoch  220 | Loss 0.57276 | Correct   50 | Time 1.3934s
Epoch  230 | Loss 0.76667 | Correct   50 | Time 1.4425s
Epoch  240 | Loss 0.56983 | Correct   50 | Time 1.4418s
Epoch  250 | Loss 0.90663 | Correct   50 | Time 1.4264s
Epoch  260 | Loss 0.54150 | Correct   50 | Time 1.4936s
Epoch  270 | Loss 0.89642 | Correct   50 | Time 1.4479s
Epoch  280 | Loss 0.53462 | Correct   50 | Time 1.4103s
Epoch  290 | Loss 0.14341 | Correct   50 | Time 1.4835s
Epoch  300 | Loss 0.04127 | Correct   50 | Time 1.4737s
Epoch  310 | Loss 0.83304 | Correct   50 | Time 1.4318s
Epoch  320 | Loss 0.21716 | Correct   50 | Time 1.4245s
Epoch  330 | Loss 0.08941 | Correct   50 | Time 1.4560s
Epoch  340 | Loss 0.69952 | Correct   50 | Time 1.6775s
Epoch  350 | Loss 0.49359 | Correct   50 | Time 1.5325s
Epoch  360 | Loss 0.39930 | Correct   50 | Time 1.5148s
Epoch  370 | Loss 0.06705 | Correct   50 | Time 1.5990s
Epoch  380 | Loss 0.65905 | Correct   50 | Time 1.5837s
Epoch  390 | Loss 0.88022 | Correct   50 | Time 1.5622s
Epoch  400 | Loss 0.39326 | Correct   50 | Time 1.6647s
Epoch  410 | Loss 0.43799 | Correct   50 | Time 1.5187s
Epoch  420 | Loss 0.14457 | Correct   50 | Time 1.4733s
Epoch  430 | Loss 0.66708 | Correct   50 | Time 1.4555s
Epoch  440 | Loss 0.37095 | Correct   50 | Time 1.5172s
Epoch  450 | Loss 0.08203 | Correct   50 | Time 1.4317s
Epoch  460 | Loss 0.30288 | Correct   50 | Time 1.4408s
Epoch  470 | Loss 0.13297 | Correct   50 | Time 1.4708s
Epoch  480 | Loss 0.39856 | Correct   50 | Time 1.4625s
Epoch  490 | Loss 0.08320 | Correct   50 | Time 1.4715s
Epoch  499 | Loss 0.13840 | Correct   50 | Time 1.4718s
```

Training Statistics:
* Total Training Time: 749.34s
* Average Epoch Time: 1.4986s
* Final Accuracy: 100.0% (50/50 correct)
* Final Loss: 0.13840

---

### Simple Dataset - Small Model (GPU)
Parameters:
* Hidden Layers: 100
* Learning Rate: 0.05
* Backend: GPU
* Dataset Size: 50 points

Training Progress:
```
Epoch    0 | Loss 5.73036 | Correct   38 | Time 5.4117s
Epoch   10 | Loss 7.46764 | Correct   40 | Time 0.0780s
Epoch   20 | Loss 4.00682 | Correct   39 | Time 0.0980s
Epoch   30 | Loss 5.70571 | Correct   40 | Time 0.1834s
Epoch   40 | Loss 4.86282 | Correct   45 | Time 0.0774s
Epoch   50 | Loss 3.33000 | Correct   42 | Time 0.0763s
Epoch   60 | Loss 1.78046 | Correct   42 | Time 0.0778s
Epoch   70 | Loss 1.96509 | Correct   49 | Time 0.0760s
Epoch   80 | Loss 2.86069 | Correct   49 | Time 0.0751s
Epoch   90 | Loss 1.04152 | Correct   50 | Time 0.0786s
Epoch  100 | Loss 1.86045 | Correct   46 | Time 0.0762s
Epoch  110 | Loss 2.05591 | Correct   49 | Time 0.0759s
Epoch  120 | Loss 1.55844 | Correct   50 | Time 0.0758s
Epoch  130 | Loss 0.87196 | Correct   49 | Time 0.0763s
Epoch  140 | Loss 0.94036 | Correct   50 | Time 0.0769s
Epoch  150 | Loss 0.70849 | Correct   49 | Time 0.0778s
Epoch  160 | Loss 0.82492 | Correct   50 | Time 0.1226s
Epoch  170 | Loss 0.58113 | Correct   50 | Time 0.1091s
Epoch  180 | Loss 0.07695 | Correct   49 | Time 0.0772s
Epoch  190 | Loss 0.77315 | Correct   50 | Time 0.0812s
Epoch  200 | Loss 0.72960 | Correct   50 | Time 0.0791s
Epoch  210 | Loss 0.15449 | Correct   50 | Time 0.0773s
Epoch  220 | Loss 0.50639 | Correct   50 | Time 0.0780s
Epoch  230 | Loss 0.60734 | Correct   50 | Time 0.0785s
Epoch  240 | Loss 0.42044 | Correct   50 | Time 0.0779s
Epoch  250 | Loss 0.06462 | Correct   50 | Time 0.0788s
Epoch  260 | Loss 0.67198 | Correct   50 | Time 0.0884s
Epoch  270 | Loss 0.01186 | Correct   50 | Time 0.0781s
Epoch  280 | Loss 1.35441 | Correct   50 | Time 0.0840s
Epoch  290 | Loss 0.71836 | Correct   50 | Time 0.0773s
Epoch  300 | Loss 0.53341 | Correct   50 | Time 0.0781s
Epoch  310 | Loss 0.28494 | Correct   50 | Time 0.0894s
Epoch  320 | Loss 0.56080 | Correct   50 | Time 0.1887s
Epoch  330 | Loss 0.27318 | Correct   50 | Time 0.0760s
Epoch  340 | Loss 0.40789 | Correct   50 | Time 0.0772s
Epoch  350 | Loss 0.67686 | Correct   50 | Time 0.0783s
Epoch  360 | Loss 0.42108 | Correct   50 | Time 0.0778s
Epoch  370 | Loss 0.56009 | Correct   50 | Time 0.0794s
Epoch  380 | Loss 0.01390 | Correct   50 | Time 0.0769s
Epoch  390 | Loss 0.39220 | Correct   50 | Time 0.0791s
Epoch  400 | Loss 0.01260 | Correct   50 | Time 0.0802s
Epoch  410 | Loss 0.29561 | Correct   50 | Time 0.0779s
Epoch  420 | Loss 0.57358 | Correct   50 | Time 0.0778s
Epoch  430 | Loss 0.52167 | Correct   50 | Time 0.0770s
Epoch  440 | Loss 0.56615 | Correct   50 | Time 0.0772s
Epoch  450 | Loss 0.58075 | Correct   50 | Time 0.1272s
Epoch  460 | Loss 0.00863 | Correct   50 | Time 0.1145s
Epoch  470 | Loss 0.60918 | Correct   50 | Time 0.0780s
Epoch  480 | Loss 0.50905 | Correct   50 | Time 0.0785s
Epoch  490 | Loss 0.51520 | Correct   50 | Time 0.0771s
Epoch  499 | Loss 0.29050 | Correct   50 | Time 0.0793s
```

Training Statistics:
* Total Training Time: 637.32s
* Average Epoch Time: 1.2746s
* Final Accuracy: 100.0% (50/50 correct)
* Final Loss: 0.03225

### Simple Dataset - Large Model (GPU)
Parameters:
* Hidden Layers: 200
* Learning Rate: 0.05
* Backend: GPU
* Dataset Size: 50 points

Training Progress:
```
Epoch    0 | Loss 4.06979 | Correct   37 | Time 3.9478s
Epoch   10 | Loss 2.27346 | Correct   46 | Time 1.5100s
Epoch   20 | Loss 1.51909 | Correct   50 | Time 1.5230s
Epoch   30 | Loss 1.05299 | Correct   50 | Time 1.4143s
Epoch   40 | Loss 0.46480 | Correct   50 | Time 1.4000s
Epoch   50 | Loss 0.63809 | Correct   50 | Time 1.4424s
Epoch   60 | Loss 0.59637 | Correct   50 | Time 1.3955s
Epoch   70 | Loss 0.48705 | Correct   50 | Time 1.4264s
Epoch   80 | Loss 2.70457 | Correct   43 | Time 1.4420s
Epoch   90 | Loss 0.24843 | Correct   50 | Time 1.5344s
Epoch  100 | Loss 0.67585 | Correct   50 | Time 1.4229s
Epoch  110 | Loss 0.37031 | Correct   50 | Time 1.4562s
Epoch  120 | Loss 0.42785 | Correct   50 | Time 1.4623s
Epoch  130 | Loss 0.09608 | Correct   50 | Time 1.4719s
Epoch  140 | Loss 0.33079 | Correct   50 | Time 1.4679s
Epoch  150 | Loss 0.33817 | Correct   50 | Time 1.4297s
Epoch  160 | Loss 0.46576 | Correct   50 | Time 1.4772s
Epoch  170 | Loss 0.21241 | Correct   50 | Time 1.4238s
Epoch  180 | Loss 0.37398 | Correct   50 | Time 1.4412s
Epoch  190 | Loss 0.01268 | Correct   50 | Time 1.4876s
Epoch  200 | Loss 0.40780 | Correct   50 | Time 1.4820s
Epoch  210 | Loss 0.03919 | Correct   50 | Time 1.4364s
Epoch  220 | Loss 0.16351 | Correct   50 | Time 1.4420s
Epoch  230 | Loss 0.12939 | Correct   50 | Time 1.4701s
Epoch  240 | Loss 0.12997 | Correct   50 | Time 1.4629s
Epoch  250 | Loss 0.05388 | Correct   50 | Time 1.4420s
Epoch  260 | Loss 0.13055 | Correct   50 | Time 1.4875s
Epoch  270 | Loss 0.01926 | Correct   50 | Time 1.4772s
Epoch  280 | Loss 0.04037 | Correct   50 | Time 1.4420s
Epoch  290 | Loss 0.28513 | Correct   50 | Time 1.5032s
Epoch  300 | Loss 0.18201 | Correct   50 | Time 1.4718s
Epoch  310 | Loss 0.06989 | Correct   50 | Time 1.4627s
Epoch  320 | Loss 0.07214 | Correct   50 | Time 1.4473s
Epoch  330 | Loss 0.01230 | Correct   50 | Time 1.5049s
Epoch  340 | Loss 0.03242 | Correct   50 | Time 1.4752s
Epoch  350 | Loss 0.15199 | Correct   50 | Time 1.4468s
Epoch  360 | Loss 0.06327 | Correct   50 | Time 1.5048s
Epoch  370 | Loss 0.13233 | Correct   50 | Time 1.4886s
Epoch  380 | Loss 0.12891 | Correct   50 | Time 1.4562s
Epoch  390 | Loss 0.10641 | Correct   50 | Time 1.4562s
Epoch  400 | Loss 0.06814 | Correct   50 | Time 1.4869s
Epoch  410 | Loss 0.06299 | Correct   50 | Time 1.5052s
Epoch  420 | Loss 0.09696 | Correct   50 | Time 1.4735s
Epoch  430 | Loss 0.14353 | Correct   50 | Time 1.4624s
Epoch  440 | Loss 0.11075 | Correct   50 | Time 1.4890s
Epoch  450 | Loss 0.08585 | Correct   50 | Time 1.4419s
Epoch  460 | Loss 0.08268 | Correct   50 | Time 1.4568s
Epoch  470 | Loss 0.03665 | Correct   50 | Time 1.4940s
Epoch  480 | Loss 0.09378 | Correct   50 | Time 1.4888s
Epoch  490 | Loss 0.04007 | Correct   50 | Time 1.4580s
Epoch  499 | Loss 0.01324 | Correct   50 | Time 1.4580s
```

Training Statistics:
* Total Training Time: 735.54s
* Average Epoch Time: 1.4711s
* Final Accuracy: 100.0% (50/50 correct)
* Final Loss: 0.01324

---

### Split Dataset - Small Model (GPU)
Parameters:
* Hidden Layers: 100
* Learning Rate: 0.05
* Backend: GPU
* Dataset Size: 50 points

Training Progress:
```
Epoch    0 | Loss 9.75676 | Correct   18 | Time 4.3865s
Epoch   10 | Loss 3.79190 | Correct   37 | Time 1.1591s
Epoch   20 | Loss 4.00464 | Correct   42 | Time 1.1366s
Epoch   30 | Loss 2.52475 | Correct   44 | Time 1.1283s
Epoch   40 | Loss 3.95984 | Correct   45 | Time 1.1493s
Epoch   50 | Loss 1.67677 | Correct   45 | Time 1.1428s
Epoch   60 | Loss 1.69466 | Correct   46 | Time 1.1382s
Epoch   70 | Loss 3.37929 | Correct   44 | Time 1.2228s
Epoch   80 | Loss 1.49102 | Correct   47 | Time 1.1394s
Epoch   90 | Loss 2.04178 | Correct   45 | Time 1.1284s
Epoch  100 | Loss 3.55293 | Correct   48 | Time 1.1377s
Epoch  110 | Loss 1.93283 | Correct   49 | Time 1.1379s
Epoch  120 | Loss 1.59389 | Correct   49 | Time 1.1616s
Epoch  130 | Loss 1.54533 | Correct   48 | Time 1.1770s
Epoch  140 | Loss 0.46630 | Correct   48 | Time 1.1549s
Epoch  150 | Loss 0.99609 | Correct   49 | Time 1.1593s
Epoch  160 | Loss 1.76474 | Correct   49 | Time 1.2044s
Epoch  170 | Loss 0.51757 | Correct   47 | Time 1.1400s
Epoch  180 | Loss 1.82454 | Correct   49 | Time 1.2207s
Epoch  190 | Loss 0.70064 | Correct   49 | Time 1.1563s
Epoch  200 | Loss 0.70578 | Correct   49 | Time 1.1854s
Epoch  210 | Loss 0.63537 | Correct   50 | Time 1.1671s
Epoch  220 | Loss 1.45896 | Correct   49 | Time 1.1287s
Epoch  230 | Loss 0.86262 | Correct   49 | Time 1.1280s
Epoch  240 | Loss 0.99798 | Correct   50 | Time 1.1322s
Epoch  250 | Loss 1.67712 | Correct   50 | Time 1.1932s
Epoch  260 | Loss 1.09545 | Correct   48 | Time 1.1336s
Epoch  270 | Loss 1.54779 | Correct   48 | Time 1.1984s
Epoch  280 | Loss 0.30301 | Correct   48 | Time 1.1249s
Epoch  290 | Loss 0.48493 | Correct   50 | Time 1.1437s
Epoch  300 | Loss 0.12676 | Correct   48 | Time 1.1300s
Epoch  310 | Loss 0.73989 | Correct   50 | Time 1.1559s
Epoch  320 | Loss 0.10944 | Correct   50 | Time 1.1645s
Epoch  330 | Loss 0.07068 | Correct   49 | Time 1.1284s
Epoch  340 | Loss 2.70179 | Correct   49 | Time 1.1321s
Epoch  350 | Loss 0.77562 | Correct   49 | Time 1.1677s
Epoch  360 | Loss 0.92516 | Correct   50 | Time 1.2226s
Epoch  370 | Loss 0.19960 | Correct   48 | Time 1.1698s
Epoch  380 | Loss 0.57462 | Correct   50 | Time 1.1803s
Epoch  390 | Loss 0.84407 | Correct   50 | Time 1.1429s
Epoch  400 | Loss 1.03353 | Correct   49 | Time 1.1773s
Epoch  410 | Loss 0.06579 | Correct   49 | Time 1.1693s
Epoch  420 | Loss 0.17955 | Correct   48 | Time 1.1703s
Epoch  430 | Loss 0.05393 | Correct   48 | Time 1.1332s
Epoch  440 | Loss 0.01757 | Correct   50 | Time 1.1345s
Epoch  450 | Loss 1.02905 | Correct   50 | Time 1.1830s
Epoch  460 | Loss 1.83248 | Correct   49 | Time 1.1508s
Epoch  470 | Loss 0.02148 | Correct   50 | Time 1.1416s
Epoch  480 | Loss 0.47079 | Correct   50 | Time 1.1363s
Epoch  490 | Loss 0.66070 | Correct   50 | Time 1.1264s
Epoch  499 | Loss 1.17932 | Correct   50 | Time 1.1261s
```

Training Statistics:
* Total Training Time: 622.66s
* Average Epoch Time: 1.2453s
* Final Accuracy: 100.0% (50/50 correct)
* Final Loss: 1.17932

### Split Dataset - Large Model (GPU)
Parameters:
* Hidden Layers: 200
* Learning Rate: 0.05
* Backend: GPU
* Dataset Size: 50 points

Training Progress:
```
Epoch    0 | Loss 3.77236 | Correct   36 | Time 4.0782s
Epoch   10 | Loss 3.40589 | Correct   46 | Time 1.4972s
Epoch   20 | Loss 3.36476 | Correct   40 | Time 1.5102s
Epoch   30 | Loss 2.76617 | Correct   46 | Time 1.6631s
Epoch   40 | Loss 1.18821 | Correct   50 | Time 1.5601s
Epoch   50 | Loss 1.24138 | Correct   50 | Time 1.5826s
Epoch   60 | Loss 1.36455 | Correct   48 | Time 1.5035s
Epoch   70 | Loss 1.36011 | Correct   48 | Time 1.5464s
Epoch   80 | Loss 1.00579 | Correct   50 | Time 1.5328s
Epoch   90 | Loss 0.12830 | Correct   48 | Time 1.5800s
Epoch  100 | Loss 1.41277 | Correct   50 | Time 1.5339s
Epoch  110 | Loss 1.63617 | Correct   50 | Time 1.5441s
Epoch  120 | Loss 0.44823 | Correct   50 | Time 1.5948s
Epoch  130 | Loss 0.47143 | Correct   50 | Time 1.5726s
Epoch  140 | Loss 1.29353 | Correct   49 | Time 1.5499s
Epoch  150 | Loss 1.61560 | Correct   50 | Time 1.5434s
Epoch  160 | Loss 0.50076 | Correct   50 | Time 1.5902s
Epoch  170 | Loss 0.66482 | Correct   50 | Time 1.5300s
Epoch  180 | Loss 0.21309 | Correct   50 | Time 1.5551s
Epoch  190 | Loss 0.25131 | Correct   50 | Time 1.6081s
Epoch  200 | Loss 0.08780 | Correct   50 | Time 1.5900s
Epoch  210 | Loss 0.65833 | Correct   50 | Time 1.5400s
Epoch  220 | Loss 0.54805 | Correct   50 | Time 1.5349s
Epoch  230 | Loss 0.22751 | Correct   50 | Time 1.5736s
Epoch  240 | Loss 0.12081 | Correct   50 | Time 1.5402s
Epoch  250 | Loss 0.36226 | Correct   50 | Time 1.5601s
Epoch  260 | Loss 0.24569 | Correct   50 | Time 1.6047s
Epoch  270 | Loss 0.25639 | Correct   50 | Time 1.6060s
Epoch  280 | Loss 0.47111 | Correct   50 | Time 1.5504s
Epoch  290 | Loss 0.20175 | Correct   50 | Time 1.6259s
Epoch  300 | Loss 0.23989 | Correct   50 | Time 1.5879s
Epoch  310 | Loss 0.06519 | Correct   50 | Time 1.5661s
Epoch  320 | Loss 0.10900 | Correct   50 | Time 1.5656s
Epoch  330 | Loss 0.30311 | Correct   50 | Time 1.6201s
Epoch  340 | Loss 0.11841 | Correct   50 | Time 1.6220s
Epoch  350 | Loss 0.19159 | Correct   50 | Time 1.5470s
Epoch  360 | Loss 0.10635 | Correct   50 | Time 1.5650s
Epoch  370 | Loss 0.03769 | Correct   50 | Time 1.5798s
Epoch  380 | Loss 0.32805 | Correct   50 | Time 1.5837s
Epoch  390 | Loss 0.12366 | Correct   50 | Time 1.5701s
Epoch  400 | Loss 0.10145 | Correct   50 | Time 1.6721s
Epoch  410 | Loss 0.04365 | Correct   50 | Time 1.5900s
Epoch  420 | Loss 0.06210 | Correct   50 | Time 1.6202s
Epoch  430 | Loss 0.12679 | Correct   50 | Time 1.5600s
Epoch  440 | Loss 0.13806 | Correct   50 | Time 1.6838s
Epoch  450 | Loss 0.02661 | Correct   50 | Time 1.5499s
Epoch  460 | Loss 0.10745 | Correct   50 | Time 1.5735s
Epoch  470 | Loss 0.12813 | Correct   50 | Time 1.6101s
Epoch  480 | Loss 0.03737 | Correct   50 | Time 1.6022s
Epoch  490 | Loss 0.07327 | Correct   50 | Time 1.5762s
Epoch  499 | Loss 0.11434 | Correct   50 | Time 1.6235s
```

Training Statistics:
* Total Training Time: 795.43s
* Average Epoch Time: 1.5909s
* Final Accuracy: 100.0% (50/50 correct)
* Final Loss: 0.11434

---