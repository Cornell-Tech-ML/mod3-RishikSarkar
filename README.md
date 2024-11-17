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

---

## Training Results

### XOR Dataset - Small Model
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

---

### Simple Dataset - Small Model
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

---

### Split Dataset - Small Model
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

---