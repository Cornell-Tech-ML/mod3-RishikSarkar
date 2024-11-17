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

## Training Results

### Split Dataset - Small Model
Parameters:
* Hidden Layers: 100
* Learning Rate: 0.05
* Backend: CPU
* Dataset Size: 50 points

Training Progress:
```
Epoch    0 | Loss 6.41258 | Correct   32 | Time 26.4551s
Epoch   10 | Loss 7.62842 | Correct   22 | Time 0.0941s
Epoch   20 | Loss 6.82458 | Correct   22 | Time 0.0941s
Epoch   30 | Loss 4.33809 | Correct   42 | Time 0.0930s
Epoch   40 | Loss 5.09160 | Correct   46 | Time 0.0785s
Epoch   50 | Loss 3.10454 | Correct   41 | Time 0.0937s
Epoch   60 | Loss 2.63838 | Correct   44 | Time 0.0941s
Epoch   70 | Loss 3.58128 | Correct   47 | Time 0.0941s
Epoch   80 | Loss 3.06032 | Correct   49 | Time 0.1852s
Epoch   90 | Loss 4.25588 | Correct   48 | Time 0.1055s
Epoch  100 | Loss 3.29212 | Correct   38 | Time 0.0941s
Epoch  110 | Loss 2.16834 | Correct   49 | Time 0.0876s
Epoch  120 | Loss 1.57968 | Correct   49 | Time 0.0941s
Epoch  130 | Loss 2.19076 | Correct   45 | Time 0.1049s
Epoch  140 | Loss 2.43811 | Correct   49 | Time 0.0941s
Epoch  150 | Loss 1.71128 | Correct   47 | Time 0.0785s
Epoch  160 | Loss 1.21172 | Correct   50 | Time 0.0941s
Epoch  170 | Loss 0.37472 | Correct   49 | Time 0.1001s
Epoch  180 | Loss 0.46065 | Correct   50 | Time 0.0884s
Epoch  190 | Loss 1.47038 | Correct   49 | Time 0.0941s
Epoch  200 | Loss 1.77882 | Correct   49 | Time 0.0941s
Epoch  210 | Loss 0.43337 | Correct   49 | Time 0.0941s
Epoch  220 | Loss 0.41826 | Correct   49 | Time 0.0885s
Epoch  230 | Loss 0.46321 | Correct   49 | Time 0.1097s
Epoch  240 | Loss 0.49498 | Correct   49 | Time 0.0941s
Epoch  250 | Loss 1.64883 | Correct   49 | Time 0.0947s
Epoch  260 | Loss 0.79516 | Correct   48 | Time 0.0892s
Epoch  270 | Loss 0.18389 | Correct   49 | Time 0.0941s
Epoch  280 | Loss 0.07255 | Correct   49 | Time 0.0941s
Epoch  290 | Loss 1.40513 | Correct   47 | Time 0.0941s
Epoch  300 | Loss 0.38093 | Correct   49 | Time 0.0871s
Epoch  310 | Loss 3.26727 | Correct   47 | Time 0.1001s
Epoch  320 | Loss 0.76144 | Correct   49 | Time 0.0941s
Epoch  330 | Loss 1.35627 | Correct   47 | Time 0.0946s
Epoch  340 | Loss 0.21264 | Correct   49 | Time 0.0882s
Epoch  350 | Loss 0.55407 | Correct   49 | Time 0.0992s
Epoch  360 | Loss 0.74813 | Correct   49 | Time 0.0941s
Epoch  370 | Loss 1.96867 | Correct   48 | Time 0.0941s
Epoch  380 | Loss 0.63003 | Correct   49 | Time 0.0935s
Epoch  390 | Loss 0.68243 | Correct   49 | Time 0.0941s
Epoch  400 | Loss 0.31680 | Correct   49 | Time 0.0941s
Epoch  410 | Loss 1.09192 | Correct   49 | Time 0.0941s
Epoch  420 | Loss 0.13729 | Correct   49 | Time 0.0935s
Epoch  430 | Loss 2.15317 | Correct   47 | Time 0.0942s
Epoch  440 | Loss 0.11118 | Correct   49 | Time 0.0941s
Epoch  450 | Loss 0.08229 | Correct   49 | Time 0.0941s
Epoch  460 | Loss 1.33169 | Correct   49 | Time 0.0941s
Epoch  470 | Loss 0.26228 | Correct   49 | Time 0.1037s
Epoch  480 | Loss 0.27676 | Correct   49 | Time 0.1106s
Epoch  490 | Loss 0.70482 | Correct   49 | Time 0.0941s
Epoch  499 | Loss 0.10956 | Correct   50 | Time 0.0941s
```

Training Statistics:
* Total Training Time: 73.37s
* Average Epoch Time: 0.1467s
* Final Accuracy: 100.0% (50/50 correct)
* Final Loss: 0.10956