---
title: "Opsets"
linkTitle: "Opsets"
date: 2022-03-23
weight: 3
description: >
  Supported operations
---

## Tensorflow

| Operation        | Comments                                                                              |
| ---------------- | ------------------------------------------------------------------------------------- |
| MatMul           |                                                                                       |
| Conv2D           | Only SAME and VALID paddings are supported.                                           |
| BiasAdd          |                                                                                       |
| ResizeBilinear   | Resize image with align corners is not supported.                                     |
| FusedBatchNormV3 |                                                                                       |
| MaxPool          | Only SAME and VALID paddings are supported.                                           |
| AvgPool          | Only SAME and VALID paddings are supported.                                           |
| Mean             | Only channel mean is supported.                                                       |
| Relu             |                                                                                       |
| LeakyRelu        |                                                                                       |
| AddV2            |                                                                                       |
| ConcatV2         | Only last dimension concat is supported.                                              |
| Split            | Only last dimension split is supported.                                               |
| SplitV           | Only last dimension split is supported.                                               |
| Pad              | Only 4D padding is supported. Only height/width padding is supported.                 |
| Reshape          |                                                                                       |
| Cast\*           | Only DT\_INT32 to DT\_FLOAT cast is supported.                                        |
| Tile\*           |                                                                                       |
| Pack\*           | Only first axis pack is supported.                                                    |
| StridedSlice\*   | Only 1D strided slice is supported. Only strided slice with shrink axis is supported. |
| Shape\*          |                                                                                       |

* Only compile-time constants folding

## Onnx

We support a subset of ONNX v8.

