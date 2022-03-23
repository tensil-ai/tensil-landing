---
title: "High-level intermediate representation"
linkTitle: "HIR"
date: 2022-03-23
description: >
  Explanation of HIR
---

High-level intermediate representation (HIR) is an interface offered by the scheduler. It expresses common ML operations abstract from specific ML frameworks, such as ONNX. It also operates in terms of memory objects.

Following are a few examples of HIR.

```scala
def emitMatMul(
      weightsObjs: Seq[MemoryObject],
      biasObj: Option[MemoryObject],
      inputOutputPairs: Seq[MemoryOptionalInputOutputObjects]
  ): Unit
```

The `emitMatMul` function takes weights and the bias memory objects, and a sequence of input-output object pairs. It performs matrix multiplication for each input memory object and places results in the output memory object. Input is optional, in which case it is assumed to be all zeroes. Weights and the bias must be Consts objects. Input must be Vars object, and output must be Temp object.

```scala
def emitRelu(
      inputObj: MemoryObject,
      outputObj: MemoryObject
  ): Unit
```

The `emitRelu` function performs ReLU activation on the input object and places the result in the output object. Both input and output must be Temp objects.

```scala
def emitSave(
      inputObj: MemoryObject,
      outputObj: MemoryObject
  ): Unit
```

The `emitSave` function moves data from the Temp input object to Vars output object, usually at the end of the layer.
