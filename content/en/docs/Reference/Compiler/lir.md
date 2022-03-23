---
title: "Low-level intermediate representation"
linkTitle: "LIR"
date: 2022-03-23
description: >
  Explanation of LIR
---

Low-level intermediate representation (LIR) is an interface offered by the backend. It expresses instructions supported by the processing unit. Unlike HIR, it operates in terms of memory addresses. Each memory address is tagged with its memory type. While HIR memory objects are expected to contain Vars, Consts and Temp tagged addresses, LIR only accepts Vars, Consts, Local memory, and Accumulator tagged addresses. One of the key scheduler roles is to do this translation.

Following are a few examples of LIR. Each produces the corresponding processing unit instruction. Note that LIR is not using instruction flags directly. The backend role is to infer these flags based on LIR arguments, such as accumulate and `toLocal` booleans and memory address tags.

```scala
def emitMatMul(
      accumulate: Boolean,
      localStride: Int,
      localAddress: MemoryAddress,
      accumulatorStride: Int,
      accumulatorAddress: MemoryAddress,
      size: Long,
      comments: List[String] = List()
  ): Unit

def emitSIMD(
      accumulate: Boolean,
      simdOp: Int,
      simdSourceLeft: Int,
      simdSourceRight: Int,
      simdDestination: Int,
      writeAccumulatorAddress: MemoryAddress,
      readAccumulatorAddress: MemoryAddress,
      comments: List[String] = List()
  ): Unit

def emitDataMove(
      toLocal: Boolean,
      accumulate: Boolean,
      localAddress: MemoryAddress,
      address: MemoryAddress,
      size: Long,
      comments: List[String] = List()
  ): Unit
```