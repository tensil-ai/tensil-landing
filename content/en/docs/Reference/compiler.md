---
title: "Compiler architecture"
linkTitle: "Compiler architecture"
date: 2022-03-16
description: >
  Description of compiler concepts, components and their interaction
---

![architecture](/images/compiler/architecture.png)

- [Frontend](#frontend)
- [Memory Manager](#memory_manager)
- [HIR](#hir)
- [Scheduler](#scheduler)
- [LIR](#lir)
- [Backend](#backend)

## Frontend

The frontend is responsible for handling the compiler's primary input--an ML model. With many ML frameworks in existence, the compiler isolates the specific framework support in the frontend. In other words, we envision multiple dedicated frontends able to handle models created by each ML framework. Currently, there are two frontends supporting TensorFlow and ONNX with input in the form of _model.pb_ and _model.onnx_ files correspondingly. The frontend parses the model, represented in the form of a graph. It uses one or more output nodes to linearize the graph in a series of nodes respecting dataflow dependencies.

The frontend then processes this linearized series. During this processing, the frontend is grouping model nodes to form _layers_. Each layer represents one entire cycle started with matrix multiplication, followed by a series of accumulator operations and finalized with moving the result out of accumulators. In essence, the content of accumulators and systolic array weights is never shared between layers.

The frontend interacts with the _memory manager_ to obtain necessary _memory objects_. There are two banks of memory directly accessible to the host: DRAM0 and DRAM1. The compiler dedicates DRAM0 to store variable data objects (_Vars_) such as inputs, outputs, and the data passed between layers. Next, it dedicates DRAM1 to various constants (_Consts_), such as matrix multiplication weights and bias, constants used in accumulator operations, and constants used to _blend_ with variable data objects (like zero-padding). The frontend creates a new instance of the _scheduler_ for each layer and submits a series of high-level intermediate representation (_HIR_) operations based on model nodes present in the layer. The frontend allocates special temporary (_Temp_) memory objects to pass the data between HIR operations within a single layer. The scheduler is later responsible for mapping this temporary memory to available accumulators.

## Memory Manager

The memory manager is responsible for allocating and freeing, when necessary, memory objects. Memory object represents a series of memory addresses (_memory span_) with associated tensor _dimensions_. The scheduler uses dimensions to ensure the correctness of the dataflow. In addition, the memory manager is tracking _pending constants_ found in model nodes. The pending means that when the frontend processes the constant, it is unknown if it will become a memory object or be used as a parameter to one of the HIR operations. When a pending constant becomes a Const memory object, it gets emitted as a part of the _model.tdata_ file later used by the _driver_ to place into host memory. The memory manager also emits map files for Consts and Vars memories. Such a map file informs the driver of the memory layout to place data from or to the stream, such as _model.tdata_ file, input, and output.

## HIR

High-level intermediate representation (HIR) is an interface offered by the scheduler. It expresses common ML operations abstract from specific ML frameworks, such as ONNX. It also operates in terms of memory objects.

Following are a few examples of HIR.

```
def emitMatMul(
      weightsObjs: Seq[MemoryObject],
      biasObj: Option[MemoryObject],
      inputOutputPairs: Seq[MemoryOptionalInputOutputObjects]
  ): Unit
```

The `emitMatMul` function takes weights and the bias memory objects, and a sequence of input-output object pairs. It performs matrix multiplication for each input memory object and places results in the output memory object. Input is optional, in which case it is assumed to be all zeroes. Weights and the bias must be Consts objects. Input must be Vars object, and output must be Temp object.

```
def emitRelu(
      inputObj: MemoryObject,
      outputObj: MemoryObject
  ): Unit
```

The `emitRelu` function performs ReLU activation on the input object and places the result in the output object. Both input and output must be Temp objects.

```
def emitSave(
      inputObj: MemoryObject,
      outputObj: MemoryObject
  ): Unit
```

The `emitSave` function moves data from the Temp input object to Vars output object, usually at the end of the layer.

## Scheduler

![scheduler](/images/compiler/scheduler.png)

The scheduler is responsible for transforming the high-level intermediate representation (HIR) produced by the frontend to the low-level intermediate representation (LIR) consumed by the backend. The main objective of such transformation is to schedule HIR operations expressed in terms of relatively large Vars, Consts, and unlimited Temp memories to limited SRAM local memory and accumulators available to a specific configuration of the processing unit. Internally it achieves this by building a dataflow graph based on memory addresses and finding its maximum partitioning that fits the local memory and accumulators. Such a partition is called a _stage_. The scheduler then produces LIR for every stage independently. Like for the frontend layers, stages don't share weights in the systolic array nor the content of accumulators. At the moment, they don't share data in the local memory either, which we expect to change once the compiler has to work efficiently with larger-sized local memory.

## LIR

Low-level intermediate representation (LIR) is an interface offered by the backend. It expresses instructions supported by the processing unit. Unlike HIR, it operates in terms of memory addresses. Each memory address is tagged with its memory type. While HIR memory objects are expected to contain Vars, Consts and Temp tagged addresses, LIR only accepts Vars, Consts, Local memory, and Accumulator tagged addresses. One of the key scheduler roles is to do this translation.

Following are a few examples of LIR. Each produces the corresponding processing unit instruction. Note that LIR is not using instruction flags directly. The backend role is to infer these flags based on LIR arguments, such as accumulate and `toLocal` booleans and memory address tags.

```
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

## Backend

The backend is responsible for translating LIR into the _model.tprog_ and _model.tmodel_ files containing the binary representation of the processing unit program and the information required by the driver to feed the program into the processing unit. It computes the instruction layout based on compiler options such as memory and SIMD registers depth. To produce instruction binary form, the backend infers instruction flags based on LIR arguments.