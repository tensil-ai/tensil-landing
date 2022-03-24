---
title: "Frontend"
linkTitle: "Frontend"
date: 2022-03-23
weight: 2
description: >
  Description of compiler frontend
---

The frontend is responsible for handling the compiler's primary input--an ML model. With many ML frameworks in existence, the compiler isolates the specific framework support in the frontend. In other words, we envision multiple dedicated frontends able to handle models created by each ML framework. Currently, there are two frontends supporting TensorFlow and ONNX with input in the form of _model.pb_ and _model.onnx_ files correspondingly. The frontend parses the model, represented in the form of a graph. It uses one or more output nodes to linearize the graph in a series of nodes respecting dataflow dependencies.

The frontend then processes this linearized series. During this processing, the frontend is grouping model nodes to form _layers_. Each layer represents one entire cycle started with matrix multiplication, followed by a series of accumulator operations and finalized with moving the result out of accumulators. In essence, the content of accumulators and systolic array weights is never shared between layers.

The frontend interacts with the _memory manager_ to obtain necessary _memory objects_. There are two banks of memory directly accessible to the host: DRAM0 and DRAM1. The compiler dedicates DRAM0 to store variable data objects (_Vars_) such as inputs, outputs, and the data passed between layers. Next, it dedicates DRAM1 to various constants (_Consts_), such as matrix multiplication weights and bias, constants used in accumulator operations, and constants used to _blend_ with variable data objects (like zero-padding). The frontend creates a new instance of the _scheduler_ for each layer and submits a series of high-level intermediate representation (_HIR_) operations based on model nodes present in the layer. The frontend allocates special temporary (_Temp_) memory objects to pass the data between HIR operations within a single layer. The scheduler is later responsible for mapping this temporary memory to available accumulators.
