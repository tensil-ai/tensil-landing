---
title: "Memory manager"
linkTitle: "Memory manager"
date: 2022-03-23
weight: 4
description: >
  Description of compiler memory manager
---

The memory manager is responsible for allocating and freeing, when necessary, memory objects. Memory object represents a series of memory addresses (_memory span_) with associated tensor _dimensions_. The scheduler uses dimensions to ensure the correctness of the dataflow. In addition, the memory manager is tracking _pending constants_ found in model nodes. The pending means that when the frontend processes the constant, it is unknown if it will become a memory object or be used as a parameter to one of the HIR operations. When a pending constant becomes a Const memory object, it gets emitted as a part of the _model.tdata_ file later used by the _driver_ to place into host memory. The memory manager also emits a memory map for Consts and Vars memories. Such a map is included in _model.tmodel_ file to inform the driver of the memory layout to place the content of _model.tdata_ file, as well as model's inputs and outputs.
