---
title: "Scheduler"
linkTitle: "Scheduler"
date: 2022-03-23
weight: 6
description: >
  Description of compiler execution scheduler
---
![scheduler](/images/compiler/scheduler.png)

The scheduler is responsible for transforming the high-level intermediate representation (HIR) produced by the frontend to the low-level intermediate representation (LIR) consumed by the backend. The main objective of such transformation is to schedule HIR operations expressed in terms of relatively large Vars, Consts, and unlimited Temp memories to limited SRAM local memory and accumulators available to a specific configuration of the processing unit. Internally it achieves this by building a dataflow graph based on memory addresses and finding its maximum partitioning that fits the local memory and accumulators. Such a partition is called a _stage_. The scheduler then produces LIR for every stage independently. Like for the frontend layers, stages don't share weights in the systolic array nor the content of accumulators. At the moment, they don't share data in the local memory either, which we expect to change once the compiler has to work efficiently with larger-sized local memory.
