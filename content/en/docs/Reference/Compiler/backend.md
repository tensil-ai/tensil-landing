---
title: "Backend"
linkTitle: "Backend"
date: 2022-03-23
weight: 8
description: >
  Description of compiler backend
---

The backend is responsible for translating LIR into the _model.tprog_ and _model.tmodel_ files containing the binary representation of the processing unit program and the information required by the driver to feed the program into the processing unit. It computes the instruction layout based on compiler options such as memory and SIMD registers depth. To produce instruction binary form, the backend infers instruction flags based on LIR arguments.