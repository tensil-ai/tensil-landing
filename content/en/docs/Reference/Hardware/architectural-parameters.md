---
title: "Architectural paremeters"
linkTitle: "Architectural parameters"
date: 2022-03-03
description: >
  A list of architectural parameters and their descriptions
---

|Parameter|Description|Allowable values|Example value|
|--------------|-----------|----------------|-------------|
|Data type     |The numerical format used to perform calculations in hardware|`FP16BP8`,`FP32B16`|`FP16BP8`, which means "Fixed point format with width 16 bits and with the binary point at 8 bits"|
|Array size|The size of the systolic array and also the number of scalars in each vector|2-256|8|
|DRAM0 depth|The number of vectors allocated in DRAM bank 0|2^{1-32}|1048576 (= 2^20)|
|DRAM1 depth|The number of vectors allocated in DRAM bank 1|2^{1-32}|1048576 (= 2^20)|
|Local depth|The number of vectors allocated in on-fabric main memory|2^{1-16}|16384 (= 2^14)|
|Accumulator depth|The number of vectors allocated in on-fabric accumulator memory|2^{1-16}|4096 (= 2^12)|
|SIMD registers depth|The number of registers to instantiate for each ALU in the SIMD module|0-16|1|
|Number of threads|The number of threads for managing concurrent execution|1-8|2|