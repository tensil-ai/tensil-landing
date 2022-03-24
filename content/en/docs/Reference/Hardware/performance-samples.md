---
title: "Performance samples"
linkTitle: "Performance samples"
date: 2022-03-03
description: >
  A description of the Tensor Compute Unit performance samples
---

## Performance sampling

The program counter and decoder control bus handshake signals can be sampled at a fixed interval of L cycles in order to measure system performance. The samples are written out to the sample IO bus in blocks of N sample words. The block is terminated by asserting the AXI stream TLAST signal. Each sample word is a 64-bit word, with the following meaning:

| Bus name    | Signal | Bit field(s) | Comments  |
| --------------- | ---------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Program counter || 0:31             | Contains all 1s if the sample is invalid. Invalid samples are produced when the sampling interval is set to 0. |                                           |
| Array           | Valid            | 32                                                                                                             | Contains all 0s if the sample is invalid. |
|| Ready           | 33               |                                                                                                                |                                           |
| Acc             | Valid            | 34                                                                                                             |                                           |
|| Ready           | 35               |                                                                                                                |                                           |
| Dataflow        | Valid            | 36                                                                                                             |                                           |
|| Ready           | 37               |                                                                                                                |                                           |
| DRAM1           | Valid            | 38                                                                                                             |                                           |
|| Ready           | 39               |                                                                                                                |                                           |
| DRAM0           | Valid            | 40                                                                                                             |                                           |
|| Ready           | 41               |                                                                                                                |                                           |
| MemPortB        | Valid            | 42                                                                                                             |                                           |
|| Ready           | 43               |                                                                                                                |                                           |
| MemPortA        | Valid            | 44                                                                                                             |                                           |
|| Ready           | 45               |                                                                                                                |                                           |
| Instruction     | Valid            | 46                                                                                                             |                                           |
|| Ready           | 47               |                                                                                                                |                                           |
| &lt;unused>     | 48:64            |                                                                                                                |                                           |

Value of L can be changed by setting the configuration register. Value of N is defined by architecture.
