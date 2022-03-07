---
title: "Benchmarks"
linkTitle: "Benchmarks"
date: 2022-03-07
description: >
  Performance benchmarks and information
---

## Methodology

Benchmarks are generated using the Tensil compiler. Each instruction is evaluated against a latency model to compute expected execution time. Actual results may therefore differ somewhat from the numbers listed here. [Help us](https://github.com/tensil-ai/tensil/blob/main/tools/src/tensil/tools/compiler/BackendStream.scala#L289) improve the latency model!

## ResNet-20v2

Trained for CIFAR.

|FPGA Board|Tensil Array Size|Clock (MHz)|Latency (ms)|Frames per second|
|----|-----------------|-----------|------------|-----------------|
|[Arty A7-35](https://digilent.com/reference/programmable-logic/arty-a7/start)|8x8|150|21|48|
|[Pynq Z1](https://digilent.com/reference/programmable-logic/pynq-z1/start)|12x12|150|14|71|
|[Ultra96-V2](https://www.avnet.com/wps/portal/us/products/avnet-boards/avnet-board-families/ultra96-v2/)|16x16|300|4|250|

## YoloV4-tiny

Trained for ImageNet.


|FPGA Board|Tensil Array Size|Clock (MHz)|Latency (ms)|Frames per second|
|----|-----------------|-----------|------------|-----------------|
|[Arty A7-35](https://digilent.com/reference/programmable-logic/arty-a7/start)|8x8|150|175|5.7|
|[Pynq Z1](https://digilent.com/reference/programmable-logic/pynq-z1/start)|12x12|150|112|8.9|
|[Ultra96-V2](https://www.avnet.com/wps/portal/us/products/avnet-boards/avnet-board-families/ultra96-v2/)|16x16|300|36|28|


## ResNet-50v2

Trained for ImageNet.


|FPGA Board|Tensil Array Size|Clock (MHz)|Latency (ms)|Frames per second|
|----|-----------------|-----------|------------|-----------------|
|[Arty A7-35](https://digilent.com/reference/programmable-logic/arty-a7/start)|8x8|150|1969|0.5|
|[Pynq Z1](https://digilent.com/reference/programmable-logic/pynq-z1/start)|12x12|150|833|1.2|
|[Ultra96-V2](https://www.avnet.com/wps/portal/us/products/avnet-boards/avnet-board-families/ultra96-v2/)|16x16|300|260|3.8|