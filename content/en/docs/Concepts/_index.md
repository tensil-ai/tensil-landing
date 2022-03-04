---
title: "Concepts"
linkTitle: "Concepts"
weight: 10
description: >
  Key concepts that will help you understand what Tensil does
---

## Register Transfer Level (RTL) code

RTL is code that describes the behaviour of computational hardware. It contains constructs like modules, input and output ports, signals, registers and low-level operations. Typical RTL languages are Verilog and VHDL. An example of Verilog is shown below. Electronic design automation (EDA) tools can turn RTL into descriptions of physically realizable circuits, which can be flashed onto an FPGA or taped out as an ASIC.


```verilog
module foo(
  input a,
  input b,
  output c
);
  assign c = a || b;
endmodule
```

## RTL Generator

An RTL generator produces a blob of RTL given some high level architectural parameters. This allows you to easily create customized RTL
that is specialized for a given application or use case without having to redesign the whole system. Tensil contains an RTL generator for ML accelerators.

## Model compiler

A model compiler takes an ML model and a target architecture and produces binary artifacts that can be executed by that architecture. In Tensil's case, the model
compiler produces three artifacts. The `.tprog` file is the executable containing
instructions to be interpreted by the accelerator, the `.tdata` file contains the model's parameters in the appropriate format, and the `.tmodel` file tells the driver how to set up inputs and outputs.


## Driver

A driver takes an architecture description and a compiled model and interacts with the abstractions in the operating system or execution environment (i.e. low-level libraries) to feed the compiled model into the hardware. It is also responsible for setting up inputs and outputs, and managing any other resources that might be revelant on a given hardware platform.

