---
title: "Instruction set"
linkTitle: "Instruction set"
date: 2022-03-03
description: >
  A description of the Tensor Compute Unit instruction set
---



| Name | Description | Opcode | Flags | Operand #0 | Operand #1 | Operand #2 |
|-|-|-|-|-|-|-|
| NoOp        | Do nothing                                                                                 | 0x0     | \-                                 | \-                          | \-                                 | \-                   |
| MatMul      | Load input at memory address into systolic array and store result at accumulator address   | 0x1     | Accumulate? Zeroes?                | Local Memory stride/address | Accumulator stride/address         | Size                 |
| DataMove    | Move data between the main memory and either the accumulators or one of two off-chip DRAMs | 0x2     | Data flow control enum (see below) | Local Memory stride/address | Accumulator or DRAM stride/address | Size                 |
| LoadWeight  | Load weight from memory address into systolic array                                        | 0x3     | Zeroes? (Ignores operand #0)       | Local Memory stride/address | Size                               | \-                   |
| SIMD        | Perform computations on data in the accumulator                                            | 0x4     | Read? Write? Accumulate?           | Accumulator write address   | Accumulator read address           | SIMD sub-instruction |
| LoadLUT     | Load lookup tables from memory address.                                                    | 0x5     | \-                                 | Local Memory stride/address | Lookup table number                | \-                   |
| &lt;unused> | \-                                                                                         | 0x6-0xE | \-                                 | \-                          | \-                                 | \-                   |
| Configure   | Set configuration registers                                                                | 0xF     | \-                                 | Register number             | Value                              | \-                   |


## Notes

- Weights should be loaded in reverse order

- Since Size = 0 doesn’t make sense, the size argument is interpreted as 1 less than the size of data movement requested i.e.

  - size = 0 means transfer 1 vector
  - size = 1 means transfer 2 vectors
  - size = 255 means transfer 256 vectors etc.

- Instruction width is a parameter supplied to the RTL generator

  - Opcode field is always 4 bits
  - Flags field is always 4 bits
  - Instruction must be large enough to fit the maximum values of all operands in the longest instruction (MatMul, DataMove, SIMD)

- Flags are represented in the following order: \[3 2 1 0]

  - i.e. the first flag listed is at bit 0 (the 4th bit), second flag is at bit 1 (the 3rd bit) etc.

- Arguments are in the following order: \[2 1 0]

  - e.g. in MatMul the bits of the instruction will be, from most significant bit to least: opcode, optional zero padding, accumulate?, size, accumulator stride/address, memory stride/address

  - Address unit for all memories is one array vector

  - Stride has a fixed number of bits followed by the number of bits for the address of the largest memory that may appear in the operand. The address for smaller memories gets padded by zeros. Stride is encoded as power of 2. For example the 3-bit stride is as follows 000=1, 001=2, 010=4, 011=8,.. 111=128

    - e.g. in a 2-byte argument with an 11-bit address and a 3-bit stride, the bit layout would be

      - 15:14 = padding, to be set to zeroes
      - 13:11 = stride
      - 10:0 = address

  - Size unit is one array vector

- Data flow control enum flag values are:

  - 0b0000 = 0 = 0x0 = DRAM0 to memory
  - 0b0001 = 1 = 0x1 = memory to DRAM0
  - 0b0010 = 2 = 0x2 = DRAM1 to memory
  - 0b0011 = 3 = 0x3 = memory to DRAM1
  - ...
  - 0b1100 = 12 = 0xc = accumulator to memory
  - 0b1101 = 13 = 0xd = memory to accumulator
  - 0b1110 = 14 = 0xe = &lt;reserved>
  - 0b1111 = 15 = 0xf = memory to accumulator (accumulate)

- SIMD instructions have some subtleties

  - can read or write (+accumulate) in the same instruction

  - when the read flag is set, data is read from the accumulators into the ALU array

  - when the write flag is set, the ALU array output is written into the accumulators

    - the accumulate flag determines whether this is an accumulate or an overwrite

    - the output to be written is computed from the input that was read in on the same instruction

      - i.e. if \`x\` is read from the accumulators at the specified read address, and the ALU computes \`f(\_)\` then \`f(x)\` will be written to the accumulators at the specified write address from the same instruction

    - data is output from the ALU array on every instruction i.e. even if the destination is specified as register 1, you can still write into the accumulators from the output

  - before reading out from the accumulators with a DataMove, you should wait at least 2 instructions since the last SIMD instruction in which the write flag was high. This is because the data takes about 2 instructions to propagate into the accumulators from the ALUs. The easiest way to achieve this is just to insert 2 no-ops before the DataMove instruction.

    - 2 instructions is an empirical estimate. The number may need to be higher in certain cases. If you see data being dropped/corrupted/repeated, talk to[tom@tensil.ai](mailto:tom@tensil.ai) about it


## SIMD sub-instructions

All SIMD instructions are composed of 4 parts: opcode, source left, source right and destination. The widths are as follows:

- opcode = ceil(log2(numOps))

  - numOps is currently fixed at 16, so opcode is 4 bits

- source left = ceil(log2(numRegisters+1))

  - numRegisters is currently fixed at 1, so source left is 1 bit

- source right = source left

- dest = source left

Source left is the left argument for binary operations, and the single argument for unary operations. Source right is the right argument for binary operations, and is ignored for unary operations.

The_Move_ opcode allows you to move data from one register to another, or to read the data in a register to output. The _NoOp_ opcode is only a true no-op when both the read and write flags are set to false in the SIMD instruction. Otherwise, _NoOp_ has an overloaded meaning: it is used to trigger an external read or write. That is, to write into the accumulators from the PE array, or to read out from the accumulators into on-chip main memory.

| **Opcode**                                                                                                                                                                                                                                 | **Source left**                          | **Source right**                         | **Destination**                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- | ---------------------------------------- | ----------------------------------------------------------- |
| 0x00 = NoOp\*\*<br/>0x01 = Zero<br/>0x02 = Move\*<br/>0x03 = Not\*<br/>0x04 = And<br/>0x05 = Or<br/>0x06 = Increment\*<br/>0x07 = Decrement\*<br/>0x08 = Add<br/>0x09 = Subtract<br/>0x0A = Multiply<br/>0x0B = Abs\*<br/>0x0C = GreaterThan<br/>0x0D = GreaterThanEqual<br/>0x0E = Min<br/>0x0F = Max<br/>0x10 = Lookup\* | 0 = input<br/>1 = register 1<br/>2 = register 2... | 0 = input<br/>1 = register 1<br/>2 = register 2... | 0 = output<br/>1 = output & register 1<br/>2 = output & register 2... |

\*unary operation

\*\*arguments are ignored


### Lookup sub-instruction

The lookup sub-instruction returns_N+1_ result values where _N_ is the number of lookup tables in the architecture. The results are, in order

- the difference between the argument and the closest key found in the lookup table index
- the value found in the first lookup table
- the value found in the second lookup table
- etc.

The destination register_d_specifies the register which will receive the first result i.e. the difference. The remaining results will be populated in the registers numbered ascending from_d_, so that the result from lookup table_i_ will be written to register _d+i+1_. This assumes that the architecture is configured with sufficient registers to store all the results, and that_d+N_ &lt;= total number of registers. The behaviour of the lookup sub-instruction is undefined if these requirements aren’t met.
