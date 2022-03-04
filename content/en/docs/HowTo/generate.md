---
title: "Generate an accelerator"
date: 2022-03-03
weight: 3
description: >
  How to generate an accelerator with a given architecture
---

## Things you'll need

- an architecture file in `.tarch` format. If you don't know what this is yet, continue on and we'll supply one for you.
- an AXI data width in bits (check your FPGA product page)


## 1. Run the Tensil RTL generator

First, ensure you have Tensil installed by pulling and running the Tensil Docker container:

```bash
$ docker pull tensilai/tensil:latest
$ docker run -v $(pwd):/work -w /work -it tensilai/tensil:latest bash
```

Then from the container shell, run:

```bash
$ tensil rtl -a <tarch_file> -d <axi_port_width>
```

To compile with an example model and architecture file, the command is
```
$ tensil rtl -a /demo/arch/ultra96v2.tarch -d 128
```

You should see some output like this:

```
$ tensil rtl -a /demo/arch/ultra96v2.tarch -d 128
Elaborating design...
Done elaborating.
-------------------------------------------------------
ARTIFACTS
-------------------------------------------------------
Verilog bram_dp_256x4096:   /work/bram_dp_256x4096.v
Verilog bram_dp_256x20480:  /work/bram_dp_256x20480.v
Verilog top_ultra96v2:      /work/top_ultra96v2.v
Driver parameters C header: /work/architecture_params.h
-------------------------------------------------------
```


## Next Steps

You've generated several RTL artifacts (the files ending in `.v`) - now it's time to [integrate them into your system]({{< relref "/docs/howto/integrate" >}}).


## Troubleshooting

### I can't figure out what AXI width to use

Here's a table with some known values:

|FPGA Family     |AXI Data Width|Tensil Flag|
|----------------|--------------|-----------|
|Zynq-7000       |64 bit        |`-d 64`    |
|Zynq Ultrascale+|128 bit       |`-d 128`   |

If your FPGA family isn't listed and you need help, ask a question on [Discord](https://discord.gg/TSw34H3PXr) or
email us at [support@tensil.ai](mailto:support@tensil.ai).