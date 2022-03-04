---
# categories: ["Examples", "Placeholders"]
# tags: ["test","docs"] 
title: "Getting Started"
linkTitle: "Getting Started"
weight: 2
description: >
  The essentials for getting started with Tensil
---

## Prerequisites

The easiest way to get started with Tensil is through our Docker containers. Therefore, we recommend [installing Docker](https://docs.docker.com/engine/install/) before continuing.

## Installation

To install from Docker, run:

```bash
$ docker pull tensilai/tensil:latest
$ docker run -v $(pwd):/work -w /work -it tensilai/tensil:latest bash
```

You will be dropped into a shell inside the Tensil container. Run

```bash
$ tensil compile --help
```

to verify that it is working correctly.

## Try it out!

Try compiling an example ML model:

```bash
$ tensil compile -a /demo/arch/ultra96v2.tarch -m /demo/models/resnet20v2_cifar.onnx -o "Identity:0" -s true
```

Next up, try a [tutorial]({{< relref "/docs/tutorials" >}}) to learn how to use Tensil.

## For Contributors
### Installation from source

See the project [README](https://github.com/tensil-ai/tensil#for-maintainers) for instructions on how to build from source.
