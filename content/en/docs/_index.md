
---
title: "Welcome to Tensil"
linkTitle: "Documentation"
weight: 20
menu:
  main:
    weight: 20
---

What if you could just run this to get a custom ML accelerator specialized to your needs?

```
$ tensil rtl --arch <my_architecture>
```


What if compiling your ML model for that accelerator target was as easy as running this?

```
$ tensil compile --arch <my_architecture> --model <my_model>
```

Wonder no more: with Tensil you can!


## What is Tensil?

Tensil is a set of tools for running machine learning models on custom accelerator
architectures. It includes an RTL generator, a model compiler, and a set of drivers. It enables you to create a custom accelerator, compile an ML model targeted at it, and then deploy and run that compiled model.

The primary goal of Tensil is to allow anyone to accelerate their ML workloads. Currently, we are focused on supporting convolutional neural network inference on edge FPGA (field programmable gate array) platforms, but we aim to support all
model architectures on a wide variety of fabrics for both training and inference.

You should use Tensil if:

- you have a convolutional neural network based ML workload
- you need to run it at the edge (i.e. not in a data-center)
- you want to avoid changing your model to make it work on a GPU/CPU
- you want to offload heavy computation from your host CPU or microcontroller

## Unique benefits of Tensil

With Tensil you can:

- run your model as-is, without quantization or other degradation
- achieve significantly better performance per watt
- make use of a huge variety of FPGA platforms

## Limitations of Tensil (for now)

At present, these are Tensil's limitations:

- only supports convolutional neural networks
- driver support for FPGAs only

Join us on Discord or on Github to help us plan our roadmap!

## Where should I go next?

Select a section below to dive in. We recommend beginning at [Getting Started](/docs/getting-started/).


