
---
title: "Show HN"
linkTitle: "Shown HN"
date: 2022-03-10
description: >
  Tensil open source annoucement on HN
---


We recently launched Tensil AI and to engage with an initial set of users we published
a [Show HN thread](https://news.ycombinator.com/item?id=30615605). Below are a summary of interesting questions and our responses with
links to documentation for reference.

## Why FPGAs?

There were several questions [1]](https://news.ycombinator.com/item?id=30615605#30622434)[2](https://news.ycombinator.com/item?id=30615605#30617736)[3](https://news.ycombinator.com/item?id=30615605#30619695) about Tensil's choice to focus initially on FPGAs.

> What kind of FPGAs can this reasonably run on?

Generally speaking, FPGAs with some kind of DSP (digital signal processing) capability will work best, since they can most efficiently implement the multiply-accumulate operations needed.

Here are some examples in the benchmarks section of our docs: https://www.tensil.ai/docs/reference/benchmarks/

> So is this only for edge compute use cases, or can I use tensil on an FPGA I have running in my data centre?

You absolutely can use it in a data centre. You can even tape out an ASIC using these designs! Currently we've done most of our prototyping with edge FPGA platforms but if you want to try other platforms we'd love to help you get started.

> In terms of perf per watt, could FPGAs compete against a coral-style TPU?

FPGAs are pretty amazing devices, but one thing that's been holding them back is how difficult they have been to work with. Typically to actually make use of an FPGA you'd need to have an FPGA expert and an embedded software engineer on your team, along with all the requisite tools and materials. Our focus is on changing that dynamic, to help get your ML model running on an edge FPGA in minutes.


## What Models are Supported?

There were several questions [1]](https://news.ycombinator.com/item?id=30615605#30622434)[2](https://news.ycombinator.com/item?id=30615605#30622283) about Tensil's ability to support various ML model architectures.

>  I wonder if you plan to support CGRAs and LSTMs? 

Yes. We're aiming to support all machine learning model architectures. The broader point is that Tensil is extremely flexible, so you can try out lots of different accelerator configurations to find the one that works best for your ML model. Think of it as optimizing the hardware first, then the software if needed.

> Is performance model dependent?

Yes, and we provide tools to help you analyize various hardware configurations to find the best performance.

## What About Quantization?

Quantization is a common technique typically required with edge ML devices. Tensil makes this optional.

