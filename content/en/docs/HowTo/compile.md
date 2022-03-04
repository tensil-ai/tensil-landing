---
title: "Compile an ML model"
date: 2022-03-03
weight: 2
description: >
  How to compile your ML model for an accelerator architecture
---

## Things you'll need

- your ML model. If you don't have one handy, continue on to use one of the demo ones.
- an architecture file in `.tarch` format. If you don't know what this is yet, continue on and we'll supply one for you.

## 1. Convert your ML model to ONNX

The first thing you need to do is convert your ML model to the ONNX format. ONNX stands for Open Neural Network Exchange, and converting to ONNX is supported by all the major frameworks. Instructions for:

- [Tensorflow, Tflite and Keras](https://github.com/onnx/tensorflow-onnx/blob/main/README.md)
- [PyTorch](https://pytorch.org/docs/stable/onnx.html)
- [Others](https://onnx.ai/supported-tools.html)


## 2. Run the Tensil compiler

First, ensure you have Tensil installed by pulling and running the Tensil Docker container:

```bash
$ docker pull tensilai/tensil:latest
$ docker run -v $(pwd):/work -w /work -it tensilai/tensil:latest bash
```

Then from the container shell, run:

```bash
$ tensil compile -a <tarch_file> -m <onnx_file> -o output_node -s true
```

To compile with an example model and architecture file, the command is
```bash
$ tensil compile -a /demo/arch/ultra96v2.tarch -m /demo/models/resnet20v2_cifar.onnx -o "Identity:0" -s true
```

You should see some output like this:

```
$ tensil compile -a /demo/arch/ultra96v2.tarch -m /demo/models/resnet20v2_cifar.onnx -o "Identity:0" -s true
NCHW[1,3,32,32]=NHWC[1,32,32,1]=1024*16
List(-1, 256)
----------------------------------------------------------------------------------------------
COMPILER SUMMARY
----------------------------------------------------------------------------------------------
Model:                                           resnet20v2_cifar_onnx_ultra96v2 
Data type:                                       FP16BP8                         
Array size:                                      16                              
Consts memory size (vectors/scalars/bits):       2,097,152                       33,554,432 21
Vars memory size (vectors/scalars/bits):         2,097,152                       33,554,432 21
Local memory size (vectors/scalars/bits):        20,480                          327,680    15
Accumulator memory size (vectors/scalars/bits):  4,096                           65,536     12
Stride #0 size (bits):                           3                               
Stride #1 size (bits):                           3                               
Operand #0 size (bits):                          24                              
Operand #1 size (bits):                          24                              
Operand #2 size (bits):                          16                              
Instruction size (bytes):                        9                               
Consts memory maximum usage (vectors/scalars):   35,743                          571,888    
Vars memory maximum usage (vectors/scalars):     13,312                          212,992    
Consts memory aggregate usage (vectors/scalars): 35,743                          571,888    
Vars memory aggregate usage (vectors/scalars):   46,097                          737,552    
Number of layers:                                23                              
Total number of instructions:                    102,741                         
Compilation time (seconds):                      71.562                          
True consts scalar size:                         568,474                         
Consts utilization (%):                          97.210                          
True MACs (M):                                   61.476                          
MAC efficiency (%):                              0.000                           
----------------------------------------------------------------------------------------------
---------------------------------------------
ARTIFACTS
---------------------------------------------
Manifest:  /work/resnet20v2_cifar_onnx.tmodel
Constants: /work/resnet20v2_cifar_onnx.tdata
Program:   /work/resnet20v2_cifar_onnx.tprog
---------------------------------------------
```

If you got an error or saw something you didn't expect, please let us know! You can either join our [Discord]() to ask a question, [open an issue on Github](https://github.com/tensil-ai/tensil/issues/new) or email us at [support@tensil.ai](mailto:support@tensil.ai).


## Next Steps

Congrats! You've compiled your model and generated three important artifacts, a `.tmodel`, `.tdata` and `.tprog`. All three are needed to run your compiled model,
so keep them handy. Assuming you have an accelerator built, you're now ready to [run your model]({{< relref "/docs/howto/run" >}}). If not, it's time to [generate an accelerator]({{< relref "/docs/howto/generate" >}}).

## Troubleshooting

### Converting to ONNX didn't work?

If you're using Tensorflow and the ONNX converter failed, don't despair! We also support compiling from a frozen graph in PB format. To freeze a Tensorflow model, use the `freeze_graph` tool located [here in the Tensorflow repo](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py).

If you have Tensorflow installed, you can use it in a script by doing

```python
from tensorflow.python.tools.freeze_graph import freeze_graph

graph_def = "some_graph_def.pb"
ckpt = "model.ckpt-1234567"
output_graph = "frozen_graph.pb"
output_nodes = ["softmax"]
input_binary = graph_def.split(".")[-1] == "pb"

freeze_graph(
        graph_def,
        "",
        input_binary,
        ckpt,
        ",".join(outputs_nodes),
        "save/restore_all",
        "save/Const:0",
        output_graph,
        True,
        )
```

or you can use it directly from the command line by running

```bash
python -m tensorflow.python.tools.freeze_graph \
 --input_graph=some_graph_def.pb --input_binary \
 --input_checkpoint=model.ckpt-1234567 \
 --output_graph=frozen_graph.pb --output_node_names=softmax
```