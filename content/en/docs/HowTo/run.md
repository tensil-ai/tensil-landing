---
title: "Run a compiled model"
date: 2022-03-03
weight: 5
description: >
  How to run your compiled model on a system with a Tensil accelerator
---

## Things you'll need

- an FPGA board (e.g. the [Ultra96-V2](https://www.avnet.com/wps/portal/us/products/avnet-boards/avnet-board-families/ultra96-v2/))
- a [compiled model]({{< relref "/docs/howto/compile" >}}) (e.g. the set of three files: `resnet20v2_cifar_onnx.tmodel`, `resnet20v2_cifar_onnx.tdata`, `resnet20v2_cifar_onnx.tprog`)
- a fully implemented bitstream (`.bit`) and a hardware handoff file (`.hwh`): if you don't have these, learn how to [integrate the RTL]({{< relref "/docs/howto/integrate" >}})

In this guide we'll assume you are using the [PYNQ](http://www.pynq.io/) execution environment, but we also support bare metal execution with our [embedded C driver](https://github.com/tensil-ai/tensil/tree/main/drivers/embedded). 

## 1. Move files onto the FPGA

With PYNQ, you can achieve this by running

```
$ scp <my_model>.t* xilinx@192.168.2.99:~/
```

and then doing the same for the `.bit` and `.hwh` files. For example:

```
$ scp resnet20v2_cifar_onnx.t* xilinx@192.168.2.99:~/
$ scp design_1_wrapper.bit xilinx@192.168.2.99:~/ultra96-tcu.bit
$ scp design_1.hwh xilinx@192.168.2.99:~/ultra96-tcu.hwh
```

Note that with PYNQ, the `.bit` and `.hwh` files must have the same name up to the extension.


## 2. Copy the Python driver onto the FPGA

If you haven't already cloned the repository, get the Tensil source code from [Github](https://github.com/tensil-ai/tensil/releases), e.g.

```
curl -L https://github.com/tensil-ai/tensil/archive/refs/tags/v1.0.0.tar.gz | tar xvz
```

Now copy the Python driver over:

```
$ scp -r tensil-1.0.0/drivers/tcu_pynq xilinx@192.168.2.99:~/
```

## 3. Execute

Now it's time to hand everything over to the driver and tell it to execute the model. This guide will only cover the bare necessities for doing so, go here for a more [complete example](https://github.com/tensil-ai/tensil/blob/main/notebooks/Tensil%20TCU%20Demo%20-%20ResNet-20%20CIFAR.ipynb).


### Import the Tensil driver

```python
from pynq import Overlay
import sys
sys.path.append('/home/xilinx')
from tcu_pynq.driver import Driver
from tcu_pynq.architecture import ultra96
```

### Flash the bitstream onto the FPGA

```python
bitstream = '/home/xilinx/ultra96-tcu.bit'
overlay = Overlay(bitstream)
tcu = Driver(ultra96, overlay.axi_dma_0)
```

### Load the compiled model

```python
resnet = '/home/xilinx/resnet20v2_cifar_onnx_ultra96v2.tmodel'
tcu.load_model(resnet)
```

### Run

Pass your input data to the driver in the form of a dictionary. You can see which inputs the driver expects by printing `tcu.model.inputs`. 

```python
img = ...
inputs = {'x:0': img}
outputs = tcu.run(inputs)
```

If all went well, `outputs` should contain the results of running your model.

## Next Steps

You've successfully run your compiled model on Tensil's accelerator implemented on your FPGA. You're ready to use this capability in your application. Reach out to us if you need help taking it from here.


## Troubleshooting

As always, if you run into trouble please ask a question on [Discord](https://discord.gg/TSw34H3PXr) or
email us at [support@tensil.ai](mailto:support@tensil.ai).
