from zipfile import ZipFile
from os.path import basename

import os
import sys

def build_model(operations):
    # open file + clear file
    file1 = open("data/model.py", "a")
    file1.truncate(0)

    track_index = {"conv2d": 0, "maxpool2d": 0, "relu":0, "dense":0}

    setup_imports(file1)
    file1.write("\n")

    file1.write("class Net(nn.Module):\n")
    file1.write("    def __init__(self):\n")
    file1.write("        super(Net, self).__init__()\n")

    last_output_size = operations[0][3]
    forward_operations = []
    for ops in operations[1:]:
        index = track_index[ops[0]]
        track_index[ops[0]] += 1

        if ops[0] == "conv2d":
            variable = write_conv_2d(file1, last_output_size, ops[1], ops[2], index)
            forward_operations.append(variable)
            last_output_size = ops[1]

        elif ops[0] == "maxpool2d":
            variable = write_pool_2d(file1, ops[1], ops[2], index)
            forward_operations.append(variable)

        elif ops[0] == "relu":
            variable = write_relu(file1, index)
            forward_operations.append(variable)

        elif ops[0] == "dense":
            variable = write_dense(file1, last_output_size, ops[1], index)
            forward_operations.extend(variable)
            last_output_size = ops[1]

    file1.write("\n\n")
    file1.write("    def forward(self, x):\n")
    for op in forward_operations:
        file1.write(f"        x = {op}\n")
    file1.write("        return x\n")
    file1.write("\n")
    file1.close()


def setup_imports(input_file):
    input_file.write("import torch\n")
    input_file.write("import torch.nn as nn\n")
    input_file.write("import torch.nn.functional as F\n")
    input_file.write("import torchvision\n")
    input_file.write("import torchvision.transforms as transforms\n")

def write_conv_2d(input_file, last_output_size, filters, kernel, index):
    variable = f"self.conv2D_{index}(x)"
    line = f"nn.Conv2d({last_output_size}, {filters}, {kernel})"
    input_file.write("        " + variable[:-3] + " = " + line + "\n")
    return variable

def write_pool_2d(input_file, kernel_size, stride, index):
    variable = f"self.pool_{index}(x)"
    line = f"nn.MaxPool2d({kernel_size}, {stride})"
    input_file.write("        " + variable[:-3] + " = " + line + "\n")
    return variable

def write_relu(input_file, index):
    variable = f"self.relu_{index}(x)"
    line = f"F.relu"
    input_file.write("        " + variable[:-3] + " = " + line + "\n")
    return variable

def write_dense(input_file, last_output_size, nodes, index):
    if index==0:
        last_output_size = last_output_size * 5 * 5
        variable_2 = f"x = x.view(-1, {last_output_size})"
        variable = f"self.linear_{index}(x)"
        line = f"nn.Linear({last_output_size}, {nodes})"
        input_file.write("        " + variable[:-3] + " = " + line + "\n")
        return [variable_2, variable]

    else:
        variable = f"self.linear_{index}(x)"
        line = f"nn.Linear({last_output_size}, {nodes})"
        input_file.write("        " + variable[:-3] + " = " + line + "\n")
        return [variable]

def build_dataset(dataset_name, batch_size, num_workers):
    out_file = open("data/train.py", "a")
    out_file.truncate(0)
    
    out_file.write("import torch\n")
    out_file.write("import torch.nn as nn\n")
    out_file.write("import torchvision\n")
    out_file.write("import torchvision.transforms as transforms\n")
    out_file.write("\n")
    out_file.write("def train():\n")
    out_file.write("    transform = transforms.Compose(\n")
    out_file.write("        [transforms.ToTensor(),\n")
    out_file.write("        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])\n")
    out_file.write("\n")
    out_file.write(f"    trainset = torchvision.datasets.{dataset_name}(root='./data', train=True,\n")
    out_file.write("                                        download=True, transform=transform)\n")
    out_file.write(f"    trainloader = torch.utils.data.DataLoader(trainset, batch_size={batch_size},\n")
    out_file.write(f"                                              shuffle=True, num_workers={num_workers})\n")
    out_file.write("\n")
    out_file.write(f"    testset = torchvision.datasets.{dataset_name}(root='./data', train=False,\n")
    out_file.write("                                    download=True, transform=transform)\n")
    out_file.write(f"    testloader = torch.utils.data.DataLoader(testset, batch_size={batch_size},\n")
    out_file.write(f"                                             shuffle=False, num_workers={num_workers})\n")

def build_training_loop(epoch, lr, momentum, loss, PATH):
    file1 = open("data/train.py", "a")
    file1.write("\n")
    file1.write("    from model import Net\n")
    file1.write("    import torch.optim as optim\n")
    file1.write("\n")
    file1.write("    net = Net()\n")
    file1.write(f"    criterion = nn.{loss}()\n")
    file1.write(f"    optimizer = optim.SGD(net.parameters(), lr={lr}, momentum={momentum})\n")
    file1.write("\n")
    file1.write(f"    for EPOCH in range({epoch}):\n")
    file1.write("        running_loss = 0.0\n")
    file1.write("        for i, data in enumerate(trainloader, 0):\n")
    file1.write("            # get the inputs; data is a list of [inputs, labels]\n")
    file1.write("            inputs, labels = data\n")
    file1.write("\n")
    file1.write("            #zero the parameter gradients\n")
    file1.write("            optimizer.zero_grad()\n")
    file1.write("\n")
    file1.write("            # forward + backward + optimize\n")
    file1.write("            outputs = net(inputs)\n")
    file1.write("            loss = criterion(outputs, labels)\n")
    file1.write("            loss.backward()\n")
    file1.write("            optimizer.step()\n")
    file1.write("\n")
    file1.write("            # print statistics\n")
    file1.write("            running_loss += loss.item()\n")
    file1.write("            if i % 2000 == 1999: # print every 2000 mini-batches\n")
    file1.write("                print('[%d, %5d] loss: %.3f' %\n")
    file1.write("                    (EPOCH + 1, i + 1, running_loss / 2000))\n")
    file1.write("                running_loss = 0.0\n")
    file1.write("        torch.save(dict(epoch=EPOCH,\n")
    file1.write("                   model_state_dict= net.state_dict(),\n")
    file1.write("                   optimizer_state_dict= optimizer.state_dict(),\n")
    file1.write("                   loss= loss,\n")
    file1.write(f"                  ), '{PATH}')\n")
    file1.write("\n")
    file1.write("    print('Finished Training')\n")
    file1.write("\n")
    file1.write("if __name__=='__main__':\n")
    file1.write("    train()")


