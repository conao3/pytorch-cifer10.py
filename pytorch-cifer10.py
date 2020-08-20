import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.tensorboard as tfutil

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)   # 32x32x3 -> 10x10x6
        self.pool = nn.MaxPool2d(2, 2)    # 10x10x16 -> 5x5x16
        self.conv2 = nn.Conv2d(6, 16, 5)  # 10x10x6 -> 10x10x16
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net().to(device)

# criterion = nn.CrossEntropyLoss()
criterion = nn.CosineEmbeddingLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.001)

# "./logs/pool2-right-celoss-adam-lr0_001"
with tfutil.SummaryWriter("./logs/cosEmbLoss-adam-lr0_001-momentum0_9") as writer:
    for epoch in range(6):
        running_loss = 0.0
        traintotal, traincorrect = 0, 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, trainpredicted = torch.max(outputs.data, 1)
            traintotal += labels.size(0)
            traincorrect += (trainpredicted == labels).sum().item()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                x, y = epoch*12000+i, running_loss/2000
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, y))
                writer.add_scalar("loss/train", y, x)
                writer.add_scalar("accuracy/train", 100 *
                                  traincorrect/traintotal, x)
                running_loss = 0.0
                traintotal, traincorrect = 0, 0

                with torch.no_grad():
                    testtotal, testcorrect = 0, 0
                    for testdata in testloader:
                        testimages, testlabels = testdata
                        testimages, testlabels = testimages.to(
                            device), testlabels.to(device)
                        testoutputs = net(testimages)
                        testloss = criterion(testoutputs, testlabels)

                        _, predicted = torch.max(testoutputs.data, 1)
                        testtotal += testlabels.size(0)
                        testcorrect += (predicted == testlabels).sum().item()

                    writer.add_scalar("accuracy/varidation", 100 *
                                      testcorrect/testtotal, x)
                    writer.add_scalar("loss/varidation", testloss.item(), x)


print('Finished Training')

# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)

# net = Net()
# net.load_state_dict(torch.load(PATH))

total = 0
correct = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' %
      (100 * correct / total))

class_correct = list(0.0 for i in range(10))
class_total = list(0.0 for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' %
          (classes[i], 100 * class_correct[i] / class_total[i]))
