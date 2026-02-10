# PyTorch and Computer Vision

Some libraries to be aware of:
- `torchvision` central library for datasets, model architecture, and image transformations often used for computer vision problems
- `torchvision.datasets` bunch of datasets for computer vision
- `torchvision.models` bunch of models for computer vision
- `torchvision.transforms` common image transformations (turn into numbers, processed, augmented)
```python
import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")
```

**Output:**
```
PyTorch version: 2.9.0+cu126
torchvision version: 0.24.0+cu126
```

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setup training dataset with torchvision
train_data = datasets.FashionMNIST(
  root="data",
  train=True,
  download=True,
  transform=ToTensor(), # can transform the images from their PIL format to Torch tensors
  target_transform=None # cant transform the labels
)

# Setup testing data
test_data = datasets.FashionMNIST(
  root="data",
  train=False,
  download=True,
  transform=ToTensor()
)
```

```python
image, label = train_data[0]
print(image.shape, label)
print(len(train_data.data), len(train_data.targets), len(test_data.data), len(test_data.targets))
print(len(train_data.classes))
```

**Output:**
```
torch.Size([1, 28, 28]) 9
60000 60000 10000 10000
10
```

```python
import matplotlib.pyplot as plt

image, label = train_data[0]
# Plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(train_data.classes[label])
    plt.axis(False);
```

**Output:**
```
<Figure size 900x900 with 16 Axes>
```

![[Pasted image 20251030152624.png]]

# Conv2d Documentation
Performs a convolution on the data. Can check out https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
- in_channels refers to the number of channels the input has (in RGB , that 3)
- out_channels refers to the number channels we want to end up with
- kernel_size is the height/width (can be a tuple) of the kernel to run convolutions with
- stride is the horizontal/vertical "jump" between areas in the image to run convolutions on
- padding is the amount of padding around an image. useful to catch features in the edges and corners of the image
- dilation is the spacing of the convolution itself

**The number of kernels in a single conv2d layer is the number of input_channels * output_channels** because for every output channel, we need one kernal per input channel.
```python
# Prepare Data
from torch.utils.data import DataLoader
import torchmetrics, mlxtend

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

BATCH_SIZE = 32
LEARNING_RATE = 0.2
EPOCHS = 100

train_dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# Model
class FashionMNISTClassifier(nn.Module):
  def __init__(self, device):
    super().__init__()

    self.conv0 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=2, padding=1, device=device)
    self.conv1 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=2, padding=1, device=device)
    self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=2, padding=1, device=device)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax()

  def forward(self, x):
    x = self.relu(self.conv0(x))
    x = self.relu(self.conv1(x))
    x = self.avgpool(self.conv2(x))
    return x.squeeze()

model = FashionMNISTClassifier(device)
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

def compute_pred_accuracy(y_pred, y):
  y_softmax = torch.softmax(y_pred, dim=1)
  return sum(torch.eq(y_softmax.argmax(dim=1), y)) / len(y)

def compute_accuracy(model, dataloader):
  return sum(compute_pred_accuracy(model(x_b.to(device)), y_b.to(device)) for x_b, y_b in dataloader) / len(dataloader)

def train(model, train_dl, opt, loss_fn):
  model.train()
  losses = []
  training_acc = []
  for x_b, y_b in train_dl:
    # load batch into GPU
    x_b = x_b.to(device)
    y_b = y_b.to(device)

    # Forward
    pred = model(x_b) # get prediction logits (no softmax + log + nll)
    loss = loss_fn(pred, y_b)

    # Log accuracies
    losses.append(loss)
    training_acc.append(compute_pred_accuracy(pred, y_b))

    opt.zero_grad()
    loss.backward()
    opt.step()

  print(f"Training loss {sum(losses) / len(losses)} | TRAIN ACC {sum(training_acc) / len(training_acc)}")

def eval(model, test_dl):
  model.eval()
  with torch.inference_mode():
    test_acc = compute_accuracy(model, test_dl)

    print(f"====== TEST ACC {test_acc} ======")
 

# training loop
for e in range(EPOCHS):
  train(model, train_dl, opt, loss_fn)

  # Only evaluate after 10 epochs
  if e % 10 == 0:
    eval(model, test_dl)

```

**Output:**
```
Training loss 1.081870198249817 | TRAIN ACC 0.6086833477020264
====== TEST ACC 0.7177516222000122 ======
Training loss 0.695329487323761 | TRAIN ACC 0.7548333406448364
Training loss 0.6107842326164246 | TRAIN ACC 0.7808499932289124
Training loss 0.5636925101280212 | TRAIN ACC 0.7980833053588867
Training loss 0.5310357213020325 | TRAIN ACC 0.8095166683197021
Training loss 0.5095382332801819 | TRAIN ACC 0.8185333609580994
Training loss 0.4920312166213989 | TRAIN ACC 0.8248166441917419
Training loss 0.47657620906829834 | TRAIN ACC 0.8297833204269409
Training loss 0.4631468951702118 | TRAIN ACC 0.8347833156585693
Training loss 0.4530208706855774 | TRAIN ACC 0.8394833207130432
Training loss 0.4471907615661621 | TRAIN ACC 0.8399999737739563
====== TEST ACC 0.8336661458015442 ======
Training loss 0.44009098410606384 | TRAIN ACC 0.8425999879837036
Training loss 0.434322714805603 | TRAIN ACC 0.8448166847229004
Training loss 0.43351295590400696 | TRAIN ACC 0.8457666635513306
Training loss 0.42730405926704407 | TRAIN ACC 0.8481166958808899
Training loss 0.4250723123550415 | TRAIN ACC 0.8486499786376953
Training loss 0.4233935475349426 | TRAIN ACC 0.8488666415214539
Training loss 0.4202239215373993 | TRAIN ACC 0.8490999937057495
Training loss 0.4178701937198639 | TRAIN ACC 0.8524333238601685
Training loss 0.41460442543029785 | TRAIN ACC 0.8515666723251343
Training loss 0.41286197304725647 | TRAIN ACC 0.853950023651123
====== TEST ACC 0.8458466529846191 ======
Training loss 0.411700963973999 | TRAIN ACC 0.8537166714668274
Training loss 0.4102802872657776 | TRAIN ACC 0.8554666638374329
Training loss 0.4081258773803711 | TRAIN ACC 0.8546833395957947
Training loss 0.40706920623779297 | TRAIN ACC 0.8548833131790161
Training loss 0.405464231967926 | TRAIN ACC 0.8557000160217285
Training loss 0.4034602642059326 | TRAIN ACC 0.8566166758537292
Training loss 0.403064489364624 | TRAIN ACC 0.8551333546638489
Training loss 0.4011331796646118 | TRAIN ACC 0.8557166457176208
Training loss 0.40126827359199524 | TRAIN ACC 0.857366681098938
Training loss 0.3994111120700836 | TRAIN ACC 0.8582333326339722
====== TEST ACC 0.8529353141784668 ======
Training loss 0.39841407537460327 | TRAIN ACC 0.8587499856948853
Training loss 0.39885827898979187 | TRAIN ACC 0.8589000105857849
Training loss 0.3965250849723816 | TRAIN ACC 0.859083354473114
Training loss 0.3982420563697815 | TRAIN ACC 0.8568500280380249
Training loss 0.3948778510093689 | TRAIN ACC 0.8590499758720398
Training loss 0.3953261077404022 | TRAIN ACC 0.8588833212852478
Training loss 0.3943737745285034 | TRAIN ACC 0.8593500256538391
Training loss 0.39554449915885925 | TRAIN ACC 0.859333336353302
Training loss 0.3919506072998047 | TRAIN ACC 0.8591499924659729
Training loss 0.3925478756427765 | TRAIN ACC 0.8606333136558533
====== TEST ACC 0.8492411971092224 ======
Training loss 0.3924988806247711 | TRAIN ACC 0.8609166741371155
Training loss 0.3918140232563019 | TRAIN ACC 0.8603000044822693
Training loss 0.39049583673477173 | TRAIN ACC 0.861466646194458
Training loss 0.39042791724205017 | TRAIN ACC 0.8614166378974915
Training loss 0.39001408219337463 | TRAIN ACC 0.8612333536148071
Training loss 0.3873157799243927 | TRAIN ACC 0.8630666732788086
Training loss 0.3883238136768341 | TRAIN ACC 0.8607833385467529
Training loss 0.38839957118034363 | TRAIN ACC 0.8617333173751831
Training loss 0.38852041959762573 | TRAIN ACC 0.8617333173751831
Training loss 0.38731157779693604 | TRAIN ACC 0.8622000217437744
====== TEST ACC 0.8392571806907654 ======
Training loss 0.3878650963306427 | TRAIN ACC 0.8618166446685791
Training loss 0.3870152533054352 | TRAIN ACC 0.8619499802589417
Training loss 0.3874465227127075 | TRAIN ACC 0.8628666400909424
Training loss 0.38585853576660156 | TRAIN ACC 0.8618166446685791
Training loss 0.3840395212173462 | TRAIN ACC 0.8636166453361511
Training loss 0.3863951861858368 | TRAIN ACC 0.8621333241462708
Training loss 0.38429534435272217 | TRAIN ACC 0.8621833324432373
Training loss 0.3837733864784241 | TRAIN ACC 0.8625166416168213
Training loss 0.38455045223236084 | TRAIN ACC 0.8627499938011169
Training loss 0.38421109318733215 | TRAIN ACC 0.8626166582107544
====== TEST ACC 0.8521365523338318 ======
Training loss 0.38454991579055786 | TRAIN ACC 0.8625333309173584
Training loss 0.3840777575969696 | TRAIN ACC 0.8635833263397217
Training loss 0.38363704085350037 | TRAIN ACC 0.862766683101654
Training loss 0.38337141275405884 | TRAIN ACC 0.8638833165168762
Training loss 0.3833909034729004 | TRAIN ACC 0.8634999990463257
Training loss 0.38224470615386963 | TRAIN ACC 0.8641666769981384
Training loss 0.3829042911529541 | TRAIN ACC 0.8633833527565002
Training loss 0.3835487365722656 | TRAIN ACC 0.8628333210945129
Training loss 0.3816915452480316 | TRAIN ACC 0.8644833564758301
Training loss 0.3817364275455475 | TRAIN ACC 0.8638499975204468
====== TEST ACC 0.8504393100738525 ======
Training loss 0.38077405095100403 | TRAIN ACC 0.8648999929428101
Training loss 0.38126617670059204 | TRAIN ACC 0.8648166656494141
Training loss 0.37974971532821655 | TRAIN ACC 0.8647666573524475
Training loss 0.38004031777381897 | TRAIN ACC 0.8650500178337097
Training loss 0.38153892755508423 | TRAIN ACC 0.8627166748046875
Training loss 0.38082727789878845 | TRAIN ACC 0.8640000224113464
Training loss 0.37907955050468445 | TRAIN ACC 0.8656499981880188
Training loss 0.3805506229400635 | TRAIN ACC 0.8649500012397766
Training loss 0.3795810639858246 | TRAIN ACC 0.8653166890144348
Training loss 0.3802793622016907 | TRAIN ACC 0.8648999929428101
====== TEST ACC 0.8580271601676941 ======
Training loss 0.37931451201438904 | TRAIN ACC 0.8648166656494141
Training loss 0.3807264268398285 | TRAIN ACC 0.8635333180427551
Training loss 0.3797339200973511 | TRAIN ACC 0.8635833263397217
Training loss 0.378546804189682 | TRAIN ACC 0.8651999831199646
Training loss 0.3781156837940216 | TRAIN ACC 0.8642666935920715
Training loss 0.37969329953193665 | TRAIN ACC 0.864633321762085
Training loss 0.3787648677825928 | TRAIN ACC 0.8643666505813599
Training loss 0.37790387868881226 | TRAIN ACC 0.8649666905403137
Training loss 0.37813252210617065 | TRAIN ACC 0.8657666444778442
Training loss 0.37809523940086365 | TRAIN ACC 0.8655999898910522
====== TEST ACC 0.857627809047699 ======
Training loss 0.3777090907096863 | TRAIN ACC 0.8641166687011719
Training loss 0.3774780333042145 | TRAIN ACC 0.8649666905403137
Training loss 0.3776569664478302 | TRAIN ACC 0.8656499981880188
Training loss 0.37669047713279724 | TRAIN ACC 0.8651833534240723
Training loss 0.37804093956947327 | TRAIN ACC 0.8640499711036682
Training loss 0.37611308693885803 | TRAIN ACC 0.8665666580200195
Training loss 0.37766605615615845 | TRAIN ACC 0.8658666610717773
Training loss 0.37646251916885376 | TRAIN ACC 0.8651999831199646
Training loss 0.3772161900997162 | TRAIN ACC 0.8646833300590515
```

