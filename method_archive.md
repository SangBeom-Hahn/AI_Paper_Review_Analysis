# PyTroch

<!-- 이거 쓰삼
<ul>
  <li><h3>데이터 로더 배치 크기만큼 그리드 출력</h3></li>
</ul>

```python

```
-->

## 모델 생성 및 학습 흐름

<ul>
  <li><h3>모델 직접 구현</h3></li>
</ul>

```python
# CNN과 트랜스포머 중 어떤 것을 쓸지 혼합할 지 고민해라
# cnn은 얼굴같은 가까운 거 잘잡고, 트랜은 멀고 데이터가 많아야만 잘 잡음
class MyModel(nn.Module): 
  def __init__(self):
    super().__init__()

    self.layer1 = nn.Sequential(
        # 입력 이미지 차원(흑백 : 1/ 컬럼 : 3)/ 커널 개수(conv와 batch가 개수가 같음)
        nn.Conv2d(3, 16, kernel_size = 3, stride = 2, padding = 0),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
    )

    self.layer2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 0),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
    )

    self.layer3 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 0),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
    )

    self.drop_out = nn.Dropout(0.5)

    self.fc1 = nn.Linear(3 * 3 * 64, 1000, bias = True)
    self.fc2 = nn.Linear(1000, 1)

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)

    out = out.view(out.size(0), -1)
    out = self.drop_out(out)
    out = self.fc1(out)
    out = self.fc2(out)
    return out
```

<ul>
  <li><h3>모델 직접 구현 list, for문 활용</h3></li>
</ul>

```python
class ConvolutionalNeuralNetworkClass(nn.Module):
    """
        Convolutional Neural Network (CNN) Class
    """
    def __init__(self,name='cnn',xdim=[1,28,28],
                 ksize=3,cdims=[32,64],hdims=[1024,128],ydim=10,
                 USE_BATCHNORM=False):
        super(ConvolutionalNeuralNetworkClass,self).__init__()
        self.name = name
        self.xdim = xdim
        self.ksize = ksize
        self.cdims = cdims
        self.hdims = hdims
        self.ydim = ydim
        self.USE_BATCHNORM = USE_BATCHNORM

        # Convolutional layers
        self.layers = [] # 모든 레이어를 리스트에 모으고 나중에 Sequential에 넣을 것이다.
        prev_cdim = self.xdim[0] # 최초 입력 이미지 차원 초기화
        for cdim in self.cdims: # cnn 출력 채널 수 [32,64]만큼 for문 수행
            self.layers.append(
                nn.Conv2d(
                    # FILL IN HERE
                    in_channels = prev_cdim,
                    out_channels = cdim,
                    kernel_size = self.ksize,
                    stride = (1,1),
                    padding = self.ksize//2
                ) # convlution
            )

            if self.USE_BATCHNORM:
                self.layers.append(nn.BatchNorm2d(cdim)) # batch-norm

            self.layers.append(nn.ReLU(True))  # activation # conv 레이어 밑에 하나하나 렐루 등을 붙임
            self.layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))) # max-pooling
            self.layers.append(nn.Dropout2d(p=0.5))  # dropout
            prev_cdim = cdim # for문 한번 끝나면 새로운 conv를 만들텐데 새로운 conv의 입력을 이전 conv의 출력 차원과 맞춤

        # Dense layers
        self.layers.append(nn.Flatten())
        prev_hdim = prev_cdim*(self.xdim[1]//(2**len(self.cdims)))*(self.xdim[2]//(2**len(self.cdims))) # conv의 출력과 dens의 입력을 맞춤
        for hdim in self.hdims:
            self.layers.append(nn.Linear(
                # FILL IN HERE
                in_features = prev_hdim,
                out_features = hdim,
                bias = True
                ))
            self.layers.append(nn.ReLU(True))  # activation
            prev_hdim = hdim
        # Final layer (without activation)
        self.layers.append(nn.Linear(prev_hdim,self.ydim,bias=True))


        # Concatenate all layers #
        self.net = nn.Sequential()
        for l_idx,layer in enumerate(self.layers):
            layer_name = "%s_%02d"%(type(layer).__name__.lower(),l_idx)
            self.net.add_module(layer_name,layer)
        self.init_param() # initialize parameters

    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d): # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d): # init BN
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        return self.net(x)

C = ConvolutionalNeuralNetworkClass(
    name='cnn',xdim=[1,28,28],ksize=3,cdims=[32,64],
    hdims=[32],ydim=10).to(device)
```

<ul>
  <li><h3>학습 프로세스</h3></li>
</ul>

```python
# 🚨 학습 속도 최적화 방법(GradScaler, zero_grad non_blocking) 들어 있음
# 간단한 건 ml basic 따라치기 1
train_dataset = (데이터 로더 활용 블로그, 레벨 1 커스텀 데이터 셋 참고)
test_dataset = (")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 학습 설정값을 지정합니다.
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.1

model.to(device)
criti = nn.BCEWithLogitsLoss()
# 옵티마이저를 어떤 파라미터에 대해 할 것인지
opt = optim.Adam(model.parameters(), lr = LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()

# 간단하게 : 아래 코드, ML basic 따라치기 2 참고
# 깊게[약간 추상화] : 파탬 참고/ 깊게[추상화] : 레벨 2참고
# 로스 : 레벨 1 참고
for epoch in range(1, EPOCHS+1):
  epoch_loss = 0
  epoch_acc = 0
  # 네트워크 모델을 train 모드로 두어 gradient을 계산하고, 여러 sub module (배치 정규화, 드롭아웃 등)이 train mode로 작동할 수 있도록 함
  model.train() 

  for X_batch, y_batch in dataloader:
    X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True).type(torch.cuda.FloatTensor)
  
    opt.zero_grad(set_to_none = True)
    with torch.cuda.amp.autocast():
      y_pred = model(X_batch)
      y_pred = torch.argmax(y_pred, dim=-1)
      loss = criti(y_pred, y_batch.unsqueeze(1)) # y를 1행 n열이 아닌 n행 1열로 만듬 (n, )가 (n, 1)로 됨

    acc = binary_acc(y_pred, y_batch.unsqueeze(1))
    epoch_loss += loss.item() # tensor([3]) 텐서에서 값(3)만 가져오기
    epoch_acc += acc.item()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': epoch_loss,
        }, f"{PATH}/checkpoint_model_{epoch}_{epoch_loss/len(dataloader)}_{epoch_acc/len(dataloader)}.pt")

    # 에폭별 손실과 정확도를 출력합니다.
    print(f'Epoch {epoch+0:03}: | Loss: {epoch_loss/len(dataloader):.5f} | Acc: {epoch_acc/len(dataloader):.3f}')
```

<ul>
  <li><h3>인퍼런스</h3></li>
</ul>

```python
# 파탬 참고/ 추상화 : 레벨 2참고
data_loader = 테스트 데이터 로더 로드
model = 모델 구격 로드
criterion = 로스 로드
metrics = 지표 로드
checkpoint = torch.load()
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict) # 모델에 가중치 삽입

# 인퍼런스(로스 log에 출력)
with torch.no_grad():
    for i, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        loss = criterion(output, target)
```

<ul>
  <li><h3>교차 검증 학습</h3></li>
</ul>

```python
# 파탬 참고
def kFoldTrain(config):
    # 5-fold Stratified KFold 5개의 fold를 형성하고 5번 Cross Validation을 진행합니다.
    n_splits = 2

    skf = StratifiedKFold(n_splits=n_splits)
    patience = 10
    dataset = config.init_obj("dataset", module_dataset)
    labels = dataset.y

    # Stratified KFold를 사용해 Train, Valid fold의 Index를 생성합니다.
    # labels 변수에 담긴 클래스를 기준으로 Stratify를 진행합니다.
    # 매 이터레이션 총 k개
    for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_list, labels)):
```

<ul>
  <li><h3>모델 저장 및 로드</h3></li>
</ul>


```python
PATH = "saved"

if(not os.path.exists(PATH)):
  os.makedirs(PATH)

# 1. 가중치만 저장
torch.save(model.state_dict(), os.path.join(PATH, "model.pt"))

# 로드
new_model = MyModel()
new_model.load_state_dict(torch.load(os.path.join(PATH, "model.pt")))

# 2. 전체 모델을 파일로 저장합니다.
torch.save(model, os.path.join(PATH, "model_pickle.pt"))

# 저장된 전체 모델을 불러옵니다.
model = torch.load(os.path.join(PATH, "model_pickle.pt"))

# 3. 가중치 파일 로드
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint["model_state_dict"])
opt.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint("epoch")

# 학습이 완료된 모델은 네트워크 모델을 eval 모드로 두어 여러 sub module들이 eval mode로 작동할 수 있게 함
model.eval()
```

<ul>
  <li><h3>전이 학습 구현</h3></li>
</ul>


```python
# 파이토치 심화 과제 1 참고
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = models.vgg16(pretrained = True).to(device)

# 모델의 모듈 바꿀 수 있음(안 바꾸는 게 좋음)
for name, layer in vgg.named_modules():
  print(name, layer)

# vgg.fc = torch.nn.Linear(1000, 1)
# vgg.classifier._modules['6'] = torch.nn.Linear(4096, 1)

import torch.nn as nn

class MyVgg(nn.Module):
  def __init__(self):
    super().__init__()
    self.vgg19 = models.vgg19(pretrained = True)
    self.linear_layers = nn.Linear(1000, 1, bias = True)
    nn.init.xavier_uniform_(self.linear_layers.weight)
    # bias를 초기화
    stdv = 1. / math.sqrt(self.linear_layers.weight.size(1))
    self.linear_layers.bias.data.uniform_(-stdv, stdv)

  def forward(self, x):
    x = self.vgg19(x)
    return self.linear_layers(x)

myVgg = MyVgg()

# freeze 전이 학습
for param in myVgg.parameters():
  param.requires_grad = False

for param in myVgg.linear_layers.parameters()  :
  param.requires_grad = True

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size = BATCH_SIZE,
                                         shuffle = True,
                                         num_workers = 8)

for epoch in range(1, EPOCHS+1):
  # ~~
```

## 유용한 함수
<ul>
  <li><h3>가중치 샘플러</h3></li>
</ul>

```python
# 데이터 셋 클래스 네부에 y를 Counter 돌려서 분포 확인하는 메서드
def target_class_distribution(self):
    target_dist = Counter(self.y)
    return target_dist, self.y

# 전체 데이터 셋
def getSampler(train_dataset):
    target_dist, y_train = train_dataset.target_class_distribution()
    sort_target_dist = sorted(target_dist.items(), key = lambda x:x[0])
    num_samples = len(train_dataset)
    class_weights = [num_samples / sort_target_dist[i][1] for i in range(len(sort_target_dist))]
    weights = [class_weights[y_train[i]] for i in range(num_samples)] # 해당 레이블마다의 가중치 비율
    return WeightedRandomSampler(torch.DoubleTensor(weights), int(15120))

# spilt 한 경우
train_set, val_set = 데이터 셋 분리 상태
_, multi_class_labels = train_dataset.target_class_distribution()
multi_class_labels = np.array(multi_class_labels)
# 이건 분리한 셋이 Subset 타입일 경우 원본에서의 인덱스 반환 가능하다는 예시
train_set_y = multi_class_labels[train_set.indices]
target_dist = Counter(train_set_y)

def getSampler(target_dist, train_set_y):
    sort_target_dist = sorted(target_dist.items(), key = lambda x:x[0])
    num_samples = len(train_set_y)
    class_weights = [num_samples / sort_target_dist[i][1] for i in range(len(sort_target_dist))]
    
    weights = [class_weights[train_set_y[i]] for i in range(int(num_samples))] #해당 레이블마다의 가중치 비율
    return WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
```

<ul>
  <li><h3>클래스 별 Grayscale 그래프</h3></li>
</ul>

```python
# 기초 플젝 따라치기 EDA 
plt.figure()
plt.subplot(111)

for class_id in num2class:
    img = np.array(Image.open(os.path.join(cfg.img_dir, img_id, class_id+ext)).convert('L'))
    histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))
    sns.lineplot(data=histogram)

plt.legend(num2class)
plt.title('Class Grayscale Histogram Plot', fontsize=15)
plt.show()
```

<ul>
  <li><h3>데이터 로더 배치 크기만큼 그리드 출력</h3></li>
</ul>

```python
# 무작위 학습 이미지 몇 개 선택
dataiter = iter(trainloader)
images, labels = next(iter(dataiter))

# 이미지 그리드 생성
img_grid = torchvision.utils.make_grid(images)

# 이미지 출력
matplotlib_imshow(img_grid, one_channel=True)
```

<ul>
  <li><h3>이미지가 흑백 1일 때 모델이 3이면 </h3></li>
</ul>

```python
# 데이터를 3으로 변환
common_transform = torchvision.transforms.Compose(
  [
    torchvision.transforms.Grayscale(num_output_channels=3), # grayscale의 1채널 영상을 3채널로 동일한 값으로 확장함
    torchvision.transforms.ToTensor() # PIL Image를 Tensor type로 변경함
  ]
)
```

```python
# 모델을 1로 변환
target_model.conv1 = torch.nn.Conv2d(FASHION_INPUT_NUM, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
```

<ul>
  <li><h3>토치 서머리</h3></li>
</ul>

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 사전 학습된 VGG16 모델을 불러옵니다.
imagenet_resnet18 = torchvision.models.resnet18(pretrained=True).to(device)
# 모델의 구조 및 파라미터 요약 출력
summary(imagenet_resnet18, (3, 224, 224))
```

<ul>
  <li><h3>파이썬 Enum</h3></li>
</ul>

```python
class Age(int, Enum):
    """Enum 클래스 -> enum만 모아두자"""
    
    YOUNG = 0
    MIDDLE = 1
    OLD = 2
    
    @classmethod
    def _label_ages(cls, ages):
        if(ages < 30):
            return cls.YOUNG
        elif(ages < 60):
            return cls.MIDDLE
        else:
            return cls.OLD
```

<ul>
  <li><h3>데이터 셋 라벨 더미화</h3></li>
</ul>

```python
# dic로 더미화
def _convert_class_to_numeric(self, class_array):
    class_to_number = {class_value : idx for idx, class_value in enumerate(np.unique(class_array))}
    return np.vectorize(class_to_number.get)(class_array)

# if문으로 더미화
def _convert_age_to_dummy(self, ages):
    return np.vectorize(Age._label_ages)(ages)
```

<ul>
  <li><h3>json kfold split</h3></li>
</ul>

```python
annotation = "./cleansing.json"
output_dir = './json_folder'

with open(annotation) as f:
    data = json.load(f)
var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
X = np.ones((len(data['annotations']), 1))
y = np.array([v[1] for v in var])
groups = np.array([v[0] for v in var])
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=207)

for i,(train_idx, val_idx) in enumerate(cv.split(X, y, groups), start = 1) :
    train_data_img_ids = set([data['annotations'][idx]['image_id'] for idx in train_idx])
    train_data_imgs = [img for img in data['images'] for img_id in train_data_img_ids if img['id'] == img_id]
    train_data = {'images' : train_data_imgs,
                 'categories' : data['categories'],
                 'annotations': [data['annotations'][idx] for idx in train_idx],}

    val_data_img_ids = set([data['annotations'][idx]['image_id'] for idx in val_idx])
    val_data_imgs = [img for img in data['images'] for img_id in val_data_img_ids if img['id'] == img_id]
    val_data = {'images' : val_data_imgs,
                'categories' : data['categories'],
               'annotations': [data['annotations'][idx] for idx in val_idx],}
    
    train_filename = os.path.join(output_dir, f'train_fold_{i}.json')
    with open(train_filename, 'w') as f:
        json.dump(train_data, f, indent=4)
        
    val_filename = os.path.join(output_dir, f'val_fold_{i}.json')
    with open(val_filename, 'w') as f:
        json.dump(val_data, f, indent=4)
```

## 데이터 파이프 라인

<ul>
  <li><h3>모델에 random x 딱 하나 테스트로 넣기</h3></li>
</ul>

```python
# mnist 모델에 넣을 784 차원 데이터 1개 준비
x_numpy = np.random.rand(1,784)

# 토치는 모델을 이용하려면 무조건 토치 타입으로 바꿔야 하고 cpu든 gpu등 장치를 지정해야 한다.
# to. 안하면 RuntimeError: Expected all tensors to be on the same device, 발생
# 토치에서 데이터 x, y에는 device를 무조건 연결해줘야 한다.
x_torch = torch.from_numpy(x_numpy).float().to(device)

y_torch = M(x_torch) # 10 차원 데이터 반환

''' 거짓
# 텐서를 넘파이로 바꾸기
# 넘파이는 또 cpu를 먹어서 
# TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() 발생
y_torch.numpy()

# req_grad를 없애기 위해 detach로 없애라고 함
# detach() method는 gradient의 전파를 멈추는 역할을 한다. 이런 것이 다 추론 시 메모리를 위한 최적화
# RuntimeError: Use tensor.detach().numpy() instead. 발생
y_torch.cpu().numpy()
'''

# 진실 코드
y_numpy = y_torch.detach().cpu().numpy()
```

<ul>
  <li><h3>모델에 데이터 배치 x 딱 하나 Dataset에서 꺼내서 테스트로 넣기</h3></li>
</ul>

```python
qd_train_dataset = QuickDrawDataset(train_data, train_label, transform)
x_data = next(iter(qd_train_dataset))[0].squeeze(0) # 배치 1 만들어줘야 함
y_data = next(iter(qd_train_dataset))[1]

c = 모델(in_channels = 3, out_channels = 64)
y_pred = c(x_data)
y_pred
```


## 공통

<ul>
  <li><h3>모델 가중치 저장</h3></li>
</ul>

```python
def save_weight_to_json(model):
  cur_dir = os.getcwd() # 현재 작업 디렉터리
  ckpt_dir = "checkpoints" # weight를 저장할 디렉토리
  file_name = "weights.ckpt" # 저장 파일명
  dir = os.path.join(cur_dir, ckpt_dir) 
  os.makedirs(dir, exist_ok = True) # dir를 만듬

  file_path = os.path.join(dir, file_name) # dir 경로 + 파일 이름의 파일 경로를 join함
  model.save_weights(file_path)

  model_json = model.to_json() # 모델 구조도 저장하여 model.json으로 저장
  with open("model.json", "w") as json_file: 
    json_file.write(model_json)
```



# Tensorflow

<ul>
  <li><h3>모델 가중치 로드</h3></li>
</ul>

```python
from keras.models import model_from_json 

def load_weight_to_json():
  json_file = open("model.json", "r")
  loaded_model_json = json_file.read() 
  json_file.close()

  loaded_model = model_from_json(loaded_model_json)
  loaded_model.load_weights(file_path) # file_path = os.path.join(dir, file_name)
```


<ul>
  <li><h3>훈련 중단 후 재개시 가중치 로드</h3></li>
</ul>

```python
LOAD_FROM_CK_PT = False
if LOAD_FROM_CK_PT:
    num = '00210'
    gen_model = load_model(f'output_b4_pts250/models/{num}_gen_model.h5')
    d_model = load_model(f'output_b4_pts250/models/{num}_d_model.h5')
else:
    gen_model = get_generator_model()
    d_model = get_discriminator_model()

gan_model = get_gan_model(gen_model, d_model, L1_loss_lambda=100)
```


<ul>
  <li><h3>손실 그래프 생성</h3></li>   
</ul>

```python
def plotLoss(G_loss, D_loss, epoch):
  cur_dir = os.getcwd()
  loss_dir = "loss_graph"
  file_name = 'gan_loss_epoch_%d.png' % epoch
  dir = os.path.join(cur_dir, loss_dir) 
  os.makedirs(dir, exist_ok = True)

  file_path = os.path.join(dir, file_name)

  plt.figure(figsize=(10, 8))
  plt.plot(D_loss, label='Discriminitive loss')
  plt.plot(G_loss, label='Generative loss')
  plt.xlabel('BatchCount')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig(file_path)
```


<ul>
  <li><h3>모델 저장</h3></li>
</ul>

```python
def save_model(model, model_path='saved_model/model.h5'):
  print('\nsave model : \"{}\"'.format(model_path))
  model.save(model_path)
```


<ul>
  <li><h3>모델 로드</h3></li>
</ul>

```python
def load_model(model, model_path='saved_model/model.h5'):
  print('\nload model : \"{}\"'.format(model_path))
  model = tf.keras.models.load_model(model_path)
```

<ul>
  <li><h3>훈련 정확도, 손실 </h3></li>
</ul>

```python
import matplotlib.pyplot as plt

def show_history(history): # history에 val를 뽑으려면 fit할 때 validation_data를 써야한다.
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='valid')
    plt.legend()
    
def show_history(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.legend()
```

## cv -> 참고 : 이미지는 머러(3, 4), adv(gan), 랜드마크 잇기

<ul>
  <li><h3>GAN 생성 이미지 저장</h3></li>
</ul>

```python
def sample_images(epoch, latent_dim = 128):
  cur_dir = os.getcwd()
  image_dir = "images"
  file_name = '%d.png' % epoch
  dir = os.path.join(cur_dir, image_dir) 
  os.makedirs(dir, exist_ok = True)

  file_path = os.path.join(dir, file_name)


  r, c = 5, 5
  noise = np.random.normal(0, 1, (r * c, latent_dim))
  gen_imgs = generator.predict(noise)

  # Rescale images 0 - 1
  gen_imgs = 0.5 * gen_imgs + 0.5

  fig, axs = plt.subplots(r, c)
  cnt = 0
  for i in range(r):
      for j in range(c):
          axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
          axs[i,j].axis('off')
          cnt += 1
  fig.savefig(file_path)
  plt.close()
```

<ul>
  <li><h3>이미지 데이터 끌어모아서 한 곳에 저장</h3></li>
</ul>

```python
def create_target_images():
    pathname = f'{config.ZAPPOS_DATASET_SNEAKERS_DIR}/*/*.jpg' 
    print(pathname)
    print(glob.glob(pathname)) 
  
    for filepath in glob.glob(pathname):
        filename = os.path.basename(filepath)
        img_target = load_img(filepath, target_size=(config.IMG_HEIGHT, config.IMG_WIDTH))
        img_target = np.array(img_target)  
        img_target_filepath = os.path.join(config.TRAINING_TARGET_DIR, filename) 
        save_img(img_target_filepath, img_target) 
```



<ul>
  <li><h3>한 폴더에 모여있는 이미지 데이터 다른 폴더로 옮기기</h3></li>
</ul>


```python
def create_source_imgs(target_dir, source_dir):
    pathname = f'{target_dir}/*.jpg' # data/training/target
    print(pathname)
    for filepath in glob.glob(pathname):
        img_target = load_img(filepath, target_size=(config.IMG_HEIGHT, config.IMG_WIDTH))
        img_target = np.array(img_target)
        img_source = detect_edges(img_target)

        filename = os.path.basename(filepath)
        img_source_filepath = os.path.join(source_dir, filename)
        save_img(img_source_filepath, img_source)
```

<ul>
  <li><h3>폴더 내 모든 이미지 edge detection </h3></li>
</ul>

```python
import cv2
from keras_preprocessing.image import load_img, save_img

def detect_edges(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.bilateralFilter(img_gray, 5, 50, 50)
    img_gray_edges = cv2.Canny(img_gray, 45, 100)
    img_gray_edges = cv2.bitwise_not(img_gray_edges) # invert black/white
    img_edges = cv2.cvtColor(img_gray_edges, cv2.COLOR_GRAY2RGB)
    
    return img_edges

def create_edge_imgs(target_dir, source_dir):
    pathname = f'{target_dir}/*.jpg' # target_dir에 폴더명
    for filepath in glob.glob(pathname):
        img_target = load_img(filepath, target_size=(256, 256))
        img_target = np.array(img_target)
        img_source = detect_edges(img_target) # 아 소스 이미지는 엣지 이미지구나

        filename = os.path.basename(filepath)
        img_source_filepath = os.path.join(source_dir, filename)
        save_img(img_source_filepath, img_source) 
        
# 사용법 : 원본 이미지 폴더 경로, 엣지 이미지를 저장할 폴더 경로
# create_edge_imgs("/content/trainB", "/content/trainA")
```

<ul>
  <li><h3>이미지(jpg)든 뭐든 csv로 만들기</h3></li>
</ul>

```python
import os, natsort, csv, re

# file_path = 'photo/'

def toCSV(file_path):
    file_lists = os.listdir(file_path)
    file_lists = natsort.natsorted(file_lists)

    f = open('train.csv', 'w', encoding='utf-8') #valid.csv, test.csv
    wr = csv.writer(f)
    wr.writerow(["Img_name", "Class"])
    for file_name in file_lists:
        print(file_name)
        wr.writerow([file_name, re.sub('-\d*[.]\w{3}', '', file_name)])
    f.close()
```

<ul>
  <li><h3>이미지 자르기</h3></li>
</ul>

```python
import cv2

def seperateImg(img): # img = '/content/drive/MyDrive/Colab Notebooks/KakaoTalk_20220916_023415527.jpg'
    src = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    dst1 = src[:, 0:256].copy()   # 선만 있는 거
    dst2 = src[:, 257:512].copy()  # 색 있는 거
    cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/color.jpg',dst1)  # 저장
    cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/color2.jpg',dst2)  # 저장
```

<ul>
  <li><h3>크롤링 이미지 mnist 형태로</h3></li>
</ul>

```python
# image_file_path = './notMNIST_small/*/*.png'

def myImage(image_file_path):
    paths = glob.glob(image_file_path)
    paths = np.random.permutation(paths)
    독립 = np.array([plt.imread(paths[i]) for i in range(len(paths))])
    종속 = np.array([paths[i].split('/')[-2] for i in range(len(paths))]) # A/test.jpg -> 클래스 A, B/test1.jpg -> 클래스 
    print(독립.shape, 종속.shape)
```

<ul>
  <li><h3>넘파이 이미지 한 번에 resize하고 jpeg로 저장까지</h3></li>
</ul>

```python
img_array.shape # (60000, 28, 28, 1)

def image_resize(img_array):
    for i in range(len(img_array)):
      img_resize = cv2.resize(img_array[i], dsize = (256, 256))
      img_resize = Image.fromarray(img_resize)
      img_resize = img_resize.convert('RGB')
      img_resize.save(f"{i}.jpeg")
```

<ul>
  <li><h3>train_X 시각화</h3></li>
</ul>

```python
import matplotlib.pyplot as plt

def visualizeTrainX():
    ncols = 10 # 조정

    figure, axs = plt.subplots(figsize = (10, 5), nrows=1, ncols = ncols)

    for i in range(ncols):
      axs[i].imshow(train_images[i]) # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```


## nlp -> 참고 : 텍스트는 머러(7, 8), 메타코드(영화리뷰), base(Q/A), 논문 분석 레포만 보면 된다,

<ul>
  <li><h3>max_len 구하기</h3></li>
</ul>

```python
def get_max_len(sentences):
    seq_lengths = np.array([len(s.split()) for s in sentences])
    print([(p, np.percentile(seq_lengths, p)) for p in [75, 80, 90, 95, 99, 100]])
```
