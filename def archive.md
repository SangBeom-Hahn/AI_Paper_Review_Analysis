<ul>
  <li><h3>모델 가중치 저장</h3></li>
</ul>

```python
def save_weight_to_json(model):
  cur_dir = os.getcwd() # 현재 작업 디렉
  ckpt_dir = "checkpoints" # weight를 저장할 디렉토리
  file_name = "weights.ckpt" # 저장 파일명
  dir = os.path.join(cur_dir, ckpt_dir) 
  os.makedirs(dir, exist_ok = True) # dir이라는 폴더를 만듬

  file_path = os.path.join(dir, file_name) # dir 경로 + 파일 이름의 파일 경로를 join함
  model.save_weights(file_path)

  model_json = model.to_json() # 모델 구조도 저장하여 model.json으로 저장
  with open("model.json", "w") as json_file: 
    json_file.write(model_json)
```


<ul>
  <li><h3>모델 가중치 로드</h3></li>
</ul>

```python
def load_weight_to_json():
  json_file = open("model.json", "r")
  loaded_model_json = json_file.read() 
  json_file.close()

  loaded_model = model_from_json(loaded_model_json)
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
  <li><h3>크롤링 이미지 mnist 형태로 </h3></li>
</ul>

```python
# image_file_path = './notMNIST_small/*/*.png'

def myImage(image_file_path):
    paths = glob.glob(image_file_path)
    paths = np.random.permutation(paths)
    독립 = np.array([plt.imread(paths[i]) for i in range(len(paths))])
    종속 = np.array([paths[i].split('/')[-2] for i in range(len(paths))])
    print(독립.shape, 종속.shape)
```

<ul>
  <li><h3>train_X 시각화 </h3></li>
</ul>

```python
import matplotlib.pyplot as plt

def visualizeTrainX():
    ncols = 10 # 조정

    figure, axs = plt.subplots(figsize = (10, 5), nrows=1, ncols = ncols)

    for i in range(ncols):
      axs[i].imshow(train_images[i]) # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

<ul>
  <li><h3>훈련 정확도, 손실 </h3></li>
</ul>

```python
import matplotlib.pyplot as plt

def show_history(history):
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='valid')
    plt.legend()
    
def show_history(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.legend()
```
