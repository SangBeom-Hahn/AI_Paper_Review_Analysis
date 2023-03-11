# 나만의 데이터 셋 만들기 (이미지, 텍스트)


## 1. mnist처럼 ndarray로 만들고 fit      
<ul>
  <li><h3>이미지</h3></li>
</ul>

```python
# image_file_path = './notMNIST_small/*/*.png'

def myImage(image_file_path):
    paths = glob.glob(image_file_path)
    paths = np.random.permutation(paths)
    독립 = np.array([plt.imread(paths[i]) for i in range(len(paths))])

    # 이건 데이터를 폴더 A,B, C, D 이렇게 정리한다는 가정하에 클래스를 ABCD로 잡아주는 거임
    # 근데 데이터 다운 받으면 치마, 코드 이렇게 폴더로 나뉘어서 저장되어 있긴 함!!
    종속 = np.array([paths[i].split('/')[-2] for i in range(len(paths))]) 
    print(독립.shape, 종속.shape)
```

<ul>
  <li><h3>텍스트</h3></li>
</ul>

```python
# studying...
```






## 2. tensorflow dataset 객체 만들고 fit

#### Image
<ul>
  <li><h3>In Memory 방식</h3></li>
</ul>

```python
# mnist처럼 numpy로 만들어서 하기, numpy로 안 바꾸고 바로 jpg로 하기

(X_train, y_train), (X_test, y_test) = mnist.load_data()
train_ds = Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(1000)
train_ds = train_ds.batch(30)

test_ds = Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(30)

model.fit(train_ds, epochs=3)
result = model.evaluate(test_ds)
```

<ul>
  <li><h3>tf.data pix2pix 버전(from tensor slice 버전)</h3></li>
</ul>

```python
# 이 폴더 안에는 jpg 존재 
# pix2pix라서 이미지 pair 필요
real_path = "/content/trainB/"
input_path = "/content/trainA/"


def load_image_train(input_path, real_path):
  input_image, real_image = load(input_path, real_path)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image


def load(input_file, real_file): # real이 오른쪽
  real_image = tf.io.read_file(real_file)
  input_image = tf.io.read_file(input_file)
  real_image = tf.image.decode_jpeg(real_image)
  input_image = tf.image.decode_jpeg(input_image)

  real_image = tf.cast(real_image, tf.float32)
  input_image = tf.cast(input_image, tf.float32)
  return input_image, real_image 
  
  
# 데이터 셋 객체 생성
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
```

<ul>
  <li><h3>tf.data classification 버전(list_files 버전)</h3></li>
</ul>

```python
# 전체 ds 불러오기
list_ds = tf.data.Dataset.list_files(data_dir+'*/*', shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

# 전체 ds의 클래스 가져오기
class_names = np.array(sorted(os.listdir(data_dir)))

# 전체 ds를 split하기
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

# 이미지 경로를 받아서 클래스 반환
def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return tf.argmax(one_hot)

# 이미지 resize 하는 util 함수
def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

# 이미지 경로를 받아서 이미지와 라벨 반환
def process_path(file_path):
  label = get_label(file_path)
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label  
  
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# 성능을 위해 잘 섞고 배치 처리
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

# 학습
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)
```


<ul>
  <li><h3>image dataset from directory</h3></li>
</ul>


#### Text

<ul>
  <li><h3>감성분석 버전</h3></li>
</ul>

```python

# 이 안에는 리스트에 문장이 [ ['i', 'am'], ['he', 'is'] ] 이렇게 있다.
texts

# 라벨에는 [1.0.1.0]
labels

# 토큰화 > 인코딩
text_sequences = tokenizer.texts_to_sequences(texts)

# 패딩
text_sequences = tf.keras.preprocessing.sequence.pad_sequences(text_sequences)

# 라벨 원핫처리
cat_labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES) 

dataset = tf.data.Dataset.from_tensor_slices((text_sequences, cat_labels))
dataset = dataset.shuffle(10000)

# test. train 분리
test_size = num_records // 4
val_size = (num_records - test_size) // 10

test_dataset = dataset.take(test_size)
val_dataset = dataset.skip(test_size).take(val_size)
train_dataset = dataset.skip(test_size + val_size)

test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
```
