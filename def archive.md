```python
def save_weight_to_json(model):
  cur_dir = os.getcwd() # 현재 작업 디렉
  ckpt_dir = "checkpoints" # weight를 저장할 디렉토리
  file_name = "gan_weights.ckpt" # 저장 파일명
  dir = os.path.join(cur_dir, ckpt_dir) 
  os.makedirs(dir, exist_ok = True) # dir이라는 폴더를 만듬

  file_path = os.path.join(dir, file_name) #dir 경로 + 파일 이름의 파일 경로를 join함
  model.save_weights(file_path)

  model_json = model.to_json() # 모델 구조도 저장하여 model.json으로 저장
  with open("model.json", "w") as json_file : 
    json_file.write(model_json)
```
