# blenderproc-test
## Install
```
$ python3 -m venv .venv
$ source .venv/bin/activate.fish
(.venv) $ pip install -U pip wheel
(.venv) $ pip install blenderproc
(.venv) $ pip install -r requirements.txt
```

## Render
```
blenderproc run put_on_floor.py
```


## データ拡張
```
$ source .venv/bin/activate.fish

(.venv) $ PYTHONPATH=. python src/tools/augmentation_data.py aug source_dir target_dir
```

## データのビジュアライズ（チェック）
```
$ source .venv/bin/activate.fish

(.venv) $ PYTHONPATH=. python src/tools/augmentation_data.py vis base_dir
```
