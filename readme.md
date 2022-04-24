### Build environment

```
conda env create -f environment.yml
```

### Check all avaliable servers

```
python run.py server
```

### Online inference

```
python run.py run \
--server ws1.joshuachou.ink \
--temp_dir temp \
--dict_dcm 'Example/data_1_2020-05-01/raw_data' \
--output_dir 'Example/data_1_2020-05-01/prediction' \
--file_name 'A01' \
--pub_key 'Example/public.pem' \
--pri_key 'Example/private.pem'
```

