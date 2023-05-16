# SuperICL
Code for "Small Models are Valuable Plug-ins for Large Language Models".

![supericl_workflow](https://github.com/JetRunner/SuperICL/assets/22514219/4567f26b-2c21-4f00-bfba-92622dfab47b)

## How to Run Code
### Setup
#### Install Requirements
```bash
pip install -r requirements.txt
```

#### Add OpenAI API Key
```bash
cp api_config_example.py api_config.py
vi api_config.py
```

### GLUE
```bash
python run_glue.py \
--model_path roberta-large-mnli \
--model_name RoBERTa-Large \
--dataset mnli-m \
--explanation  # Include this to enable explanation for overrides
```
For all supported tasks, see [here](https://github.com/JetRunner/SuperICL/blob/main/run_glue.py#L34).

For the complete set of parameters, see the code [here](https://github.com/JetRunner/SuperICL/blob/main/run_glue.py#L23).

### XNLI
```bash
python run_xnli.py \
--model_path /path/to/model \
--model_name XLM-V \
--lang en,ar,bg,de,el,es,fr,hi,ru,sw,th,tr,ur,vi,zh 
```
For the complete set of parameters, see the code [here](https://github.com/JetRunner/SuperICL/blob/main/run_xnli.py#L20).

## Citation
```bibtex
@article{xu2023small,
  title={Small Models are Valuable Plug-ins for Large Language Models},
  author={Canwen Xu and Yichong Xu and Shuohang Wang and Yang Liu and Chenguang Zhu and Julian McAuley},
  journal={arXiv preprint arXiv:2305.08848},
  year={2023}
}
```
