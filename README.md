# CEM-RL
Pytorch implementation of CEM-RL: https://arxiv.org/pdf/1810.01222.pdf

### Using gymnasium to run v4 env task

### Create env
```
conda env create -n gym -f yaml/gym.yaml
```

To reproduce the results of the paper:

Without importance mixing:
```console
python es_grad.py --env ENV_NAME --use_td3 --output OUTPUT_FOLDER
```

With importance mixing:
```console
python es_grad_im.py --env ENV_NAME --use_td3 --output OUTPUT_FOLDER
```

TD3:
```console
python distributed.py --env ENV_NAME --use_td3 --output OUTPUT_FOLDER
```