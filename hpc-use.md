## NCSU HPC
```bash
source ~/.bashrc  
conda activate evo2  
export CUDA_HOME=$CONDA_PREFIX  
export HF_HOME=/usr/local/usrapps/statgen/jjiang26/.cache/huggingface  

cd /share/statgen/jjiang26/  

bsub -Is -n 4 -q gpu -R "select[h100 || l40 || l40s]" -gpu "num=1:mode=shared:mps=no" -W 30  bash  
conda activate evo2  
```

### For Others
```bash
/usr/local/usrapps/statgen/anaconda3/bin/conda init
source ~/.bashrc
conda activate evo2
export HF_HOME=/usr/local/usrapps/statgen/jjiang26/.cache/huggingface/
export CUDA_HOME=$CONDA_PREFIX

```
