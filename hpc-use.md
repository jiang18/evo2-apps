## NCSU HPC
```bash
source ~/.bashrc  
conda activate evo2  
export CUDA_HOME=$CONDA_PREFIX  
export HF_HOME=/usr/local/usrapps/statgen/jjiang26/.cache/huggingface  

cd /share/statgen/jjiang26/  

bsub -Is -n 4 -q gpu -R "select[h100 || l40 || l40s]" -gpu "num=1:mode=shared:mps=no" -W 30  bash
bsub -Is -n 4 -q gpu -R "select[h100]" -gpu "num=1:mps=no:mode=exclusive_process" -W 120  bash
# mps should be set to no.  
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
