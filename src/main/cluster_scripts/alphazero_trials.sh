export PATH=$PATH:/data/vittaut/apache-ant-1.10.12/bin 
conda init bash
conda activate alphazero
ant run_trials -Dforce_vanilla=False & ant run_trials -Dforce_vanilla=False & wait