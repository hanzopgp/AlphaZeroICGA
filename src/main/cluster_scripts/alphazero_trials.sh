export PATH=$PATH:/data/vittaut/apache-ant-1.10.12/bin 
conda init bash
conda activate alphazero

#!/bin/bash

command=""
for i in  {0. .$(($1))}
do
	command+="ant run_trials -Dforce_vanilla=False & "
done
command+=" wait"
echo $command
eval $command