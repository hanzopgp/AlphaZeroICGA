export PATH=$PATH:/data/vittaut/apache-ant-1.10.12/bin 
conda init bash
conda activate alphazero

#!/bin/bash

command=""
echo "ok"
echo $1
echo "ok"
for i in $(seq 0 1 $(($1)))
do
	echo "$i"
	command+="ant run_trials -Dforce_vanilla=False & "
done
command+=" wait"
eval $command