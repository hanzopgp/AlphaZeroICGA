export PATH=$PATH:/data/vittaut/apache-ant-1.10.12/bin 
conda init bash
conda activate alphazero

command=""
for i in $(seq 0 1 $(($1)))
do
	command+="ant run_dojos & "
done
command+=" wait"
eval $command