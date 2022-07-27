export PATH=$PATH:/data/vittaut/apache-ant-1.10.12/bin 
conda init bash
conda activate alphazero

command = ""
for i in $(($1))
do
	add = "ant run_trials -Dforce_vanilla=False & "
	command += $add
done
command = "${command} wait"
echo $command
eval $command