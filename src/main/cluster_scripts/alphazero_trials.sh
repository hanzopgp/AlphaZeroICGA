export PATH=$PATH:/data/vittaut/apache-ant-1.10.12/bin 
conda init bash
conda activate alphazero

command = ""
for i in  0 .. $(($1))
do
	echo $i
	$add = "ant run_trials -Dforce_vanilla=False & "
	command += $add
	echo $add
done
command += " wait"
echo $command
eval $command