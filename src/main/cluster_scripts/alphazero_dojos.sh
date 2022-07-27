export PATH=$PATH:/data/vittaut/apache-ant-1.10.12/bin 
conda init bash
conda activate alphazero

command = ""
for i in $(($1))
do
	add = "ant run_dojos & "
	command = "$command$add"
done
command = "$command wait"
eval $command