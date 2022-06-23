ant clean

winners_file="./models/winners.txt"
alphazero_iteration=0
trial=True

echo "---> STARTING TRAINING SESSION"

while [ $alphazero_iteration -le $1 ]
do
	echo "********************************************************************************************"
	echo "************************************ITERATION ALPHAZERO "$alphazero_iteration"***********************************"
	echo "********************************************************************************************"
	echo "****************************************MCTS TRIALS*****************************************"
	echo "********************************************************************************************"
	if [[ trial ]]
	then
		ant mcts_trials	
	fi
	echo "********************************************************************************************"
	echo "***************************************TRAINING MODEL***************************************"
	echo "********************************************************************************************"
	python3 src_python/train_model.py
	if [[ $alphazero_iteration -ge 1 ]]     		# If it's the first step we won't
	then                                    		# go for a dojo since there is only
		echo "********************************************************************************************"
		echo "*****************************************MCTS DOJO******************************************"
		echo "********************************************************************************************"
		ant mcts_dojo                   		# one model ready
		winner=$( tail -n 1 $winners_file )
		if [[ ${winner::1} == "O" ]]        		# If outsider model won 
		then						# outsider becomes champion
			python3 src_python/switch_model.py 	# and we can go back to mcts_trial
			trial=True              
		fi
		elif [[ ${winner::1} == "C" ]]     		# If the champion won we need to
		then 						# train the model again without
			trial=False             		# executing mcts_trial
		fi
	((alphazero_iteration++))
done

echo "---> TRAINING SESSION DONE"
