# AlphaZeroICGA

## Table des matières

1. [Presentation](#presentation)
2. [Project Artchitecture](#project-architecture)
2. [Competition](#competition)
3. [Baseline](#baseline)
4. [Environment & Setup](#environment--setup)
5. [Try it](#try-it)
6. [Fight it](#fight-it)
7. [What I learned](#what-i-learned)

## Presentation

<p align="center"><img width="800" src="img.jpg"></p>

Implementing deep reinforcement learning algorithms for the ICGA competition. This project is carried out for my 1st year of master internship at the LIP6 (Sorbonne University / CNRS).

## Project architecture

<pre><code>AlphaZeroICGA/
      ├── src/
      |       ├── main/
      |       |      ├── agents/               (Contains the jar files of the final agents)
      |       |      ├── bin/                  (Contains the binary files compiled from src_java)
      |       |      ├── datasets/             (Contains the (state,distrib,value) datasets)
      |       |      ├── final_model/          (Contains the final weights of the best models)
      |       |      ├── libs/                 (Contains the librairies such as JPY/Ludii...)
      |       |      ├── models/               (Contains the current models)
      |       |      ├── src_java/             (Contains all the source code in java)
      |       |      ├── src_python/           (Contains all the source code in python)
      |       |      |      ├── brain/         (Contains the deep learning part)
      |       |      |      ├── mcts/          (Contains the vanilla MCTS and AlphaZero MCTS)
      |       |      |      ├── optimization/  (Contains the optimization part such as precomputations)
      |       |      |      ├── other/         (Contains utility files)
      |       |      |      ├── run/           (Contains files runned by java files such as dojo, trials...)
      |       |      |      ├── scripts/       (Contains all the scripts such as merge_datasets.py)
      |       |      |      ├── settings/      (Contains the hyperparameters and games settings)
      |       |      |      └── utils.py       (File containing the utility functions)
      |       |      ├── alphazero.py          (Script running the whole AlphaZero algorithm)
      |       |      ├── build.xml             (Build file helping us run java commands, clean...)
      |       |      └── notes.txt             (Some notes I left while doing that project)
      |       └── test/                        (Some Ludii tutorials and tests)
      ├── alphazero_env.yml                    (Conda environment save)
      ├── README.md
      └── LICENSE
</pre></code>

## Competition

"The Ludii AI Competition involves general game playing events focussed on developing agents that can play a wide variety of board games. The events use the Ludii general game system to provide the necessary games and API. Games will be provided in the Ludii game description format (.lud). The version used for this competition (1.3.2) of Ludii includes over 1,000 games.

Three events are proposed :

- Kilothon: Best utility obtained on more than 1,000 games against UCT.
- General Game Playing (GGP): Competiton on games present or not in our library.
- Learning: A set of games are announced months before the actual competition, the agents are invited to learn before competing."

**Here we focus on the learning event.**

Links :
- https://icga.org/?page_id=3468
- https://github.com/Ludeme/LudiiAICompetition

## Games

The different games of the learning event this year are :
- Bashni: https://ludii.games/details.php?keyword=Bashni
- Ploy: https://ludii.games/details.php?keyword=Ploy
- Quoridor: https://ludii.games/details.php?keyword=Quoridor
- Mini Wars: https://ludii.games/details.php?keyword=Mini%20Wars
- Plakoto: https://ludii.games/details.php?keyword=Plakoto
- Lotus: https://ludii.games/details.php?keyword=Lotus

## Baseline

We use deep reinforcement learning algorithms for this competition and we start with AlphaZero as a baseline. AlphaGo is an algorithm which can play Go at a super-human level using supervised learning and reinforcement learning. AlphaGo Zero can basically do the same but starting from scratch, hence the "Zero" in its name. AlphaZero does the same but it is able to play different games such as Chess and Shogi.

Links :
- https://www.nature.com/articles/nature16961 (AlphaGo)
- https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf (AlphaGo Zero)
- https://arxiv.org/abs/1712.01815 (AlphaZero)
- https://arxiv.org/pdf/1903.08129.pdf (Hyper-parameters sweep on AlphaZero)
- https://www.scitepress.org/Papers/2021/102459/102459.pdf (Improvements to increase the efficiency of AlphaZero)

## Environment & Setup

The games are hosted on the Ludii software, which is in java. Since we use python for our algorithms we will need a java-python bridge such as **JPY**. Microsoft Visual C++ 14.0 and Java JDK 7.0 are required to build JPY.
We also need the **Ludii** software to run our algorithms in the environment.
We compile a jar file in order to export our AI on Ludii thanks to ant so it is also required even though you can do it otherwise.

Links :
- https://github.com/Ludeme/LudiiPythonAI
- https://github.com/jpy-consortium/jpy
- https://visualstudio.microsoft.com/visual-cpp-build-tools/
- https://www.oracle.com/java/technologies/downloads/
- https://maven.apache.org/download.cgi
- https://ant.apache.org/bindownload.cgi
- https://ludii.games/download.php

First you need to clone Luddi and JPY repositories, then download C++ build and java JDK if you don't have it yet. Apache Maven is also required to build JPY. Once everything is installed go to the JPY folder and run :

`SET VS100COMNTOOLS=<visual-studio-tools-folder>`

`SET JDK_HOME=<your-jdk-dir>`

`SET PATH=<maven-bin-dir>`

`python setup.py build maven bdist_wheel`

If everything worked, you should have a build directory. Copy the content of the lib directory into the Ludii directory in a folder called **/LudiiPythonAI/libs/**. The Ludii jar file should also be moved to the libs directory. Finaly, you can build the jar file thanks to ant and the xml file, then export it in Ludii.

You might also have to specify some paths in the configuration files such as **jpyconfig.py** and **jpyconfig.properties**. You might aswell modify **build.xml** file in order to set the correct classpath for the JPY snapshot.

There is an **alphazero_env.yml** file which can be used to create a conda environnement from scratch with all the required librairies with the command `conda env create -f alphazero_env.yml`.

The required python librairies are :
- tensorflow-gpu (CUDA, cuDNN, TensorFlow...)
- onnx, onnxruntime, onnxruntime-gpu, tf2onnx
- numpy, matplotlib, keras

## Try it

Go to the src/main/ directory and run the next commands in a terminal :

`nano src_python/settings/config.py` : set the settings to run AlphaZero such as number of MCTS simulation, the model hyper-parameters...

`nano src_python/settings/game_settings.py` : set the settings for the different games AlphaZero can play such as number of rows, columns...

`python3 alphazero.py <n_loop> <n_workers>` : runs the whole loop (MCTS simulation with random moves -> dataset -> train model -> save model -> MCTS simulation with model predicting moves -> dataset -> ...). **n_loop** is the number of loop it will achieve. **n_workers** is the number of processes which will be executed in parallel.

The python alphazero script does everything, the following commands are for debugging purposes :

`ant clean` : clean all the directories (**bin/** **build/** **models/** **datasets/**).

`ant build` : compile the java file in **bin/**.

`ant run_trials Dforce_vanilla=<force_vanilla>` : runs the MCTS simulations only (randomly or using the model depending if there is a model in **models/**) and creates a dataset. force_vanilla is an argument which force the MCTS vanilla to play even if there is a model in **models/**.

`ant run_dojos` : runs a 1 versus 1 between the last model (the outsider) and the best current model (the champion model) and outputs some stats. If there is only one model (the champion), runs a 1 versus 1 between the champion model and the MCTS vanilla.

`ant run_tests` : runs tests against Ludii built-in AIs.

`ant create_agent` : takes the best model and build an agent as a jar file for the Ludii software.

`python3 src_python/brain/train_model.py <learning_rate> <force_champion>` : only trains the model using the dataset and saves the best model. force_champion is an argument which force the script to train de champion, default value is false.

`python3 src_python/scripts/merge_datasets.py` : merges all the datasets in **datasets/** with an hash into a unique dataset.

`python3 src_python/scripts/merge_txts.py` : merges all the text files in **models/** with an hash into a unique txt file.

`python3 src_python/scripts/switch_model.py` : switch outsider to champion and champion to old_star.

## Fight it

When the project will be over, the model will be available in the folder **models/final_model/** and the Ludii AI will be in the folder **agents/** as a jar file in order to load it in Ludii software. You will be able to load it against other AIs or against you on different games.

## What I learned

**General knowledge :**
- Papers implementation and understanding (AlphaGo, AlphaGo Zero, AlphaZero)
- Software architecture with different task communicating with each others (alphazero.sh)
- Java wrapper for python with JPY

**Deep learning :**
- Multi-headed neural networks (here for policy + value prediction)
- Huge CNN model with residual blocks and skip connection

**Reinforcement learning :**
- MCTS with UCB/PUCT scores
- State and action representation, reward system, temperature, dirichlet in policy for exploration etc...

**- Time and memory optimization :**
- Multithreading and GPU clusters (for the self-play games and the model training)
- Code optimization because the algorithm is very time consuming (use of profilers)
- Precomputing functions which are called huge amount of time (in MCTS algorithm)
- ONNX format for faster inference with models
