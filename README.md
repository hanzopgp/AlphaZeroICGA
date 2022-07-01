# AlphaZeroICGA

## Table des matières

1. [Presentation](#presentation)
2. [Project Artchitecture](#project-architecture)
2. [Competition](#competition)
3. [Baseline](#baseline)
4. [Environment & Setup](#environment--setup)
5. [Try it](#try-it)
6. [Fight it](#fight-it)

## Presentation

<p align="center"><img width="800" src="img.jpg"></p>

Implementing deep reinforcement learning algorithms for the ICGA competition. This project is carried out for my 1st year of master internship at the LIP6 (Sorbonne University / CNRS).

## Project architecture

<pre><code>AlphaZeroICGA/
      ├── src/
      |       ├── main/
      |       |      ├── agents/         (Contains the jar files of the final agents)
      |       |      ├── bin/            (Contains the binary files compiled from src_java)
      |       |      ├── datasets/       (Contains the (state,distrib,value) datasets)
      |       |      ├── final_model/    (Contains the final weights of the best models)
      |       |      ├── libs/           (Contains the librairies such as JPY/Ludii...)
      |       |      ├── models/         (Contains the current models)
      |       |      ├── src_java/       (Contains all the source code in java)
      |       |      ├── src_python/     (Contains all the source code in python)
      |       |      ├── alphazero.sh    (Script running the whole AlphaZero algorithm)
      |       |      ├── build.xml       (Build file helping us run java commands, clean...)
      |       |      └── notes.txt       (Some notes I left while doing that project)
      |       └── test/
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

You also have to specify some paths in the configuration files such as **jpyconfig.py** and **jpyconfig.properties**.

The required python librairies are :
- Numpy 1.22.4
- Matplotlib 3.3.4
- TensorFlow 2.9.1
- Pandas 1.1.5

## Try it

Go to the src/main/ directory and run the next commands in a terminal :

`nano src_python/config.py` : set the settings to run AlphaZero such as number of simulations, game type...

`python3 alphazero.py <n_loop> <n_workers>` : runs the whole loop (MCTS simulation with random moves -> dataset -> train model -> save model -> MCTS simulation with model predicting moves -> dataset -> ...). **n_loop** is the number of loop it will achieve. **n_workers** is the number of processes which will be executed in parallel.

The python alphazero script does everything, the following commands are for debugging purposes :

`ant clean` : clean all the directories (**bin/** **build/** **models/** **datasets/**).

`ant build` : compile the java file in **bin/**.

`ant mcts_trials` : runs the MCTS simulations only (randomly or using the model depending if there is a model in **models/**) and creates a dataset.

`ant train_model` : only trains the model using the dataset and save the best model.

`ant mcts_dojo` : runs a 1 versus 1 between the last model (the outsider) and the best current model (the champion model) and outputs some stats.

`ant create_agent` : takes the best model and build an agent as a jar file for the Ludii software.

## Fight it

When the project will be over, the models will be available in the folder **final_models/** and the Ludii AI will be in the folder **agents/** as jar files in order to load them in Ludii software. You will be able to load it against other AIs or against you on different games.

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
- Reinforcement learning things such as temperature, dirichlet in policy for exploration etc...

**- Time and memory optimization :**
    - Multithreading and GPU clusters (for the self-play games and the model training)
    - Code optimization because the algorithm is very time consuming (use of profilers)
    - Precomputing functions which are called huge amount of time (in MCTS algorithm)

