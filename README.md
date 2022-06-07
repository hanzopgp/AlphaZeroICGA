# AlphaZeroICGA

## Table des matières

1. [Presentation](#presentation-)
2. [Competition](#competition-)
3. [Environment](#environment-)
4. [Baseline](#baseline-)

## Presentation

<p align="center"><img width="500" src="img.jpg"></p>

Implementing deep reinforcement learning algorithms for the ICGA competition.

## Competition

"The Ludii AI Competition involves general game playing events focussed on developing agents that can play a wide variety of board games. The events use the Ludii general game system to provide the necessary games and API. Games will be provided in the Ludii game description format (.lud). The version used for this competition (1.3.2) of Ludii includes over 1,000 games.

Three events are proposed :

- Kilothon: Best utility obtained on more than 1,000 games against UCT.
- General Game Playing (GGP): Competiton on games present or not in our library.
- Learning: A set of games are announced months before the actual competition, the agents are invited to learn before competing.
Each event will run if at least 3 participants take part in it."

Here we focus on the learning event.

Link : https://github.com/Ludeme/LudiiAICompetition

## Environment

The games are hosted on the Ludii software, which is in java. Since we use python for our algorithms we will need a java-python bridge such as **JPY**, we also need the **Ludii** software.

Links : 
- https://github.com/Ludeme/LudiiPythonAI
- https://github.com/jpy-consortium/jpy

## Baseline

We use deep reinforcement learning algorithms for this competition and we start with AlphaZero as a baseline. AlphaGo is an algorithm which can play Go at a super-human level using supervised learning and reinforcement learning. AlphaGo Zero can basically do the same but starting from scratch, hence the "Zero" in its name. AlphaZero does the same but it is able to play different games such as Chess and Shogi.

Links : 
- https://www.nature.com/articles/nature16961 (AlphaGo)
- https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf (AlphaGo Zero)
- https://arxiv.org/abs/1712.01815 (AlphaZero)
