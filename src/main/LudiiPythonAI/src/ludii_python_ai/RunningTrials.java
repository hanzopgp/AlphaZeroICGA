package ludii_python_ai;

import java.util.ArrayList;
import java.util.List;

import game.Game;

import other.AI;
import other.GameLoader;
import other.context.Context;
import other.model.Model;
import other.trial.Trial;

import utils.RandomAI;


public class RunningTrials{

	private static final int NUM_TRIALS = 10;

	public static void main(final String args[]){
	
		System.out.println("Hi");
		final Game game = GameLoader.loadGameFromName("Tic-Tac-Toe.lud");
		final Trial trial = new Trial(game);
		final Context context = new Context(game, trial);
		final List<AI> players = new ArrayList<AI>();
		players.add(null);
		
		for(int p = 1; p <= game.players().count(); ++p){
			players.add(new RandomAI());
		}
		
		for(int i = 0; i < NUM_TRIALS; i++){
			game.start(context);
			for(int p = 1; p <= game.players().count(); ++p){
				players.get(p).initAI(game, p);
			}
			final Model model = context.model();
			while(!trial.over()){
				model.startNewStep(context, players, 1.0);
			}
			final double[] ranking = trial.ranking();
			for(int p = 1; p <= game.players().count(); ++p){
				//System.out.println("Agent " + context.state().playerToAgent(p) + " achieved rank: " + ranking[p]);
				System.out.println(p);
			}
			System.out.println();
		}
		
	}
	
}
