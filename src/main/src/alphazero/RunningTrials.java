package alphazero;

import java.util.ArrayList;
import java.util.List;

import game.Game;
import other.AI;
import other.GameLoader;
import other.context.Context;
import other.model.Model;
import other.trial.Trial;
import utils.RandomAI;

import org.jpy.PyLib;
import org.jpy.PyModule;
import org.jpy.PyObject;
import ludii_python_ai.LudiiPythonAI;


public class RunningTrials{
	
	private static final int NUM_TRIALS = 10;
	
	public static void main(String[] args){

		final Game game = GameLoader.loadGameFromName("Tic-Tac-Toe.lud");
		final Trial trial = new Trial(game);
		final Context context = new Context(game, trial);
		
		final List<AI> ais = new ArrayList<AI>();
		ais.add(null);
		/*for (int p = 1; p <= game.players().count(); ++p){
			ais.add(new RandomAI());
		}*/
		ais.add(new RandomAI());
		ais.add(new LudiiPythonAI());
		

		for (int i = 0; i < NUM_TRIALS; ++i){

			game.start(context);
			System.out.println("Starting a new trial!");
	
			for (int p = 1; p <= game.players().count(); ++p)
			{
				ais.get(p).initAI(game, p);
			}
			
			final Model model = context.model();
				
			while (!trial.over()){
				model.startNewStep(context, ais, 1.0); // 1.0 is the thinking time
			}
			

			final double[] ranking = trial.ranking();

			for (int p = 1; p <= game.players().count(); ++p){
				System.out.println("Agent " + context.state().playerToAgent(p) + " achieved rank: " + ranking[p]);
			}
			System.out.println();
		}
	}

}
