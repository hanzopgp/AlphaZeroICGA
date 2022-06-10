package alphazero;

import java.util.ArrayList;
import java.util.List;
import java.util.Iterator;

import game.Game;
import other.AI;
import other.GameLoader;
import other.context.Context;
import other.model.Model;
import other.trial.Trial;
import other.location.Location;
import utils.RandomAI;

import ludii_python_ai.LudiiPythonAI;


public class RunningTrials{
	
	private static final int NUM_TRIALS = 1;
	
	public static void main(String[] args){

		final Game game = GameLoader.loadGameFromName("Tic-Tac-Toe.lud");
		final Trial trial = new Trial(game);
		final Context context = new Context(game, trial);
		
		final List<AI> ais = new ArrayList<AI>();
		ais.add(null);
		ais.add(new RandomAI());
		ais.add(new LudiiPythonAI());
		

		for (int i = 0; i < NUM_TRIALS; ++i){

			game.start(context);
			System.out.println("Starting a new trial!");
	
			for (int p = 1; p <= game.players().count(); ++p){
				ais.get(p).initAI(game, p);
			}
			
			final Model model = context.model();
				
			// Checking context representation to understand how to represent states
			while (!trial.over()){
				model.startNewStep(context, ais, 0.1);
			
				System.out.println("===================== state functions =====================");
				int mover = context.state().mover();
				int opp_mover = (mover==1 ? 2 : 1);
				System.out.println("Mover: " + mover);
				System.out.println("Opponent: " + opp_mover);
				
				List<? extends Location>[] map_pos = context.state().owned().positions(mover);
				List<? extends Location>[] map_pos_opp = context.state().owned().positions(opp_mover);
				System.out.println("Map positions for mover:");
				for(int j=0; j<map_pos.length; j++){
					for(int k=0; k<map_pos[j].size(); k++){
						System.out.print(map_pos[j].get(k).site() + " ");
					}
					System.out.println();
				}
				System.out.println("Map positions for opponent:");
				for(int j=0; j<map_pos_opp.length; j++){
					for(int k=0; k<map_pos_opp[j].size(); k++){
						System.out.print(map_pos_opp[j].get(k).site() + " ");
					}
					System.out.println();
				}
				
				/*System.out.println("StateHash: " + context.state().stateHash());
				
				System.out.println("===================== game functions =====================");
				System.out.println("Legal moves: " + context.game().moves(context).moves());
				
				System.out.println("===================== trial functions =====================");
				Iterator it = context.trial().reverseMoveIterator();
				while(it.hasNext()){
					System.out.println(it.next())	;
				}*/
				
				System.out.println("**********************************************************************");
			}
			
			/*while (!trial.over()){
				model.startNewStep(context, ais, 1.0); // 1.0 is the thinking time
				System.out.println(
			}*/

			final double[] ranking = trial.ranking();

			for (int p = 1; p <= game.players().count(); ++p){
				System.out.println("Agent " + context.state().playerToAgent(p) + " achieved rank: " + ranking[p]);
			}
			System.out.println();
		}
	}

}
