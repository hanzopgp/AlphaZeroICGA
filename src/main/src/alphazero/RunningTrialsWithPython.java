package alphazero;

import java.util.ArrayList;
import java.util.List;

import java.io.File;
import java.net.URL;
import java.util.regex.Pattern;

import org.jpy.PyLib;
import org.jpy.PyModule;
import org.jpy.PyObject;

import game.Game;
import other.trial.Trial;
import other.context.Context;
import other.GameLoader;
import utils.RandomAI;
import other.AI;


public class RunningTrialsWithPython{

	/** The "ludii_python.running_trials" Python Module */
	private static PyModule pythonTrialModule = null;
	/** This will hold our trial object (implemented in Python) */
	private static PyObject pythonTrial = null;
	/** Did we perform initialisation required for JPY? */
	private static boolean initialisedJpy = false;

	public static void main(String[] args){
		initJPY();
		final Game game = GameLoader.loadGameFromName("Tic-Tac-Toe.lud");
		final Trial trial = new Trial(game);
		final Context context = new Context(game, trial);
		final List<AI> ais = new ArrayList<AI>();
		ais.add(null);
		ais.add(new RandomAI());
		ais.add(new RandomAI());
		
		run(game, trial, context, ais);	
	}
	
	public static void initJPY(){
		if (!initialisedJpy){
			// We always expect this AI class to be in a JAR file. Let's find out where our JAR file is
			final URL jarLoc = RunningTrialsWithPython.class.getProtectionDomain().getCodeSource().getLocation();
			final String jarPath = 
					new File(jarLoc.getFile()).getParent()
					.replaceAll(Pattern.quote("\\"), "/")
					.replaceAll(Pattern.quote("file:"), "");
			
			// Set JPY config property relative to this JAR path
			System.setProperty("jpy.config", jarPath + "/libs/jpyconfig.properties");

			// Make sure that Python is running
			if (!PyLib.isPythonRunning()) 
			{
				// We expect the python code to be in the same directory as this JAR file;
				// therefore, we include this path such that Python can discover our python code
				PyLib.startPython(jarPath);
			}
			pythonTrialModule = PyModule.importModule("ludii_python.running_trials");
			initialisedJpy = true;
		}
		// Instantiate a new trial (implemented in the Python class "RunningTrials")
		pythonTrial = pythonTrialModule.call("RunningTrials");
	}
	
	public static void run(final Game game, final Trial trial, final Context context, final List<AI> ais){
		pythonTrial.call("run", game, trial, context, ais);
	}
}
