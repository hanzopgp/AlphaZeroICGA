package alphazero;

import java.io.File;
import java.net.URL;
import java.util.regex.Pattern;

import org.jpy.PyLib;
import org.jpy.PyModule;
import org.jpy.PyObject;

import game.Game;
import other.GameLoader;


public class RunningTrialsWithPython{

	/** The "ludii_python.running_trials" Python Module */
	private static PyModule pythonTrialModule = null;
	/** This will hold our trial object (implemented in Python) */
	private static PyObject pythonTrial = null;
	/** Did we perform initialisation required for JPY? */
	private static boolean initialisedJpy = false;

	public static void main(String[] args){
		initJPY();
		Game game = initGame("Tic-Tac-Toe.lud");
		run(game);	
	}
	
	public static Game initGame(String game_str){
		final Game game = GameLoader.loadGameFromName(game_str);
		return game;
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
	
	public static void run(final Game game){
		System.out.println(pythonTrial.call("run", game));
	}
}
