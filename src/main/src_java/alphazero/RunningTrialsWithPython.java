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

	private static PyModule pythonTrialModule = null;
	private static PyObject pythonTrial = null;
	private static boolean initialisedJpy = false;

	public static void main(String[] args){
		initJPY();
		final Game game = GameLoader.loadGameFromName("Bashni.lud");
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
			final URL jarLoc = RunningTrialsWithPython.class.getProtectionDomain().getCodeSource().getLocation();
			final String jarPath = 
					new File(jarLoc.getFile()).getParent()
					.replaceAll(Pattern.quote("\\"), "/")
					.replaceAll(Pattern.quote("file:"), "");
			System.setProperty("jpy.config", jarPath + "/libs/jpyconfig.properties");
			if (!PyLib.isPythonRunning()) {
				PyLib.startPython(jarPath);
			}
			pythonTrialModule = PyModule.importModule("src_python.running_trials");
			initialisedJpy = true;
		}
		pythonTrial = pythonTrialModule.call("RunningTrials");
	}
	
	public static void run(final Game game, final Trial trial, final Context context, final List<AI> ais){
		pythonTrial.call("run", game, trial, context, ais);
	}
}
