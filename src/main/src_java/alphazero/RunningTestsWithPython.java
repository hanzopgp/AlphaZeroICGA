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

import other.AI;
import utils.RandomAI;

import search.mcts.MCTS;


public class RunningTestsWithPython{

	private static PyModule pythonDojoModule = null;
	private static PyObject pythonDojo = null;
	private static boolean initialisedJpy = false;

	public static void main(String[] args){
		initJPY();
		final Settings settings = new Settings();
		final Game game = GameLoader.loadGameFromName(settings.game);
		System.out.println("--> Game chosen : " + settings.game);
		final Trial trial = new Trial(game);
		final Context context = new Context(game, trial);
		run(game, trial, context, new RandomAI());	
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
			pythonDojoModule = PyModule.importModule("src_python.run.running_tests");
			initialisedJpy = true;
		}
		pythonDojo = pythonDojoModule.call("RunningTests");
	}
	
	public static void run(final Game game, final Trial trial, final Context context, final AI ludiiAI){
		pythonDojo.call("run_test", game, trial, context, ludiiAI);
	}
}
