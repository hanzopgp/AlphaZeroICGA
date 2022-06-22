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


public class RunningDojoWithPython{

	private static PyModule pythonDojoModule = null;
	private static PyObject pythonDojo = null;
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
			pythonDojoModule = PyModule.importModule("src_python.dojo");
			initialisedJpy = true;
		}
		pythonDojo = pythonDojoModule.call("Dojo");
	}
	
	public static void run(final Game game, final Trial trial, final Context context, final List<AI> ais){
		pythonDojo.call("run_dojo", game, trial, context, ais);
	}
}
