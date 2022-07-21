package alphazero;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import java.io.FileReader;  
import java.io.BufferedReader; 
import java.io.FileNotFoundException; 
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

	public static void main(String[] args) throws FileNotFoundException{
		initJPY();
		final Settings settings = new Settings();
		final Game game = GameLoader.loadGameFromName(settings.game);
		System.out.println("--> Game chosen : " + settings.game);
		final Trial trial = new Trial(game);
		final Context context = new Context(game, trial);
		run(game, trial, context, Boolean.parseBoolean(args[0]));
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
			pythonTrialModule = PyModule.importModule("src_python.run.running_trials");
			initialisedJpy = true;
		}
		pythonTrial = pythonTrialModule.call("RunningTrials");
	}
	
	public static int getNObjects() throws FileNotFoundException{
		int nObjects = -1;
		File file = new File("src_python/config.py");
		try {
			Scanner scanner = new Scanner(file);
			int lineNum = 0;
			while (scanner.hasNextLine()) {
				String line = scanner.nextLine();
				lineNum++;
				if(line.contains("MAX_WORKERS")) { 
					nObjects = Integer.parseInt(line.substring(14));
					return nObjects;
				}
			}
		} catch(FileNotFoundException e) { 
			System.out.println(e);
		}
		return nObjects;
	}
	
	public static void run(final Game game, final Trial trial, final Context context, final boolean force_vanilla){
		pythonTrial.call("run_trial", game, trial, context, force_vanilla);
	}
}
