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
		final int nObjects = getNObjects();
		System.out.println("--> Loading " + nObjects + " game objects...");
		final Game[] games = new Game[nObjects];
		final Trial[] trials = new Trial[nObjects];
		final Context[] contexts = new Context[nObjects];
		final ArrayList<AI>[] ais = new ArrayList[nObjects];
		// Need to give list of objects because we are going to use multi threading
		for(int i=0; i<nObjects; i++){
			Game game = GameLoader.loadGameFromName("Bashni.lud");
			games[i] = game;
			Trial trial = new Trial(game);
			trials[i] = trial;
			contexts[i] = new Context(game, trial);
			// Need to create a Java List object here, if we give 2 ais or it won't work
			// because we need to give Java List to Java methods python-side
			ais[i] = new ArrayList<AI>();
			ais[i].add(null);
			ais[i].add(new RandomAI());
			ais[i].add(new RandomAI());
		}
		System.out.println("--> Done !");
		run(games, trials, contexts, ais, nObjects);
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
	
	public static void run(final Game[] games, final Trial[] trials, final Context[] contexts, final List<AI>[] ais, final int nObjects){
		//pythonTrial.call("run_trial", game, trial, context, ais);
		pythonTrial.call("run_parallel_trials", games, trials, contexts, ais, nObjects);
	}
}
