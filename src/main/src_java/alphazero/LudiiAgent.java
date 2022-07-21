package alphazero;

import java.io.File;
import java.net.URL;
import java.util.regex.Pattern;

import org.jpy.PyLib;
import org.jpy.PyModule;
import org.jpy.PyObject;

import game.Game;
import other.AI;
import other.context.Context;
import other.move.Move;


public class LudiiAgent extends AI{
	protected int player = -1;
	private PyModule ludiiPythonModule = null;
	private PyObject pythonAI = null;
	private boolean initialisedJpy = false;
	
	public LudiiAgent(){
		this.friendlyName = "BetaZero";
	}
	
	@Override
	public Move selectAction(
		final Game game, 
		final Context context, 
		final double maxSeconds,
		final int maxIterations,
		final int maxDepth){
		return (Move) pythonAI.call(
			"select_action", game, context, 
			Double.valueOf(maxSeconds), 
			Integer.valueOf(maxIterations), 
			Integer.valueOf(maxDepth)
		).getObjectValue();
	}
	
	@Override
	public void initAI(final Game game, final int playerID){
		this.player = playerID;
		if (!initialisedJpy){
			final URL jarLoc = LudiiAgent.class.getProtectionDomain().getCodeSource().getLocation();
			final String jarPath = 
					new File(jarLoc.getFile()).getParent()
					.replaceAll(Pattern.quote("\\"), "/")
					.replaceAll(Pattern.quote("file:"), "");
			System.setProperty("jpy.config", jarPath + "/libs/jpyconfig.properties");
			if (!PyLib.isPythonRunning()) {
				PyLib.startPython(jarPath);
			}
			ludiiPythonModule = PyModule.importModule("src_python.create_agent");
			initialisedJpy = true;
		}		
		pythonAI = ludiiPythonModule.call("Agent");
		pythonAI.call("init_ai", game, Integer.valueOf(playerID));
	}
}
