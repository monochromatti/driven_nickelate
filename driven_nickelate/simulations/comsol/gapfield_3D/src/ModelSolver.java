import com.comsol.model.Model;
import com.comsol.model.ModelNode;
import com.comsol.model.SolverSequence;
import com.comsol.model.util.ModelUtil;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;
import java.util.Arrays;


public class ModelSolver {

    public static void main(String[] args) {
        ModelUtil.initStandalone(false);

        System.out.println("Loading property file");
        Properties prop = readArguments("model.properties");

        String[] required_keys = {
                "modelfile", "mode"
        };
        for (String key : required_keys) {
            if (!prop.containsKey(key)) {
                System.out.println("Error: property file does not contain key '" + key + "'.");
                System.exit(1);
            }
        }

        System.out.println("Loading model");
        System.out.println(prop.getProperty("modelfile"));

        Model model = loadModel(prop.getProperty("modelfile"));

        ModelNode component = model.component("component");
        SolverSequence sol = model.sol("sol1");

        switch (prop.getProperty("mode")) {
            case "sweep":
                System.out.println("Running parameter sweep");
                runParameterSweep(model, component, sol, prop);
                break;
            case "test":
                System.out.println("Running test calculation");
                runTest(model, sol, component);
                break;
            default:
                System.out.println("Skipping calculations.");
                break;
        }
        System.exit(0);
    }

    private static Model loadModel(String modelPath) {
        try {
            return ModelUtil.loadCopy("model", modelPath);
        } catch (IOException e) {
            System.err.println("Error loading model.");
            System.exit(1);
            return null;
        }
    }

    public static Properties readArguments(String argsfile) {
        Properties prop = new Properties();
        FileInputStream input = null;
        try {
            input = new FileInputStream(argsfile);
            prop.load(input);
        } catch (IOException e) {
            System.err.println("Error reading arguments file");
            System.exit(1);
        } finally {
            if (input != null) {
                try {
                    input.close();
                } catch (IOException e) {
                    System.err.println("Error closing arguments file");
                    System.exit(1);
                }
            }
        }
        return prop;
    }

    public static void setConductivity(Model model, String cond, String tag) {
        model.material(tag).propertyGroup("def").set("electricconductivity", new String[]{cond, "0", "0", "0", cond, "0", "0", "0", cond});
    }

    public static void setFrequencies(SolverSequence sol, String freq_values) {
        sol.feature("v1").set("clist", new String[]{freq_values});
        sol.feature("s1").feature("p1").set("plistarr", new String[]{freq_values});
    }

    public static String elapsedTimeString(long startTime) {
        long elapsedSeconds = (System.currentTimeMillis() - startTime) / 1000;
        long minutes = elapsedSeconds / 60;
        long seconds = elapsedSeconds % 60;
        return String.format("%d:%02d", minutes, seconds);
    }

    public static void runTest(Model model, SolverSequence sol, ModelNode component) {
        setConductivity(model, "0[S/m]", "ndnio3_gap");
        setConductivity(model, "0[S/m]", "ndnio3");

        component.material("matlnk_film").set("link", "ndnio3");
        component.material("matlnk_gap").set("link", "ndnio3_gap");

        setFrequencies(sol, "1[THz]");

        double[] solutionTimes = new double[10];

        for (int i = 0; i < 10; i++) {
            double startTime = System.currentTimeMillis();
            sol.runAll();
            solutionTimes[i] = System.currentTimeMillis() - startTime;
            System.out.println("Solution time " + i + ": " + solutionTimes[i] / 1000 + " s");
        }
        double meanSolutionTimes = Arrays.stream(solutionTimes).sum() / solutionTimes.length;
        System.out.println("Average solution time: " + meanSolutionTimes / 1000 + " s");
    }

    public static void runParameterSweep(Model model, ModelNode component, SolverSequence sol, Properties prop) {
        setFrequencies(sol, prop.getProperty("freq_values"));
        int num_freqs = prop.getProperty("freq_values").split(",").length;
        System.out.println("Number of frequencies: " + num_freqs);

        String[] gapConductivities = prop.getProperty("gapcond_values").split(",");
        String[] filmConductivities = prop.getProperty("filmcond_values").split(",");

        component.material("matlnk_film").set("link", "ndnio3");
        component.material("matlnk_gap").set("link", "ndnio3_gap");

        String spectralData = prop.getProperty("spectral_data");
        boolean recordSpectral = (prop.containsKey("spectral_data")) && !spectralData.isEmpty();

        int num_params = gapConductivities.length;
        System.out.println("Number of parameters: " + num_params);

        long startTime = System.currentTimeMillis();
        for (int i = 0; i < num_params; i++) {

            setConductivity(model, gapConductivities[i], "ndnio3_gap");
            setConductivity(model, filmConductivities[i], "ndnio3");

            sol.runAll();
            if (recordSpectral) {
                model.result().table("params_table").clearTableData();
                model.result().numerical("gap_integral").setResult();
                model.result().numerical("line_average").appendResult();
                model.result().export("params_export")
                        .set("header", (i == 0))
                        .set("filename", spectralData);
                model.result().export("params_export").run();
            }
            double progressPercentage = ((double) (i + 1) / gapConductivities.length) * 100;
            System.out.println((int) progressPercentage + "%, " + elapsedTimeString(startTime) + " elapsed");
        }
    }
}
