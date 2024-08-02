import com.comsol.model.ExportFeature;
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

        boolean checkoutSuccess = false;
        for (int attempts = 0; attempts < 10 && !checkoutSuccess; attempts++) {
            System.out.println("Checking out license");
            checkoutSuccess = ModelUtil.checkoutLicense(new String[]{"RF"});
            if (!checkoutSuccess) {
                System.out.println("Failed. Retrying after 5 minutes.");
                try {
                    Thread.sleep(300000);
                } catch (InterruptedException e) {
                    System.out.println("Error sleeping");
                }
            }
        }

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
            case "reference":
                System.out.println("Running reference calculation");
                runSubstrateReference(model, sol, prop);
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

    public static void disableBoundaryConditions(Model model) {
        model.study("std1").feature("freq").set(
                "useadvanceddisable", true);
        model.study("std1").feature("freq").set(
                "disabledphysics", new String[]{"emw/trans1", "emw/trans2", "emw/trans3", "emw/ltbc1"});
    }

    public static String elapsedTimeString(long startTime) {
        long elapsedSeconds = (System.currentTimeMillis() - startTime) / 1000;
        long minutes = elapsedSeconds / 60;
        long seconds = elapsedSeconds % 60;
        return String.format("%d:%02d", minutes, seconds);
    }

    public static void runParameterSweep(Model model, ModelNode component, SolverSequence sol, Properties prop) {
        setFrequencies(sol, prop.getProperty("freq_values"));
        int num_freqs = prop.getProperty("freq_values").split(",").length;
        System.out.println("Number of frequencies: " + num_freqs);

        String[] gapConductivities = prop.getProperty("gapcond_values").split(",");
        String[] filmConductivities = prop.getProperty("filmcond_values").split(",");

        component.material("matlnk_film").set("link", "ndnio3");
        component.material("matlnk_gap").set("link", "ndnio3_gap");

        int num_params = gapConductivities.length;
        System.out.println("Number of parameters: " + num_params);

        String spectralData = prop.getProperty("spectral_data");
        boolean recordSpectral = (prop.containsKey("spectral_data")) && !spectralData.isEmpty();

        String spatialData = prop.getProperty("spectral_data");
        boolean recordSpatial = (prop.containsKey("spatial_data")) && !spatialData.isEmpty();

        long startTime = System.currentTimeMillis();
        int startIndex = prop.getProperty("start_index").isEmpty() ? 0 : Integer.parseInt(prop.getProperty("start_index"));
        for (int i = startIndex; i < num_params; i++) {

            setConductivity(model, gapConductivities[i], "ndnio3_gap");
            setConductivity(model, filmConductivities[i], "ndnio3");

            sol.runAll();
            if (recordSpectral) {
                model.result().table("params_table").clearTableData();
                model.result().numerical("gap_point").setResult();
                model.result().numerical("film_point").appendResult();
                model.result().numerical("global_eval").appendResult();
                model.result().export("params_export")
                        .set("header", (i == 0))
                        .set("filename", spectralData);
                model.result().export("params_export").run();
            }
            if (recordSpatial) {
                ExportFeature exportSpatial = model.result().export("spatial_data");
                exportSpatial
                        .set("header", (i == 0))
                        .set("filename", spatialData);
                exportSpatial.run();
            }

            System.out.println(i + "/" + num_params + ", " + elapsedTimeString(startTime) + " elapsed");
        }
        System.out.println("Finished. " + elapsedTimeString(startTime) + " elapsed");
    }

    public static void runSubstrateReference(Model model, SolverSequence sol, Properties prop) {
        System.out.println("Running model for substrate reference");

        disableBoundaryConditions(model);
        setFrequencies(sol, prop.getProperty("freq_values"));

        sol.runAll();
        if (prop.containsKey("spectral_data")) {
            String spectralData = prop.getProperty("spectral_data");
            if (!spectralData.isEmpty()) {
                model.result().table("params_table").clearTableData();
                model.result().numerical("global_eval").setResult();
                model.result().export("params_export").set("filename", prop.getProperty("spectral_data"));
                model.result().export("params_export").run();
            }

        }
        if (prop.containsKey("spatial_data")) {
            String spatialData = prop.getProperty("spatial_data");
            if (!spatialData.isEmpty()) {
                model.result().export("spatial_data").set("filename", prop.getProperty("spatial_data"));
                model.result().export("spatial_data").run();
            }
        }
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
}
