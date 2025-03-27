import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

import com.comsol.model.ExportFeature;
import com.comsol.model.Model;
import com.comsol.model.ModelNode;
import com.comsol.model.SolverSequence;
import com.comsol.model.util.ModelUtil;

public class ModelSolver {
    public static void main(String[] args) {
        ModelUtil.initStandalone(false);

        System.out.println("Loading property file ...");

        Properties prop = readArguments("model.properties");

        String[] required_keys = {
                "freq_range",
                "cond_range",
                "tau_range",
                "modelfile",
                "jobid",
                "run",
                "exit",
        };
        for (String key : required_keys) {
            if (!prop.containsKey(key)) {
                System.out.println("Error: property file does not contain key '" + key + "'.");
                System.exit(1);
            }
        }

        System.out.println("Loading " + prop.getProperty("modelfile"));

        Model model = loadModel("model_" + prop.getProperty("jobid"), prop.getProperty("modelfile"));
        ModelNode component = model.component("component");
        SolverSequence sol = model.sol("sol1");

        setFrequencies(sol, prop.getProperty("freq_range"));

        if (prop.getProperty("run").equals("true")) {
            System.out.println("Running parameter sweep ...");
            runParameterSweep(model, component, sol, prop);
        } else {
            System.out.println("Skipping parameter sweep.");
        }

        if (prop.getProperty("exit").equals("true")) {
            System.out.println("Exiting.");
            System.exit(0);
        }
    }

    private static Model loadModel(String modelTag, String modelPath) {
        try {
            return ModelUtil.load(modelTag, modelPath);
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
                    System.err.println("Error reading arguments file");
                }
            }
        }
        return prop;
    }

    public static void setConductivity(ModelNode component, String cond, String tag) {
        component.material(tag).propertyGroup("def")
                .set("electricconductivity",
                        new String[]{cond, "0", "0", "0", cond, "0", "0", "0",
                                cond});
    }

    public static void setFrequencies(SolverSequence sol, String freq_range) {
        sol.feature("v1").set("clist", new String[]{freq_range});
        sol.feature("s1").feature("p1").set("plistarr", new String[]{freq_range});

    }

    public static void exportParameters(Model model, String filename) {

        model.result().table("params_table").clearTableData();

        model.result().numerical("gap_point").set("table", "params_table");
        model.result().numerical("gap_point").setResult();
        model.result().numerical("gev1").appendResult();

        model.result().export("params_export").set("filename", filename);
        model.result().export("params_export").run();
    }

    public static void exportSpatial(Model model, String filename) {
        ExportFeature exportSpatial = model.result().export("spatial_data");
        exportSpatial.set("filename", filename);
        exportSpatial.run();
    }

    public static void runParameterSweep(Model model, ModelNode component, SolverSequence sol, Properties prop) {
        String[] conductivityValues = prop.getProperty("cond_range").split(",");
        String[] scatteringTimes = prop.getProperty("tau_range").split(",");

        for (int i = 0; i < conductivityValues.length; i++) {
            System.out.printf("%d: Running model for cond = %s, tau = %s%n", i,
                    conductivityValues[i], scatteringTimes[i]);

            String cond = String.format("%s[S/m]/(1 - j*6.283*freq[THz]*%s[ps])",
                    conductivityValues[i], scatteringTimes[i]);
            setConductivity(component, cond, "gap_mat");
            setConductivity(component, cond, "film_mat");
            sol.runAll();

            if (prop.containsKey("spectral_data")) {
                exportParameters(model, prop.getProperty("spectral_data"));
            }
            if (prop.containsKey("spatial_data")) {
                exportSpatial(model, prop.getProperty("spatial_data"));
            }
        }
    }
}
