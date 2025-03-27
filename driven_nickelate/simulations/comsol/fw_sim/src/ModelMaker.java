import com.comsol.model.*;
import com.comsol.model.util.ModelUtil;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

public class ModelMaker {

        public static void main(String[] args) {

                ModelUtil.initStandalone(false);

                Properties prop = readArguments("model.properties");

                String[] required_keys = {
                                "freq_range",
                                "gold_datafile",
                                "lsat_datafile",
                                "cadfile"
                };
                for (String key : required_keys) {
                        if (!prop.containsKey(key)) {
                                System.out.println("Error: property file does not contain key '" + key + "'.");
                                System.exit(1);
                        }
                }

                System.out.println("Creating model...");
                Model model = ModelUtil.create("Model");

                System.out.println("Defining parameters...");
                defineParameters(model);

                System.out.println("Creating component...");
                ModelNode component = model.component().create("component", true);

                Map<String, int[]> selections = getSelections();

                System.out.println("Building geometry...");
                GeomSequence geometry = component.geom().create("geom", 3);
                setupGeometry(geometry, prop.getProperty("cadfile"));
                geometry.run();

                System.out.println("Building mesh...");
                buildMesh(component, selections);

                System.out.println("Setting up materials...");
                setupMaterials(model, component, prop, selections);

                System.out.println("Setting up physics...");
                setupPhysics(component, selections);

                System.out.println("Creating solver...");
                createSolver(model);

                System.out.println("Creating exports...");
                createExports(model);

                try {
                        model.save("model.mph");
                        System.out.println("Model saved to model.mph");
                        System.exit(0);
                } catch (IOException e) {
                        System.out.println("Error saving model");
                        System.exit(1);
                }
        }

        public static Properties readArguments(String argsfile) {
                Properties prop = new Properties();
                try {
                        FileInputStream input = new FileInputStream(argsfile);
                        prop.load(input);
                } catch (IOException e) {
                        System.out.println("Error reading arguments file");
                        System.exit(1);
                }
                return prop;
        }

        public static void defineParameters(Model model) {
                model.param()
                                .set("theta", "0[deg]", "Elevation angle")
                                .set("phi", "0[deg]", "Azimuth angle")
                                .set("rot", "0[deg]", "Rotation angle")
                                .set("lda_c", "c_const/1[THz]", "Characteristic wavelength")
                                .set("g_c", "600[nm]", "Characteristic minimal feature size")
                                .set("n_c", "4.8", "Characteristic substrate index")
                                .set("p_c", "70[um]", "Characteristic unit cell size")
                                .set("cond", "0.00E+00[S/m]", "Conductivity")
                                .label("Characteristic quantities");
        }

        private static Map<String, int[]> getSelections() {
                Map<String, int[]> selections = new HashMap<>();
                selections.put("resonator", new int[] { 14 });
                selections.put("film", new int[] { 15, 20 });
                selections.put("gap", new int[] { 9 });
                selections.put("surface", new int[] { 14, 15, 20, 9 });
                selections.put("sbc", new int[] { 3, 13 });
                selections.put("pmc", new int[] { 1, 4, 7, 10, 21, 22, 23, 24 });
                selections.put("substrate", new int[] { 1, 2 });
                selections.put("port_emitter", new int[] { 12 });
                selections.put("port_receiver", new int[] { 6 });
                return selections;
        }

        public static void setupGeometry(GeomSequence geometry, String cadfile) {
                geometry.lengthUnit("µm");
                geometry.geomRep("comsol");

                geometry.create("wp_extrusion", "WorkPlane").label("wp_extrusion");
                geometry.feature("wp_extrusion")
                                .set("selresult", true)
                                .set("selplaneshow", true)
                                .set("unite", true);
                geometry.feature("wp_extrusion").geom().create("imp3", "Import")
                                .set("type", "dxf")
                                .set("filename", cadfile)
                                .set("alllayers", new String[] { "0", "structure", "gap", "unit_cell" })
                                .set("layerselection", "selected")
                                .set("layers", new String[] { "unit_cell" })
                                .label("unit_cell");
                geometry.feature("wp_extrusion").geom().feature("imp3").importData();
                geometry.feature("wp_extrusion").geom().create("sq1", "Square").label("wp_extrusion");
                geometry.feature("wp_extrusion").geom().feature("sq1").set("size", 100);
                geometry.feature("wp_extrusion").geom().create("int1", "Intersection")
                                .selection("input").set("imp3", "sq1");
                geometry.create("ext1", "Extrude")
                                .set("inputhandling", "keep")
                                .set("distance", new String[] { "150[um]", "100[um]", "-500[um]", "-550[um]" })
                                .set("scale", new double[][] { { 1, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 } })
                                .set("displ", new double[][] { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } })
                                .set("twist", new int[] { 0, 0, 0, 0 })
                                .label("extrude_domain");
                geometry.feature("ext1").selection("input").named("wp_extrusion");
                geometry.create("wp_structure", "WorkPlane").label("wp_structure");
                geometry.feature("wp_structure")
                                .set("selresult", true)
                                .set("selplaneshow", true)
                                .set("unite", true);
                geometry.feature("wp_structure").geom().create("imp3", "Import")
                                .set("type", "dxf")
                                .set("filename", cadfile)
                                .set("alllayers", new String[] { "0", "structure", "gap", "unit_cell" })
                                .set("layerselection", "selected")
                                .set("layers", new String[] { "structure" })
                                .label("structure");
                geometry.feature("wp_structure").geom().create("spl1", "Split")
                                .label("Split");
                geometry.feature("wp_structure").geom().feature("spl1").selection("input").set("imp3");
                geometry.feature("wp_structure").geom().create("imp4", "Import")
                                .set("type", "dxf")
                                .set("filename", cadfile)
                                .set("alllayers", new String[] { "0", "structure", "gap", "unit_cell" })
                                .set("layerselection", "selected")
                                .set("layers", new String[] { "gap" })
                                .label("surface");
                geometry.feature("wp_structure").geom().feature("imp4").importData();
                geometry.feature("wp_structure").geom().create("pt1", "Point");
                geometry.feature("wp_structure").geom().create("uni1", "Union")
                                .selection("input").set("imp4", "spl1");
                geometry.feature("wp_structure").geom().create("sq1", "Square")
                                .set("size", 100);
                geometry.feature("wp_structure").geom().create("int1", "Intersection")
                                .selection("input").set("sq1", "uni1");
                geometry.create("ballsel1", "BallSelection")
                                .set("entitydim", 0)
                                .set("r", 0.1)
                                .set("condition", "inside")
                                .label("gapcenter_ballsel");
                geometry.create("sel1", "ExplicitSelection").label("surface_selection");
                geometry.feature("sel1").selection("selection").init(2);
                geometry.feature("sel1").selection("selection").set("wp_structure.uni", 1, 2, 3, 4);
        }

        public static void buildMesh(ModelNode component, Map<String, int[]> selections) {
                MeshSequence mesh = component.mesh().create("mesh");

                mesh.feature("size")
                                .set("custom", "on")
                                .set("hmax", "lda_c/10")
                                .set("hmin", "lda_c/1E4")
                                .set("hauto", 3);

                mesh.create("ftri_surface", "FreeTri").label("surface");
                mesh.feature("ftri_surface").selection().set(selections.get("surface"));
                mesh.feature("ftri_surface").create("size1", "Size");
                mesh.feature("ftri_surface").feature("size1").selection().geom("geom");
                mesh.feature("ftri_surface").feature("size1")
                                .set("custom", "on")
                                .set("hmax", "g_c * 2.0")
                                .set("hmaxactive", true)
                                .set("hmin", "g_c / 2.0")
                                .set("hminactive", true)
                                .label("size");

                mesh.create("refine_gap", "Refine").label("gap_refine");
                mesh.feature("refine_gap").selection().geom("geom", 2);
                mesh.feature("refine_gap").selection().set(selections.get("gap"));
                mesh.feature("refine_gap").set("rmethod", "regular");

                mesh.create("ftet1", "FreeTet").label("remainder");
                mesh.feature("ftet1").create("size1", "Size");
                mesh.feature("ftet1").feature("size1")
                                .set("custom", "on")
                                .set("hmax", "lda_c / 8 / n_c")
                                .set("hmaxactive", true)
                                .set("hmin", "g_c / 2.0")
                                .set("hminactive", true)
                                .label("size");
        }

        public static void setupMaterials(Model model, ModelNode component, Properties prop,
                        Map<String, int[]> selections) {
                model.func().create("int2", "Interpolation");
                model.func("int2")
                                .set("source", "file")
                                .set("importedname", "lsat_datafile")
                                .set("importedstruct", "Spreadsheet")
                                .set("importeddim", "1D")
                                .set("funcs", new String[][] { { "lsat_eps1", "1" }, { "lsat_eps2", "2" } })
                                .set("fununit", new String[] { "", "" })
                                .set("argunit", new String[] { "Hz" })
                                .set("filename", prop.getProperty("lsat_datafile"))
                                .set("nargs", "1")
                                .set("struct", "spreadsheet")
                                .label("LSAT");
                model.func("int2").importData();

                model.func().create("int1", "Interpolation");
                model.func("int1")
                                .set("source", "file")
                                .set("importedname", "gold_datafile")
                                .set("importedstruct", "Spreadsheet")
                                .set("importeddim", "1D")
                                .set("funcs", new String[][] { { "au_eps1", "1" }, { "au_eps2", "2" } })
                                .set("fununit", new String[] { "", "" })
                                .set("argunit", new String[] { "Hz" })
                                .set("filename", prop.getProperty("gold_datafile"))
                                .set("nargs", "1")
                                .set("struct", "spreadsheet")
                                .label("Gold");
                model.func("int1").importData();

                // Materials

                String[] unitMatrix = { "1", "0", "0", "0", "1", "0", "0", "0", "1" };
                String[] zeroMatrix = { "0", "0", "0", "0", "0", "0", "0", "0", "0" };

                Material airMaterial = component.material().create("air_mat", "Common");
                airMaterial
                                .set("family", "air")
                                .label("Vacuum");
                airMaterial.propertyGroup("def")
                                .set("relpermeability", unitMatrix)
                                .set("relpermittivity", unitMatrix)
                                .set("electricconductivity", zeroMatrix);

                Material substrateMaterial = component.material().create("sub_mat", "Common");
                substrateMaterial.selection().set(selections.get("substrate"));
                substrateMaterial.label("Substrate");
                substrateMaterial.propertyGroup("def")
                                .set("relpermittivity", new String[] {
                                                "lsat_eps1(freq)-j*lsat_eps2(freq)", "0", "0",
                                                "0", "lsat_eps1(freq)-j*lsat_eps2(freq)", "0",
                                                "0", "0", "lsat_eps1(freq)-j*lsat_eps2(freq)" })
                                .set("electricconductivity", zeroMatrix)
                                .set("relpermeability", unitMatrix);

                Material resonatorMaterial = component.material().create("res_mat", "Common");
                resonatorMaterial.selection().geom("geom", 2);
                resonatorMaterial.selection().set(selections.get("resonator"));
                resonatorMaterial.label("Resonator");
                resonatorMaterial.propertyGroup("def")
                                .set("relpermittivity", new String[] {
                                                "au_eps1(freq)-j*au_eps2(freq)", "0", "0",
                                                "0", "au_eps1(freq)-j*au_eps2(freq)", "0",
                                                "0", "0", "au_eps1(freq)-j*au_eps2(freq)" })
                                .set("relpermeability", unitMatrix)
                                .set("electricconductivity", zeroMatrix);

                Material filmMaterial = component.material().create("film_mat", "Common");
                filmMaterial.selection().geom("geom", 2);
                filmMaterial.selection().set(selections.get("film"));
                filmMaterial.label("Film");
                filmMaterial.propertyGroup("def")
                                .set("relpermittivity", unitMatrix)
                                .set("relpermeability", unitMatrix)
                                .set("electricconductivity", zeroMatrix);

                Material gapMaterial = component.material().create("gap_mat", "Common");
                gapMaterial.label("Gap");
                gapMaterial.selection().geom("geom", 2);
                gapMaterial.selection().set(selections.get("gap"));
                gapMaterial.propertyGroup("def")
                                .set("relpermittivity", unitMatrix)
                                .set("relpermeability", unitMatrix)
                                .set("electricconductivity", zeroMatrix);
        }

        public static void setupPhysics(ModelNode component, Map<String, int[]> selections) {

                component.coordSystem().create("pml1", "PML");
                component.coordSystem().create("pml2", "PML");
                component.coordSystem("pml1").selection().set(4);
                component.coordSystem("pml2").selection().set(1);
                component.coordSystem("sys1").set("name", "sys").label("Boundary System");
                component.coordSystem("pml1").set("name", "pml_top").label("pml_top");
                component.coordSystem("pml2").set("name", "pml_bottom").label("pml_bottom");

                component.variable().create("var2").label("variables");
                component.variable("var2")
                                .set("k_x", "emw.k0*sin(theta)*cos(phi)", "kx for incident wave")
                                .set("k_y", "emw.k0*sin(theta)*sin(phi)", "ky for incident wave")
                                .set("k_z", "emw.k0*cos(theta)", "kz for incident wave");

                component.physics().create("emw", "ElectromagneticWaves", "geom").label("Electromagnetics");
                component.physics("emw").field("electricfield").field("E");
                component.physics("emw").field("electricfield").component(new String[] { "Ex", "Ey", "Ez" });

                component.physics("emw").feature("pec1").label("pec");
                component.physics("emw").feature("init1").label("initial_values");

                component.physics("emw").create("trans1", "TransitionBoundaryCondition", 2);
                component.physics("emw").feature("trans1").set("d", "200[nm]").set("noCoupling", true)
                                .label("tbc_resonator");
                component.physics("emw").feature("trans1").selection().set(selections.get("resonator"));

                component.physics("emw").create("trans2", "TransitionBoundaryCondition", 2);
                component.physics("emw").feature("trans2").set("d", "11[nm]").label("tbc_nickelate");
                component.physics("emw").feature("trans2").selection().set(selections.get("film"));

                component.physics("emw").create("trans3", "TransitionBoundaryCondition", 2);
                component.physics("emw").feature("trans3").set("d", "11[nm]").label("tbc_gap");
                component.physics("emw").feature("trans3").selection().set(selections.get("gap"));

                component.physics("emw").create("port1", "Port", 2);
                component.physics("emw").feature("port1").selection().set(selections.get("port_emitter"));
                component.physics("emw").feature("port1")
                                .set("PortSlit", true)
                                .set("SlitType", "DomainBacked")
                                .set("PortOrientation", "ReversePort")
                                .set("InputType", "E")
                                .set("E0", new String[] { "0", "exp(-i*k_x*x) * exp(-i*k_y*y)", "0" })
                                .set("beta", "abs(k_z)")
                                .label("port_emitter");

                component.physics("emw").create("port2", "Port", 2);
                component.physics("emw").feature("port2").selection().set(selections.get("port_receiver"));
                component.physics("emw").feature("port2")
                                .set("PortSlit", true)
                                .set("SlitType", "DomainBacked")
                                .set("E0", new String[] { "0", "exp(-i*k_x*x) * exp(-i*k_y*y)", "0" })
                                .set("beta", "abs(k_z)")
                                .label("port_receiver");

                component.physics("emw").create("pmc1", "PerfectMagneticConductor", 2).label("pmc");
                component.physics("emw").feature("pmc1").selection().set(selections.get("pmc"));

                component.physics("emw").create("sctr2", "Scattering", 2);
                component.physics("emw").feature("sctr2").selection().set(selections.get("sbc"));
                component.physics("emw").feature("sctr2").label("sbc");
        }

        public static void createSolver(Model model) {

                Study study = model.study().create("std1");
                study.create("freq", "Frequency");

                SolverSequence solver = model.sol().create("sol1");
                solver.study("std1");
                solver.attach("std1");

                SolverFeature studyStep = solver.create("st1", "StudyStep");
                studyStep.label("Compile Equations: Frequency Domain");

                SolverFeature variables = solver.create("v1", "Variables");
                variables
                                .set("clistctrl", new String[] { "p1" })
                                .set("cname", new String[] { "freq" })
                                .set("clist", new String[] { "1[THz]" })
                                .label("Dependent Variables 1.1");

                SolverFeature stationary = solver.create("s1", "Stationary");
                stationary
                                .set("stol", 0.01)
                                .set("plot", true)
                                .label("Stationary Solver 1.1");
                stationary.feature("dDef").label("Direct 2");
                stationary.feature("aDef")
                                .set("complexfun", true)
                                .label("Advanced 1");

                stationary.create("d1", "Direct")
                                .set("linsolver", "pardiso")
                                .label("Suggested Direct Solver (emw)");

                stationary.create("fc1", "FullyCoupled")
                                .set("linsolver", "d1")
                                .label("Fully Coupled 1.1");

                SolverFeature iterative = stationary.create("i1", "Iterative");
                iterative
                                .set("itrestart", 300)
                                .set("prefuntype", "right")
                                .label("Suggested Iterative Solver (emw)");
                iterative.feature("ilDef").label("Incomplete LU 1");

                SolverFeature multigrid = iterative.create("mg1", "Multigrid");
                multigrid
                                .set("iter", 1)
                                .label("Multigrid 1.1");

                SolverFeature preSmoother = multigrid.feature("pr");
                preSmoother.label("Presmoother 1");
                preSmoother.feature("soDef").label("SOR 1");

                SolverFeature vanka = preSmoother.create("va1", "Vanka");
                vanka
                                .set("iter", 1)
                                .set("vankavars", new String[] { "component_E" })
                                .set("vankasolv", "stored")
                                .set("vankarelax", 0.95)
                                .label("Vanka 1.1");

                SolverFeature postSmoother = multigrid.feature("po");
                postSmoother.label("Postsmoother 1");
                postSmoother.feature("soDef").label("SOR 1");

                SolverFeature sorVector = postSmoother.create("sv1", "SORVector");
                sorVector
                                .set("iter", 1)
                                .set("relax", 0.5)
                                .label("SOR Vector 1.1");

                stationary.create("p1", "Parametric")
                                .set("pname", new String[] { "freq" })
                                .set("plistarr", new String[] { "1[THz]" })
                                .set("punit", new String[] { "THz" })
                                .set("pcontinuationmode", "no")
                                .set("preusesol", "auto")
                                .set("plot", true)
                                .label("Parametric 1.1");

                study.label("Frequency study");
                study.feature("freq").set("punit", "THz");
                study.feature("freq").set("plist", "1[THz]");

                SolverFeature coarseSolver = multigrid.feature("cs");
                coarseSolver.label("Coarse Solver 1");
                coarseSolver.feature("dDef").label("Direct 1");
        }

        public static void createExports(Model model) {

                model.result().table().create("params_table", "Table").label("parameters");
                model.result().dataset().create("surf1", "Surface").label("Surface");
                model.result().dataset("surf1").selection().named("geom_sel1");

                NumericalFeature gapProbe = model.result().numerical().create("gap_point", "EvalPoint");
                gapProbe
                                .set("probetag", "none").set("table", "params_table")
                                .set("expr", new String[] {
                                                "real(emw.Ey)", "imag(emw.Ey)",
                                                "real(emw.sigmabnd)", "imag(emw.sigmabnd)" })
                                .set("unit", new String[] {
                                                "V/m", "V/m",
                                                "S/m", "S/m" })
                                .set("descr", new String[] {
                                                "Ey.real", "Ey.imag",
                                                "cond.real", "cond.imag" })
                                .label("gap");
                gapProbe.selection().named("geom_ballsel1");

                NumericalFeature evalGlobal = model.result().numerical().create("gev1", "EvalGlobal");
                evalGlobal
                                .set("probetag", "none")
                                .set("table", "params_table")
                                .set("expr", new String[] {
                                                "real(emw.S11)", "imag(emw.S11)",
                                                "real(emw.S21)", "imag(emw.S21)" })
                                .set("unit", new String[] { "1", "1", "1", "1" })
                                .set("descr", new String[] {
                                                "s11.real", "s11.imag",
                                                "s21.real", "s21.imag" })
                                .label("global");

                ExportFeature parametersExport = model.result().export().create("params_export", "Table");
                parametersExport
                                .set("header", false)
                                .set("ifexists", "append")
                                .label("parameters");

                ExportFeature spatialExport = model.result().export().create("spatial_data", "Data");
                spatialExport
                                .set("data", "surf1")
                                .set("expr", new String[] {
                                                "real(emw.Ex)", "imag(emw.Ex)",
                                                "real(emw.Ey)", "imag(emw.Ey)",
                                                "real(emw.Hz)", "imag(emw.Hz)",
                                                "real(emw.Jsupx)", "imag(emw.Jsupx)",
                                                "real(emw.Jsupy)", "imag(emw.Jsupy)",
                                                "real(emw.sigmabnd)", "imag(emw.sigmabnd)" })
                                .set("descr", new String[] {
                                                "Ex.real", "Ex.imag",
                                                "Ey.real", "Ey.imag",
                                                "Hz.real", "Hz.imag",
                                                "Jsupx.real", "Jsupx.imag",
                                                "Jsupy.real", "Jsupy.imag",
                                                "cond.real", "cond.imag" })
                                .set("unit", new String[] {
                                                "V/m", "V/m",
                                                "V/m", "V/m",
                                                "A/m", "A/m",
                                                "A/m", "A/m",
                                                "A/m", "A/m",
                                                "S/m", "S/m" })
                                .set("location", "regulargrid")
                                .set("regulargridx2", 250)
                                .set("regulargridy2", 250)
                                .set("header", false)
                                .set("fullprec", false)
                                .set("ifexists", "append")
                                .set("separator", ",")
                                .label("surface_solution");
        }
}