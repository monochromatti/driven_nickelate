import com.comsol.model.*;
import com.comsol.model.physics.Physics;
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

        String[] required_keys = {"gold_datafile", "lsat_datafile", "cadfile", "pump_spectrum"};
        for (String key : required_keys) {
            if (!prop.containsKey(key)) {
                System.out.println("Error: property file does not contain key '" + key + "'.");
                System.exit(1);
            }
        }
        Map<String, int[]> selections = initializeSelections();
        try {
            System.out.println("Creating model ...");

            Model model = createModel();
            ModelNode component = model.component().create("component", true);
            GeomSequence geometry = component.geom().create("geom", 3);
            MeshSequence mesh = component.mesh().create("mesh");

            System.out.println("Building geometry ...");
            buildGeometry(geometry, prop.getProperty("cadfile"));

            System.out.println("Building mesh ...");
            buildMesh(mesh, selections);

            System.out.println("Setting up materials ...");
            defineMaterials(model, prop);
            assignMaterials(component, selections);

            System.out.println("Setting up physics ...");
            setupPhysics(model, component, selections, prop);

            System.out.println("Creating solver ...");
            createSolver(model);

            System.out.println("Creating exports ...");
            createExports(model, selections);

            try {
                model.save("model.mph");
                System.out.println("Model saved to model.mph");
                System.exit(0);
            } catch (IOException e) {
                System.out.println("Error saving model");
                System.exit(1);
            }
        } catch (Exception e) {
            System.out.println("Error creating model");
            e.printStackTrace();
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

    public static Model createModel() {
        Model model = ModelUtil.create("Model");
        model.param()
                .set("lda_c", "c_const/2[THz]", "Characteristic wavelength")
                .set("g_c", "600[nm]", "Minimal feature size")
                .set("n_c", "4.8", "Characteristic substrate index")
                .set("z_up", "100[um]", "Upward extrusion")
                .set("z_down", "z_up/n_c", "Downward extrusion")
                .set("d_NNO", "12[nm]", "NNO film thickness")
                .set("d_Au", "200[nm]", "Gold film thickness")
                .set("d_STO", "2[nm]", "STO capping thickness")
                .label("Characteristic quantities");
        return model;
    }

    public static Map<String, int[]> initializeSelections() {
        Map<String, int[]> selections = new HashMap<>();

        Object[][] data = {
                {"resonator", new int[]{ 4 }},
                {"film", new int[]{17, 22}},
                {"gap_down", new int[]{ 6 }},
                {"gap_up", new int[]{ 9 }},
                {"gap_bulk", new int[]{ 2 }},
                {"gap_line", new int[]{ 6 }},
                {"sbc_source", new int[]{ 10 }},
                {"sbc_drain", new int[]{ 3 }},
                {"pmc", new int[]{1, 4, 7, 11, 38, 39}},
                {"substrate", new int[]{ 1 }},
                {"vacuum", new int[]{ 3 }},
        };

        for (Object[] entry : data) {
            selections.put((String) entry[0], (int[]) entry[1]);
        }

        return selections;
    }


    public static void buildGeometry(GeomSequence geometry, String cadfile) {

        geometry.lengthUnit("µm");
        geometry.geomRep("comsol");
        geometry.create("wp_extrusion", "WorkPlane")
                .set("selresult", true)
                .set("selplaneshow", true)
                .set("unite", true)
                .label("wp_extrusion");
        geometry.feature("wp_extrusion").geom().create("import", "Import")
                .set("type", "dxf")
                .set("filename", cadfile)
                .set("alllayers", new String[]{"0", "structure", "gap", "unit_cell"})
                .set("layerselection", "selected")
                .set("layers", new String[]{"unit_cell"});
        geometry.feature("wp_extrusion").geom().create("quadrant", "Square");
        geometry.feature("wp_extrusion").geom().feature("quadrant").set("size", 100).label("wp_extrusion");
        geometry.feature("wp_extrusion").geom().create("int1", "Intersection");
        geometry.feature("wp_extrusion").geom().feature("int1").selection("input").set("import", "quadrant");
        geometry.create("ext1", "Extrude")
                .set("inputhandling", "keep")
                .set("distance", new String[]{"z_up", "-z_down"})
                .set("scale", new double[][]{{1, 1}, {1, 1}})
                .set("displ", new double[][]{{0, 0}, {0, 0}})
                .set("twist", new int[]{0, 0})
                .label("extrude_domain");
        geometry.feature("ext1").selection("input").named("wp_extrusion");
        geometry.create("wp_structure", "WorkPlane")
                .set("selresult", true)
                .set("selplaneshow", true)
                .set("unite", true)
                .label("wp_structure");
        geometry.feature("wp_structure").geom().label("dxf_plane");
        geometry.feature("wp_structure").geom().create("import", "Import");
        geometry.feature("wp_structure").geom().feature("import")
                .set("type", "dxf")
                .set("filename", cadfile)
                .set("alllayers", new String[]{"0", "structure", "gap", "unit_cell"})
                .set("layerselection", "selected")
                .set("layers", new String[]{"structure", "gap"})
                .label("import_dxf");

        geometry.feature("wp_structure").geom().create("pt1", "Point");
        geometry.feature("wp_structure").geom().feature("pt1").label("gap_centerpoint");
        geometry.feature("wp_structure").geom().create("union", "Union");
        geometry.feature("wp_structure").geom().feature("union").label("union_dxf");
        geometry.feature("wp_structure").geom().feature("union").selection("input").set("import(1)", "import(2)");
        geometry.feature("wp_structure").geom().create("quadrant", "Square");
        geometry.feature("wp_structure").geom().feature("quadrant").label("quadrant");
        geometry.feature("wp_structure").geom().feature("quadrant").set("size", 100);
        geometry.feature("wp_structure").geom().create("int1", "Intersection");
        geometry.feature("wp_structure").geom().feature("int1").label("quadrant_intersection");
        geometry.feature("wp_structure").geom().feature("int1").selection("input").set("quadrant", "union");

        GeomFeature gapExtrude = geometry.create("ext2", "Extrude");
        gapExtrude
                .set("extrudefrom", "faces")
                .setIndex("distance", "d_NNO", 0)
                .label("gap_extrude");
        gapExtrude.selection("inputface").set("wp_structure.uni", 1);

        GeomFeature resExtrude = geometry.create("ext3", "Extrude");
        resExtrude
                .set("extrudefrom", "faces")
                .setIndex("distance", "d_Au", 0)
                .label("resonator_extrude");
        resExtrude.selection("inputface").set("ext2(1)", 6);

        geometry.run();
    }

    public static void buildMesh(MeshSequence mesh, Map<String, int[]> selections) {
        mesh.feature("size").set("hauto", 3);

        MeshFeature ftetGap = mesh.create("ftet1", "FreeTet");
        ftetGap.label("gap");
        ftetGap.selection().geom("geom", 3);
        ftetGap.selection().set(2);

        ftetGap.create("size1", "Size");
        ftetGap.feature("size1").selection().geom("geom");
        ftetGap.feature("size1")
                .set("custom", "on")
                .set("hmax", "40[nm]")
                .set("hmaxactive", true)
                .set("hmin", "10[nm]")
                .set("hminactive", true)
                .label("size");

        MeshFeature ftriResonatorEdge = mesh.create("ftri1", "FreeTri");
        ftriResonatorEdge.label("resonator_edge");
        ftriResonatorEdge.selection().set(14);

        ftriResonatorEdge.create("size1", "Size");
        ftriResonatorEdge.feature("size1").selection().geom("geom");
        ftriResonatorEdge.feature("size1")
                .set("custom", "on")
                .set("hmax", "40[nm]")
                .set("hmaxactive", true)
                .set("hmin", "10[nm]")
                .set("hminactive", true)
                .label("size");

        MeshFeature ftetRemaining = mesh.create("ftet_remaining", "FreeTet");
        ftetRemaining.label("remainder");
        ftetRemaining.create("size1", "Size");
        ftetRemaining.feature("size1")
                .set("custom", "on")
                .set("hmax", "(lda_c/n_c) / 10")
                .set("hmaxactive", true)
                .set("hmin", "50[nm]")
                .set("hminactive", true)
                .set("hgrad", 1.15)
                .set("hgradactive", false)
                .label("size");
        mesh.run();
    }

    public static void defineMaterials(Model model, Properties prop) {
        model.func().create("int1", "Interpolation");
        model.func("int1")
                .set("source", "file")
                .set("importedname", "gold.csv")
                .set("importedstruct", "Spreadsheet")
                .set("importeddim", "1D")
                .set("funcs", new String[][]{{"au_eps1", "1"}, {"au_eps2", "2"}})
                .set("fununit", new String[]{"", ""})
                .set("argunit", new String[]{"Hz"})
                .set("filename", prop.getProperty("gold_datafile"))
                .set("nargs", "1")
                .set("struct", "spreadsheet")
                .label("Gold");
        model.func("int1").importData();

        model.func().create("int2", "Interpolation");
        model.func("int2")
                .set("source", "file")
                .set("importedname", "lsat.csv")
                .set("importedstruct", "Spreadsheet")
                .set("importeddim", "1D")
                .set("funcs", new String[][]{{"lsat_eps1", "1"}, {"lsat_eps2", "2"}})
                .set("fununit", new String[]{"", ""})
                .set("argunit", new String[]{"THz"})
                .set("filename", prop.getProperty("lsat_datafile"))
                .set("nargs", "1")
                .set("struct", "spreadsheet")
                .label("LSAT");
        model.func("int2").importData();

        String[] unitMatrix = {"1", "0", "0", "0", "1", "0", "0", "0", "1"};
        String[] zeroMatrix = {"0", "0", "0", "0", "0", "0", "0", "0", "0"};

        model.material().create("vacuum", "Common", "");
        model.material("vacuum").set("family", "air").label("vacuum");
        model.material("vacuum").propertyGroup("def")
                .set("relpermeability", unitMatrix)
                .set("relpermittivity", unitMatrix)
                .set("electricconductivity", zeroMatrix);

        model.material().create("au", "Common", "");
        model.material("au").label("au");
        model.material("au").propertyGroup("def")
                .set("relpermittivity", new String[]{
                        "au_eps1(freq)-j*au_eps2(freq)", "0", "0",
                        "0", "au_eps1(freq)-j*au_eps2(freq)", "0",
                        "0", "0", "au_eps1(freq)-j*au_eps2(freq)"})
                .set("relpermeability", unitMatrix)
                .set("electricconductivity", zeroMatrix);

        model.material().create("ndnio3", "Common", "");
        model.material("ndnio3").label("ndnio3");
        model.material("ndnio3").propertyGroup("def")
                .set("relpermittivity", unitMatrix)
                .set("relpermeability", unitMatrix)
                .set("electricconductivity", zeroMatrix);

        model.material().create("ndnio3_gap", "Common", "");
        model.material("ndnio3_gap").label("ndnio3_gap");
        model.material("ndnio3_gap").propertyGroup("def")
                .set("relpermittivity", unitMatrix)
                .set("relpermeability", unitMatrix)
                .set("electricconductivity", zeroMatrix);

        model.material().create("lsat", "Common", "");
        model.material("lsat").label("lsat");
        model.material("lsat").propertyGroup("def")
                .set("relpermittivity", new String[]{
                        "lsat_eps1(freq)-j*lsat_eps2(freq)", "0", "0",
                        "0", "lsat_eps1(freq)-j*lsat_eps2(freq)", "0",
                        "0", "0", "lsat_eps1(freq)-j*lsat_eps2(freq)"})
                .set("electricconductivity", zeroMatrix)
                .set("relpermeability", unitMatrix);

        model.material().create("sto", "Common", "").label("sto");
        model.material("sto").propertyGroup("def")
                .set("electricconductivity", zeroMatrix);
        model.material("sto").propertyGroup("def")
                .set("relpermittivity", new String[]{"100", "0", "0", "0", "100", "0", "0", "0", "100"});
        model.material("sto").propertyGroup("def")
                .set("relpermeability", unitMatrix);
        model.material("sto").propertyGroup().create("shell", "Shell");
        model.material("sto").propertyGroup("shell").set("lth", "d_STO");

    }

    public static void assignMaterials(ModelNode component, Map<String, int[]> selections) {
        component.material().create("matlnk_vacuum", "Link").label("air");
        component.material("matlnk_vacuum").selection().set(selections.get("vacuum"));
        component.material("matlnk_vacuum").set("link", "vacuum");

        component.material().create("matlnk_substrate", "Link").label("substrate");
        component.material("matlnk_substrate").selection().set(selections.get("substrate"));
        component.material("matlnk_substrate").set("link", "lsat");

        component.material().create("matlnk_film", "Link").label("film");
        component.material("matlnk_film").set("link", "ndnio3");
        component.material("matlnk_film").selection().geom("geom", 2);
        component.material("matlnk_film").selection().set(selections.get("film"));

        component.material().create("matlnk_gap", "Link").label("gap");
        component.material("matlnk_gap").set("link", "ndnio3_gap");
        component.material("matlnk_gap").selection().geom("geom", 3);
        component.material("matlnk_gap").selection().set(selections.get("gap_bulk"));

        component.material().create("sto_capping", "LayeredMaterialLink").label("sto_capping");
        component.material("sto_capping").selection().set(selections.get("gap_up"));
        component.material("sto_capping").set("link", "sto");
        component.material("sto_capping").set("middlePlane", "top");
        component.material("sto_capping").set("offset", -1);

        component.material().create("resonator", "Link").label("resonator");
        component.material("resonator").selection().set(selections.get("resonator"));
        component.material("resonator").set("link", "au");

    }

    public static void setupPhysics(Model model, ModelNode component, Map<String, int[]> selections, Properties prop) {
        model.func().create("int3", "Interpolation");
        model.func("int3")
                .set("source", "file")
                .set("importedname", "pump_spectrum")
                .set("importedstruct", "Spreadsheet")
                .set("importeddim", "1D")
                .set("funcs", new String[][] { { "pump_real", "1" }, { "pump_imag", "2" } })
                .set("interp", "cubicspline")
                .set("extrap", "value")
                .set("fununit", new String[] { "V/m", "V/m" })
                .set("argunit", new String[] { "THz" })
                .set("filename", prop.getProperty("pump_spectrum"))
                .set("nargs", "1")
                .set("struct", "spreadsheet")
                .label("pump_field");
        model.func("int3").importData();

        // Physics
        component.coordSystem("sys1").label("Boundary System");
        component.coordSystem("sys1").set("name", "sys");

        Physics emw = component.physics().create("emw", "ElectromagneticWaves", "geom");
        emw.field("electricfield").field("E2");
        emw.field("electricfield").component(new String[]{"E2x", "E2y", "E2z"});

        emw.feature("pec1").label("pec");
        emw.feature("init1").label("initial_values");

        ParameterEntity pmc = emw.create("pmc1", "PerfectMagneticConductor", 2);
        pmc.label("pmc");
        pmc.selection().set(selections.get("pmc"));

        ParameterEntity tbc_film = emw.create("trans1", "TransitionBoundaryCondition", 2);
        tbc_film
                .set("d", "d_NNO")
                .label("tbc_nickelate");
        tbc_film.selection().set(selections.get("film"));

        ParameterEntity tbc_capping = emw.create("trans2", "TransitionBoundaryCondition", 2);
        tbc_capping
                .set("d", "d_STO")
                .label("tbc_capping");
        tbc_capping.selection().set(selections.get("gap_up"));

        
        String field_expr = "(pump_real(emw.freq) + j*pump_imag(emw.freq)) * exp(-j*emw.k*z)";

        emw.prop("BackgroundField")
                .set("SolveFor", "scatteredField")
                .set("Eb", new String[]{"0", field_expr, "0"});

        ParameterEntity sbc_source = emw.create("sctr1", "Scattering", 2);
        sbc_source
                .set("Order", "SecondOrder")
//                .set("IncidentField", "EField")
//                .set("E0i", new String[]{"0", field_expr, "0"})
                .label("sbc_source");
        sbc_source.selection().set(selections.get("sbc_source"));

        ParameterEntity sbc_drain = emw.create("sctr2", "Scattering", 2);
        sbc_drain
                .set("Order", "SecondOrder")
                .label("sbc_drain");
        sbc_drain.selection().set(selections.get("sbc_drain"));
    }

    public static void createSolver(Model model) {
        Study std = model.study().create("std1");
        std.create("freq", "Frequency");
        std.label("Frequency study");
        std.feature("freq").set("punit", "THz");
        std.feature("freq").set("plist", "1[THz]");

        SolverSequence sol = model.sol().create("sol1");
        sol.study("std1");
        sol.attach("std1");
        sol.attach("std1");
        sol.create("st1", "StudyStep");
        sol.feature("st1").label("Compile Equations: Frequency Domain");

        SolverFeature variables = sol.create("v1", "Variables");
        variables.label("Dependent Variables 1.1");

        SolverFeature stationary = sol.create("s1", "Stationary");
        stationary.label("Stationary Solver 1.1");
        stationary.set("stol", 0.01);
        stationary.set("plot", true);
        stationary.feature("aDef").label("Advanced 1");
        stationary.feature("aDef").set("complexfun", true);

        SolverFeature iterative = stationary.create("i1", "Iterative");
        iterative.label("Suggested Iterative Solver (emw)");
        iterative.set("itrestart", 300);
        iterative.set("prefuntype", "right");
        iterative.feature("ilDef").label("Incomplete LU 1");

        SolverFeature fullyCoupled = stationary.create("fc1", "FullyCoupled");
        fullyCoupled.label("Fully Coupled 1.1");

        SolverFeature parametric = stationary.create("p1", "Parametric");
        parametric.label("Parametric 1.1");
        parametric.set("control", "freq");

        SolverFeature multigrid = stationary.feature("i1").create("mg1", "Multigrid");
        multigrid.label("Multigrid 1.1");
        multigrid.set("iter", 1);
        multigrid.feature("pr").label("Presmoother 1");
        multigrid.feature("pr").feature("soDef").label("SOR 1");

        multigrid.feature("pr").create("sv1", "SORVector");
        multigrid.feature("pr").feature("sv1").label("SOR Vector 1.1");
        multigrid.feature("pr").feature("sv1").set("sorvecdof", new String[]{"component_E2"});

        multigrid.feature("po").create("sv1", "SORVector");
        multigrid.feature("po").label("Postsmoother 1");
        multigrid.feature("po").feature("soDef").label("SOR 1");
        multigrid.feature("po").feature("sv1").label("SOR Vector 1.1");
        multigrid.feature("po").feature("sv1").set("sorvecdof", new String[]{"component_E2"});

        multigrid.feature("cs").label("Coarse Solver 1");
        multigrid.feature("cs").feature("dDef").label("Direct 2");

        multigrid.feature("cs").create("d1", "Direct");
        multigrid.feature("cs").feature("d1").label("Direct 1.1");
        multigrid.feature("cs").feature("d1").set("linsolver", "mumps");

        stationary.feature().remove("dDef");
        stationary.feature().remove("fcDef");
        stationary.feature().remove("pDef");

        Study std_ifft = model.study().create("std2");
        std_ifft.create("ftfft", "FreqToTimeFFT");
        std_ifft.feature("ftfft").set("fftinputstudy", "std1");
        std_ifft.feature("ftfft").set("tunit", "ps");
        std_ifft.feature("ftfft").set("fftouttrange", "range(-5,6e-2,15)");

        SolverSequence sol_ifft = model.sol().create("sol2");
        sol_ifft.study("std2");
        sol_ifft.create("st1", "StudyStep");
        sol_ifft.feature("st1").set("study", "std2");
        sol_ifft.feature("st1").set("studystep", "ftfft");
        sol_ifft.create("v1", "Variables");
        sol_ifft.feature("v1").set("control", "ftfft");
        sol_ifft.create("fft1", "FFT");
        sol_ifft.feature("fft1").set("fftsteptypef", "allfreqs");
        sol_ifft.feature("fft1").set("fftextend", "on");
        sol_ifft.feature("fft1").set("ffttranstype", "transifft");
        sol_ifft.feature("fft1").set("fftbwalgtype", "fastfft");
        sol_ifft.feature("fft1").set("control", "ftfft");
        sol_ifft.attach("std2");
    }

    public static void createExports(Model model, Map<String, int[]> selections) {
        model.result().table().create("params_table", "Table").label("parameters");
        model.result().dataset().create("sample_surface", "Surface").label("Surface");
        model.result().dataset("sample_surface").selection().set(selections.get("surface"));

        NumericalFeature volumeInt = model.result().numerical().create("gap_integral", "IntVolume");
        volumeInt.selection().set(selections.get("gap_bulk"));
        volumeInt
                .set("table", "params_table")
                .set("expr", new String[]{"0.5 * realdot(emw.Ey, emw.Jy)", "emw.Qrh"})
                .set("unit", new String[]{"W", "W"})
                .set("descr", new String[]{"Qrhy", "Qrh"})
                .label("gap_volume");

        NumericalFeature lineAvg = model.result().numerical().create("line_average", "AvLine");
        lineAvg.set("intsurface", true);
        lineAvg.selection().set(selections.get("gap_line"));
        lineAvg
                .set("table", "params_table")
                .set("expr", new String[]{"emw.sigma_iso", "emw.Ey", "emw.Jy", "0.5 * realdot(emw.Ey, emw.Jy)", "emw.Qrh"})
                .set("unit", new String[]{"S/m", "V/m", "A/m^2", "W/m^3", "W/m^3"})
                .set("descr", new String[]{"cond", "Ey", "Jy", "qrhy", "qrh"})
                .label("gap_line");

        ExportFeature parametersExport = model.result().export().create("params_export", "Table");
        parametersExport
                .set("ifexists", "append")
                .set("prec", "manual")
                .set("manualprec", 3)
                .label("parameters");
    }
}
