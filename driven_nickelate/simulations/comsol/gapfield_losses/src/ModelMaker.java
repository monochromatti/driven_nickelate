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
                .set("z_up", "150[um]", "Upward extrusion")
                .set("z_down", "z_up/n_c", "Downward extrusion")
                .label("Characteristic quantities");
        return model;
    }

    public static Map<String, int[]> initializeSelections() {
        Map<String, int[]> selections = new HashMap<>();

        Object[][] data = {
                {"surface", new int[]{6, 8, 9, 12}},
                {"outer_surface", new int[]{ 9 }},
                {"inner_surface", new int[]{ 12 }},
                {"resonator", new int[]{ 8 }},
                {"film", new int[]{9, 12}},
                {"gap", new int[]{ 6 }},
                {"sbc_source", new int[]{ 7 }},
                {"sbc_drain", new int[]{ 3 }},
                {"pmc", new int[]{1, 4, 13, 14}},
                {"substrate", new int[]{1}},
                {"vacuum", new int[]{2}},
                {"resonator_edge", new int[]{12, 18}}
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

        geometry.create("ballsel1", "BallSelection");
        geometry.feature("ballsel1")
                .set("entitydim", 0)
                .set("r", 0.1)
                .set("condition", "inside")
                .label("gapcenter_ballsel");

        geometry.create("filmsel1", "ExplicitSelection").label("filmcenter_selection");
        geometry.feature("filmsel1").selection("selection").init(0);
        geometry.feature("filmsel1").selection("selection").set("ext1", 11);

        geometry.create("igv1", "IgnoreVertices");
        geometry.feature("igv1").selection("input").named("wp_structure");

        geometry.run();
    }

    public static void buildMesh(MeshSequence mesh, Map<String, int[]> selections) {

        mesh.feature("size").set("custom", "on").set("hmax", "lda_c/10").set("hmin", "lda_c/1E4").set("hauto", 3);

        MeshFeature tri_gap = mesh.create("gap", "FreeTri");
        tri_gap.label("gap");
        tri_gap.selection().set(selections.get("gap"));
        tri_gap.create("size1", "Size");
        tri_gap.feature("size1").selection().geom("geom");
        tri_gap.feature("size1")
                .set("custom", "on")
                .set("hmax", "100[nm]")
                .set("hmaxactive", true)
                .set("hmin", "10[nm]")
                .set("hminactive", true)
                .label("size");

        MeshFeature resonator_edge = mesh.create("edg1", "Edge");
        resonator_edge.label("resonator_edge");
        resonator_edge.selection().set(selections.get("resonator_edge"));
        resonator_edge.create("size1", "Size");
        resonator_edge.feature("size1")
                .set("custom", "on")
                .set("hmax", "200[nm]")
                .set("hmaxactive", true);

        MeshFeature tri_resonator = mesh.create("resonator", "FreeTri");
        tri_resonator.label("resonator");
        tri_resonator.selection().set(selections.get("resonator"));
        tri_resonator.create("size1", "Size");
        tri_resonator.feature("size1").selection().geom("geom");
        tri_resonator.feature("size1")
                .set("custom", "on")
                .set("hmax", "600[nm]")
                .set("hmaxactive", true)
                .set("hmin", "50[nm]")
                .set("hminactive", true)
                .label("size");

        MeshFeature tri_inner = mesh.create("inner_surface", "FreeTri");
        tri_inner.label("inner_surface");
        tri_inner.selection().set(selections.get("inner_surface"));
        tri_inner.create("size1", "Size");
        tri_inner.feature("size1").selection().geom("geom");
        tri_inner.feature("size1")
                .set("custom", "on")
                .set("hmax", "600[nm]")
                .set("hmaxactive", true)
                .set("hmin", "10[nm]")
                .set("hminactive", true)
                .label("size");

        MeshFeature mesh_remaining = mesh.create("ftet_remaining", "FreeTet");
        mesh_remaining.create("size1", "Size");
        mesh_remaining.label("remainder");
        mesh_remaining.feature("size1")
                .set("custom", "on")
                .set("hmax", "(lda_c/n_c) / 10")
                .set("hmaxactive", true)
                .set("hmin", "50[nm]")
                .set("hminactive", true)
                .set("hgradactive", true)
                .set("hgrad", 1.15)
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
                .set("argunit", new String[]{"Hz"})
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

        model.material().create("layered_mat", "LayeredMaterial", "");
        model.material("layered_mat")
                .set("layername", new String[]{"film_layer", "gold_layer"})
                .set("link", new String[]{"ndnio3", "au"})
                .set("rotation", new String[]{"0.0", "0.0"})
                .set("thickness", new String[]{"12[nm]", "200[nm]"})
                .set("meshPoints", new int[]{2, 2})
                .set("tag", new String[]{"lmat1_2", "lmat1_1"})
                .set("intname", new String[]{"film_layer down", "film_layer-gold_layer", "gold_layer up"})
                .set("position", new String[]{"0E0", "12E-9", "212E-9"})
                .set("matLink_int", new String[]{"", "", ""})
                .set("labelModified", new String[]{"false", "false", "false"})
                .label("layered_stack");
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
        component.material("matlnk_gap").selection().geom("geom", 2);
        component.material("matlnk_gap").selection().set(selections.get("gap"));

        component.material().create("llmat_resonator", "LayeredMaterialLink").label("resonator");
        component.material("llmat_resonator").selection().set(selections.get("resonator"));
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
                .set("d", "12[nm]")
                .label("tbc_nickelate");
        tbc_film.selection().set(selections.get("film"));

        ParameterEntity tbc_gap = emw.create("trans2", "TransitionBoundaryCondition", 2);
        tbc_gap
                .set("d", "12[nm]")
                .label("tbc_gap");
        tbc_gap.selection().set(selections.get("gap"));

        ParameterEntity lbc_resonator = emw.create("ltbc1", "LayeredTransitionBoundaryCondition", 2);
        lbc_resonator.selection().set(selections.get("resonator"));
        lbc_resonator.set("shelllist", "none").label("layered_tbc");
        lbc_resonator.active(false);

        ParameterEntity tbc_resonator = emw.create("trans3", "TransitionBoundaryCondition", 2);
        tbc_resonator
                .set("d", "200[nm]")
                .label("tbc_resonator");
        tbc_resonator.selection().set(selections.get("resonator"));

        String field_expr = "pump_real(freq)-j*pump_imag(freq)";

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

        emw.prop("BackgroundField")
                .set("SolveFor", "scatteredField")
                .set("Eb", new String[]{"0", field_expr, "0"});
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
        variables.set("clistctrl", new String[]{"p1"});
        variables.set("cname", new String[]{"freq"});
        variables.set("clist", new String[]{"1[THz]"});

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
        parametric.set("pname", new String[]{"freq"});
        parametric.set("plistarr", new String[]{"1[THz]"});
        parametric.set("punit", new String[]{"THz"});
        parametric.set("pcontinuationmode", "no");
        parametric.set("preusesol", "auto");
        parametric.set("plot", true);

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
        multigrid.feature("cs").feature("d1").set("linsolver", "pardiso");

        stationary.feature().remove("dDef");
        stationary.feature().remove("fcDef");
        stationary.feature().remove("pDef");
    }

    public static void createExports(Model model, Map<String, int[]> selections) {



        model.result().table().create("params_table", "Table").label("parameters");
        model.result().dataset().create("sample_surface", "Surface").label("Surface");
        model.result().dataset("sample_surface").selection().set(selections.get("surface"));

        NumericalFeature gapLoss = model.result().numerical().create("gap_integral", "IntSurface");
        gapLoss.selection().set(selections.get("gap"));
        gapLoss
                .set("table", "params_table")
                .set("expr", new String[]{"emw.Qsrh"})
                .set("unit", new String[]{"W"})
                .set("descr", new String[]{"Qsrh"});

        NumericalFeature gapProbe = model.result().numerical().create("gap_point", "EvalPoint");
        gapProbe
                .set("probetag", "none")
                .set("table", "params_table")
                .set("expr", new String[]{"real(emw.Ey)", "imag(emw.Ey)", "real(emw.sigmabnd)"})
                .set("unit", new String[]{"V/m", "V/m", "S/m"})
                .set("descr", new String[]{"Ey_gap.real", "Ey_gap.imag", "cond_gap"})
                .label("gap_center");
        gapProbe.selection().named("geom_ballsel1");

        NumericalFeature filmProbe = model.result().numerical().create("film_point", "EvalPoint");
        filmProbe
                .set("probetag", "none")
                .set("table", "params_table")
                .set("expr", new String[]{"real(emw.sigmabnd)"})
                .set("unit", new String[]{"S/m"})
                .set("descr", new String[]{"cond_film"})
                .label("film_center");
        filmProbe.selection().named("geom_filmsel1");

        ExportFeature parametersExport = model.result().export().create("params_export", "Table");
        parametersExport
                .set("header", false)
                .set("ifexists", "append")
                .set("prec", "manual")
                .set("manualprec", 3)
                .label("parameters");

        ExportFeature spatialExport = model.result().export().create("spatial_data", "Data");
        spatialExport
                .set("data", "sample_surface")
                .set("expr", new String[]{
                        "real(emw.Ey)", "imag(emw.Ey)",
                        "real(emw.Jy)", "imag(emw.Jy)",
                        "real(emw.Jsupy)", "imag(emw.Jsupy)",
                        "real(emw.sigmabnd)", "emw.Qsrh", "emw.Qe"})
                .set("descr", new String[]{
                        "Ey.real", "Ey.imag",
                        "Jy.real", "Jy.imag",
                        "Jsupy.real", "Jsupy.imag",
                        "cond", "Qsrh", "Qe"
                })
                .set("unit", new String[]{
                        "V/m", "V/m",
                        "A/m^2", "A/m^2",
                        "A/m", "A/m",
                        "S/m", "W/m^2", "W/m^3"
                })
                .set("location", "grid")
                .set("gridx2", "range(0,0.04,4)")
                .set("gridy2", "range(0,0.004,0.4)")
                .set("header", true)
                .set("fullprec", false)
                .set("ifexists", "append")
                .set("separator", ",")
                .set("sort", true)
                .set("gridstruct", "grid")
                .label("surface_solution");
    }
}
