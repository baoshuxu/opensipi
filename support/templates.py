import re
import pandas as pd
from .folders import pstr
import shutil
import json

HS_CREATE_JUMER_PORTS = False

# sort key for lyn: 'A9' is smaller than 'A11'
_lynsort = lambda s: ((z.group(1), int(z.group(2))) if (z := re.match(r"([A-Za-z]+)(\d+)", s)) else (s, 0))

# shortcuts
_curly = lambda e: '{'+ '} {'.join(e) +'}' if isinstance(e, list) else '{' + e +'}'
_spacy = lambda lst: " ".join(set(lst))
_tabby = lambda lst: "\n\t".join(set(lst))
_u = lambda pin: pin.split(".")[0]
_cplit = lambda lst: [e for e in re.split(r"[\s,]+", lst) if e]
_simulate = lambda df: df[(df['simulate'].str.upper() != 'FALSE') & (df['io'].str.strip() != '')]
_brace = lambda string: "{"+string+"}"
# re.sub wrapper
class rex(str):

	def __init__(self, string):
		self.s = str(string)

	def sub(self, pat, rep, count=1, flags=0):
		self.s = re.sub(pat, rep, self.s, count=count, flags=flags)
		return self.s

	def map(self, subs):
		for pat, rep in subs.items():
			self.sub(pat, rep)
		return self.s

	def __str__(self):
		return self.s

	def __repr__(self):
		return self.s


class template:

	csv_pdcresi = {
			"Resistance Name": ["net1_pin_001"],
			"Model": ["L2L"],
			"Positive Pin": ["u1.1"],
			"Negative Pin": ["u2.2"],
			"ShortVRM": ["1"],
			"OtherCKT": ["1"],
	}

	ael_test_ads = r"""
decl fh = fopen("has_ads.txt","W");
if( fh != NULL) {
	fprintf( fh, "hola ads");
	fclose(fh);
}
de_exit();
"""

	ael_add_snp = r"""
decl interface = _INTERFACE_;
decl adsfolder =  _ADSFOLDER_;
decl ts_names  = list(_TSNAMES_);
decl ts_ports = list(_TSPORTS_);
decl ts_orient = list(_TSORIENT_);
decl ts_files = list(_TSFIILES_);
decl ts_flip = list(_TSFLIP_);
decl ts_folder = "../../.snp";

defun add_s(id,file)
{
	decl flip = ts_flip[id];
	decl z = de_init_item("ads_datacmps:SnP:symbol");
	decl x = id*6;
	decl y = -3 ;
	de_place_item(z, x, y);
	if(flip>0) {
		de_select_range(x, y, x, y, 1, 1);
		de_mirror_y(x+0.375, y);
	}
	z = de_free_item(z);
	decl z = de_edit_item("SnP1");
	//decl s = strcat("S",id);
	decl s = ts_names[id];
	decl fileparts = parse(file,".");
	decl ext = fileparts[listlen(fileparts)-1];
	decl number_ports = midstr(ext,1,strlen(ext)-2);
	decl port_orient = ts_orient[id];
	decl port_name = ts_ports[id];
	de_set_item_id(z, s);
	de_set_item_parameters(z, list(
		prm_ex("ads_datacmps","StdForm",number_ports),
		prm_ex("ads_datacmps","NoRefPin"),
		prm_ex("ads_datacmps",port_orient),
		prm_ex("ads_datacmps","Loose"),
		prm_ex("ads_datacmps","dfiles",strcat("\"",file,"\"")),
		prm_ex("ads_datacmps","TouchstoneType"),
		prm_ex("ads_datacmps","StringAndReference",""),
		prm_ex("ads_datacmps","Mode0"),
		prm_ex("ads_datacmps","ID0"),
		prm_ex("ads_datacmps","EMode2"),
		prm_ex("ads_datacmps","StdForm","27.0"),
		prm_ex("ads_datacmps","y_n0"),
		prm_ex("ads_datacmps","StdForm",""),
		prm_ex("ads_datacmps","StdForm",""),
		prm_ex("ads_datacmps","StdForm",""),
		prm_ex("ads_datacmps","StdForm",""),
		prm_ex("ads_datacmps","StdForm",""),
		prm_ex("ads_datacmps","StdForm",""),
		prm_ex("ads_datacmps","StdForm",""),
		prm_ex("ads_datacmps","StdForm",""),
		prm_ex("ads_datacmps","Auto",""),
		prm_ex("ads_datacmps","StdForm",port_name),
		prm_ex("ads_datacmps","NoAction"),
		prm_ex("ads_datacmps","StdForm","")));
	de_end_edit_item(z);
	z = de_free_item(z);
}

defun add_cell(lib, cell)
{
	decl canvas = open_or_create_schematic(lib, cell, "schematic");    
	de_show_context_in_new_window(canvas);
	decl item = de_init_item("ads_simulation:S_Param:symbol");
	de_rotate_image("RIGHT");
	de_place_item(item, 2.625, 1.375);
	item = de_free_item(item);
	decl item = de_init_item("ads_simulation:TermG:symbol");
	de_rotate_image("LEFT");
	de_place_item(item, 1.25, 3);
	de_rotate_image("RIGHT");
	de_place_item(item, 6.25, 3);
	item = de_free_item(item);
	de_save_oa_design(strcat(lib,":",cell,":schematic"));
	de_refresh_view();
}

defun open_or_create_workspace( workspace )
{
	decl status = FALSE ;
	if ( ael_file_exists( workspace ) && ael_is_file_writable( workspace )
		&& directory_is_workspace( workspace) )	{
		status = de_open_workspace( workspace );
	}
	else if ( de_create_new_workspace( workspace, FALSE ) ) 	{
		status = de_open_workspace( workspace );
	}
	else 	{
		de_info(strcat("ERROR: Failed to open workspace `", workspace, "'"));
	}
	return status;
}
	
defun open_or_create_library( lib, path )
{
	decl status = FALSE;
	status = de_is_library_open( lib );
	if( !status ) {
		de_new_library( lib, path );
		de_open_library_shared_mode( lib, path);
		de_add_library_to_workspace_shared_mode( lib, path );
		status = de_is_library_open( lib );
	}
	return status;
}

defun open_or_create_schematic( lib, cell )
{
	if( !open_or_create_library( lib ) )
		return ;
	decl cellExists = de_cellview_exists(lib, cell, "schematic");
	decl context;
	if (cellExists)	{
		context = de_find_design_context_from_name( strcat(lib, ":", cell, ":schematic") );
		de_bring_context_to_top_or_open_new_window(context);
		de_select_all();
		de_delete();
	}
	else {
		context = de_create_new_schematic_view( lib, cell, "schematic");
		de_show_context_in_new_window(context);
	}
	return context;
}

defun build_chn_schmetic()
{
	decl i;
	decl mylib = strcat(interface,"_lib");
	decl mycell = strcat("chn_", interface);
	decl myproj = strcat(adsfolder,"/",toupper(interface));
	decl mylib_path = strcat(myproj,"/",mylib);
	if (!open_or_create_workspace( myproj )) {
		return;
	}
	if( open_or_create_library(mylib, mylib_path) ) {
		add_cell(mylib, mycell);
		for( i=0; i<listlen(ts_files); i++ ) {
			add_s(i,strcat(ts_folder,"/",ts_files[i]));
		}
		de_save_oa_design(strcat(mylib,":",mycell,":schematic"));
	}
	de_close_all();
}

build_chn_schmetic();
de_exit();
"""

	il_comp_descr = r"""
p= axlDMOpenFile("ALLEGRO_TEXT" "./comp_descr" "w")
complist = axlDBGetDesign()->components
foreach(item complist
refdes = item->name
fprintf(p "%s! " refdes)
descr = item->compdef->prop->DESCR
if(descr then fprintf(p "%s\n" descr) else fprintf(p "\n") )
)
axlDMClose(p)
"""

	il_comp_props = r"""
p= axlDMOpenFile("ALLEGRO_TEXT" "./comp_props" "w")
complist = axlDBGetDesign()->components
foreach(item complist
fprintf(p "%s!" item->name)
p_comp = axlDBGetProperties(item '("user" "allegro"))
write(p_comp, p)
fprintf(p "!")
p_compdef = axlDBGetProperties(item->compdef '("user" "allegro"))
write(p_compdef, p)
fprintf(p "\n")
)
axlDMClose(p)
"""

	il_fpn_cmp = r"""
file = strcat(axlCurrentDesign() ".fpn" ) )
axlReportGenerate("Function Pin Report" nil file )
file = strcat(axlCurrentDesign() ".cmp" ) )
p= axlDMOpenFile("ALLEGRO_TEXT" file "w")
foreach(o axlDBGetDesign()->components
	fprintf(p "%s!" o->name)
	comp = axlDBGetProperties(o '("user" "allegro"))
	write(comp, p)
	fprintf(p "!")
	compdef = axlDBGetProperties(o->compdef '("user" "allegro"))
	write(compdef, p)
	fprintf(p "\n")
)
axlDMClose(p)
"""

	tcl_ac_canvas = r"""
set file_name "_FILE_NAME_"
set brd_path  "_BOARD_PATH_"
set sim_path  "_SIMULI_PATH_"
set xnet     	"_EXTENDED_NET_"

set GndArray {
	_GROUND_NETS_
}
set PwrArray {
	_POWER_NETS_
}
set NetArray {
	_SIGNAL_NETS_
}

set spd_file $brd_path
append spd_file "/" $file_name
sigrity::open document $spd_file {!}

sigrity::update option -EnforceCausality 1 {!}
sigrity::update option -PowerNetImpedance {0.1} {!}
sigrity::update option -SignalNetImpedance {50.0000} {!}
sigrity::material update -name {COPPER} -type {metal} -temperature {20} -conductivity {5.2e+07} {!}
sigrity::add SurfaceRoughness -name {HurayModel} -type {Huray} -SurfaceRatio {2.5} -SnowballRadius {0.5}
sigrity::update option -CalcDCPoint {1} -PCEnforcementByBBS {0} -PDCEqualPotential {1} {!} 
sigrity::update option -ResultFileHasTouchstone {1} -ResultFileHasTouchstone2 {0} -ResultFileHasBnp {1} {!}
sigrity::update option -MarginForCutByNet {5 mm} {!} 
sigrity::update option -MaxCPUPer {90} {!}
{OTHER_OPTIONS}
puts $xnet
set out_spd $sim_path
append out_spd "/" $xnet "/" $xnet ".spd"
sigrity::save $out_spd {!}
sigrity::update net selected 0 -all {!}
sigrity::delete port -all {!}

foreach gnd $GndArray {
	sigrity::update net selected 1 $gnd {!}
	sigrity::move net {GroundNets} $gnd {!}
}
foreach rail $PwrArray {
	sigrity::update net selected 1 $rail {!}
	sigrity::move net {PowerNets} $rail {!}    
}
foreach sig $NetArray {
	sigrity::update net selected 1 $sig {!}
	sigrity::move net {NULL} $sig {!}
}

_PORTS_TCL_

sigrity::save $out_spd {!}	
sigrity::begin simulation {!}
sigrity::save $out_spd {!}	

set out_ready $sim_path
append out_ready "/" $xnet ".done"
set outfile [open $out_ready w]
close $outfile
sigrity::exit -n {!}
"""

	tcl_brd_spd = r"""
set stackup "STACKUP"
set database "DATABASE"
set library "LIBRARY"
set material "MATERIAL"
set precut   "PRECUT"
sigrity::spdif_option UseBoardOutline {1} {!}
sigrity::spdif_option ConvertStaticShape {1} {!}
sigrity::spdif_option createModelByPartNumber {1} {!}
sigrity::open document $database {!}
sigrity::update net selected 0 -all {!}
sigrity::open ammLibrary $library {!}
sigrity::assign -all {!}
sigrity::add SurfaceRoughness -name {HurayModel} -type {Huray} -SurfaceRatio {2.5} -SnowballRadius {0.5} {!}
# True or False
if {$material != "MATERIAL"} {
	sigrity::import material $material {!}
	sigrity::update material $material -all {!}
}
if {$stackup != "STACKUP"} {
	sigrity::import stackup $stackup {!}
}
if {$precut !="PRECUT" } {
	set xy [split $precut " "]
	set xy0 [lrange $xy 0 1]
	set p0 [join $xy0 ", "]
	set xy1 [lrange $xy 2 3]
	set p1 [join $xy1 ", "]
	sigrity::delete area -LeftPoint $p0 -RightPoint $p1 -Outside {!}
	sigrity::process shape {!}
}
sigrity::update option -EnforceCausality 1 {!}
sigrity::save {!}
sigrity::exit -n
"""

	tcl_bnp_snp = r"""
sigrity::open CurveFile {SRC} {!}
sigrity::save curve -NetWork {SRC} -FileName {TGT} -CurveFileType {TouchStone} -MatrixTypeToSave {S} -MatrixDataType {RI} -SaveInAFS -FreqUnit {HZ} {!}
sigrity::exit -n {!}
"""

	tcl_psi_clock = rex(tcl_ac_canvas).sub(
			"{OTHER_OPTIONS}",
			"""
sigrity::update workflow -product {PowerSI} -workflowkey {extraction} {!}
sigrity::update freq -start 0.000000 -end 10000000000.000000 -AFS {!}

#for PTH
sigrity::update option -DoglegHoleThreshold {0.00075} -ThermalHoleThreshold {0.00075} -SmallHoleThreshold {0.00075} -ViaHoleThreshold {0.00075} {!}
sigrity::update option -MaxEdgeLength {0.0075} {!}
#for HDI
#sigrity::update option -DoglegHoleThreshold {0.0001} -ThermalHoleThreshold {0.0001} -SmallHoleThreshold {0.0001} -ViaHoleThreshold {0.0001} {!}
#sigrity::update option -MaxEdgeLength {0.002000} {!}
""",
	)

	tcl_psi_serdes = rex(tcl_ac_canvas).sub(
			"{OTHER_OPTIONS}",
			"""
sigrity::update freq -freq {5000000.000000, 25000000000.000000, 5000000.000000, linear, 20} {!}
sigrity::spdif_option ConvertStaticShape {1} {!}
sigrity::update option -DoglegHoleThreshold {0.0001} -ThermalHoleThreshold {0.0001} -SmallHoleThreshold {0.0001} -ViaHoleThreshold {0.0001} {!}
sigrity::update option -SmallHoleThreshold {0.0001} -ViaHoleThreshold {0.0001} {!}
sigrity::update option -MaxEdgeLength {0.002000} {!}
""",
	)

	tcl_psi_rails = rex(tcl_ac_canvas).sub(
			"{OTHER_OPTIONS}",
			"""
sigrity::update workflow -product {PowerSI} -workflowkey {extraction} {!}
sigrity::update freq -freq {100, 1000000000, 10, log, 30}
sigrity::update option -EnablePortGenAnalysisFlow {0} -EnableDCAccurateMode {1} {!}
#for PTH
#sigrity::update option -DoglegHoleThreshold {0.00075} -ThermalHoleThreshold {0.00075} -SmallHoleThreshold {0.00075} -ViaHoleThreshold {0.00075} {!}
#sigrity::update option -MaxEdgeLength {0.0075} {!}
# for HDI
sigrity::update option -DoglegHoleThreshold {0.0001} -ThermalHoleThreshold {0.0001} -SmallHoleThreshold {0.0001} -ViaHoleThreshold {0.0001} {!}
sigrity::update option -MaxEdgeLength {0.002000} {!}
""",
	)

	tcl_clr_clock = rex(tcl_ac_canvas).sub(
			"{OTHER_OPTIONS}",
			"""
sigrity::update workflow -product {Clarity 3D Layout} -workflowkey {3DFEMExtraction} {!}
sigrity::update option -Wave3DSettingminimumAdaptiveIterations {10} -Wave3DSettingminimumConvergedIterations {2} {!}
sigrity::update option -Wave3DSettingsolutionfreq {10e+9} -Wave3DFreqBand {{1e+07 10e+9 linear 1e+07}{0 1e7 linear 1e7}}

#for PTH
sigrity::update option -DoglegHoleThreshold {0.00075} -ThermalHoleThreshold {0.00075} -SmallHoleThreshold {0.00075} -ViaHoleThreshold {0.00075} {!}
#sigrity::update option -MaxEdgeLength {0.0075} {!}
# for HDI
#sigrity::update option -DoglegHoleThreshold {0.0001} -ThermalHoleThreshold {0.0001} -SmallHoleThreshold {0.0001} -ViaHoleThreshold {0.0001} {!}
#sigrity::update option -MaxEdgeLength {0.002000} {!}
""",
	)

	dns_comma_str = r"""
set dns_list "_NOT_STUFF_"
foreach dns [split $dns_list ","] {
	set substr $dns
	set dns_info [sigrity::querydetails ckt -name $dns]
	append substr "*"
	if {[string match $substr $dns_info]} {
		sigrity::update circuit -manual {disable} $dns {!}
	}
}
"""

	tcl_pdc_resi = r"""
set file_name "_FILE_NAME_"
set file_csv  "_FILE_CSV_"
set brd_path  "_BOARD_PATH_"
set sim_path  "_SIMULI_PATH_"

set GndArray {
	_GROUND_NETS_
}
set PwrArray {
	_POWER_NETS_
}
set R0Array {
	_XW_COMPONENT_
}
set r0 _XW_MODEL_

set spd_file $brd_path
append spd_file "/" $file_name
sigrity::open document -attach $spd_file {!}

sigrity::set pdcSimMode -ResistanceMeasurement {1} {!}
sigrity::set pdcAccuracyMode {1} {!}
sigrity::update net selected 0 -all {!}
sigrity::update option -AutoSaveExcelResult {1} {!}
sigrity::set pdcAutoSaveExcelResult -fileFormat {.csv} {!}
sigrity::update option -MaxCPUPer {90} {!}

sigrity::update circuit -manual {disable} _NOT_STUFF_ {!}

foreach component $R0Array {
	sigrity::update circuit -model $r0 $component {!}
}
foreach gnd $GndArray {
	sigrity::update net selected 1 $gnd {!}
	sigrity::move net {GroundNets} $gnd {!}
}
foreach rail $PwrArray {
	sigrity::update net selected 1 $rail {!}
	sigrity::move net {PowerNets} $rail {!}
}

set csv_file $sim_path
append csv_file "/" $file_name "_resi/" $file_csv
sigrity::add pdcResist -csvfile $csv_file {!}

set out_spd $sim_path
append out_spd "/" $file_name "_resi/" $file_name "_resi.spd"
sigrity::save $out_spd {!}	

sigrity::begin simulation {!}
sigrity::save $out_spd {!}	

set out_ready $sim_path
append out_ready "/" $file_csv ".done"
set outfile [open $out_ready w]
close $outfile
sigrity::exit -n {!}
"""
	
	tcl_vrm_sinks = r'''
sigrity::add pdcVRM -manual -name {<VRM>} -resistance {0} -outputCurrent {<CURRENT>} -sensevoltage {<SENSE_VOLT>} -tolerance {0} -voltage {<VOLTAGE>} {!}
sigrity::link pdcElem {<VRM>} {Positive Pin}  {-Circuit {<REFDES_P>} -Node {<PIN_P>}} -LinkCktNode {!}
sigrity::link pdcElem {<VRM>} {Negative Pin}  {-Circuit {<REFDES_N>} -Node {<PIN_N>}} -LinkCktNode {!}
sigrity::link pdcElem {<VRM>} {Positive Sense Pin}  {-Circuit {<REFDES_P>} -Node {<PIN_P>}} -LinkCktNode {!}
sigrity::link pdcElem {<VRM>} {Negative Sense Pin}  {-Circuit {<REFDES_N>} -Node {<PIN_N>}} -LinkCktNode {!}
sigrity::add pdcSink -manual -name {<SINK>} -autoAddNodesOnPads {0} -current {<CURRENT>} -pfMode {Worst} -lowerTolerance {2,%} -upperTolerance {2,%} -voltage {<VOLTAGE>} -model {Equal Current} {!}
sigrity::link pdcElem {<SINK>} {Positive Pin}  {-Circuit {<REFDES_P>} -Node {<PIN_P>}} -LinkCktNode {!}
sigrity::link pdcElem {<SINK>} {Negative Pin}  {-Circuit {<REFDES_N>} -Node {<PIN_N>}} -LinkCktNode {!}
'''
	
	tcl_pdc_irdrop = r"""
set open_spd  "<OPEN_SPD>"
set save_spd  "<SAVE_SPD>"

set GndArray {
	<GROUND_NETS>
}
set PwrArray {
	<POWER_NETS>
}

sigrity::open document -attach $open_spd {!}
sigrity::update workflow -product {PowerDC} -workflowkey {IRDropAnalysis} {!}
sigrity::set pdcSimMode -IRDropAnalysis {1} {!}
sigrity::set pdcAccuracyMode {1} {!}
sigrity::update net selected 0 -all {!}
sigrity::update option -AutoSaveExcelResult {1} {!}
sigrity::set pdcAutoSaveExcelResult -fileFormat {.csv} {!}
sigrity::update option -MaxCPUPer {90} {!}
sigrity::update net selected 0 -all {!}
sigrity::update circuit -manual {disable} -all {!}
sigrity::update circuit -manual {enable} <ENABLE> {!}

foreach gnd $GndArray {
	sigrity::update net selected 1 $gnd {!}
	sigrity::move net {GroundNets} $gnd {!}
}
foreach rail $PwrArray {
	sigrity::update net selected 1 $rail {!}
	sigrity::move net {PowerNets} $rail {!}
}

if {true} {
	<SINK_LINES>
}

sigrity::save $save_spd {!}	

if {false} {
	sigrity::begin simulation {!}
	sigrity::save $save_spd {!}	
}

sigrity::exit -n {!}
"""
	
	tcl_psi_irdrop = r"""
set open_spd  "<OPEN_SPD>"
set save_spd  "<SAVE_SPD>"

set GndArray {
	<GROUND_NETS>
}
set PwrArray {
	<POWER_NETS>
}

sigrity::open document -attach $open_spd {!}
sigrity::update workflow -product {PowerSI} -workflowkey {extraction} {!}

sigrity::update option -EnforceCausality 1 {!}
sigrity::update option -PowerNetImpedance {0.1} {!}
sigrity::update option -CalcDCPoint {1} -PCEnforcementByBBS {0} -PDCEqualPotential {1} {!} 
sigrity::update option -EnablePortGenAnalysisFlow {0} -EnableDCAccurateMode {1} {!}
sigrity::update option -ResultFileHasTouchstone {1} -ResultFileHasTouchstone2 {0} -ResultFileHasBnp {1} {!}
sigrity::update option -MarginForCutByNet {5 mm} {!} 
sigrity::update option -MaxCPUPer {90} {!}
sigrity::update freq -freq {100, 1000000000, 10, log, 30}

sigrity::update net selected 0 -all {!}
sigrity::update circuit -manual {disable} -all {!}
sigrity::update circuit -manual {enable} <ENABLE> {!}

foreach gnd $GndArray {
	sigrity::update net selected 1 $gnd {!}
	sigrity::move net {GroundNets} $gnd {!}
}
foreach rail $PwrArray {
	sigrity::update net selected 1 $rail {!}
	sigrity::move net {PowerNets} $rail {!}
}

if {true} {
	<SINK_LINES>
}

sigrity::save $save_spd {!}	

if {false} {
	sigrity::begin simulation {!}
	sigrity::save $save_spd {!}	
}

sigrity::exit -n {!}
"""
	
	material_cmx = r"""
<?xml version="1.0" encoding="UTF-8"?>
<Cadence_Material_Lib Version="1.02">
<DataDescriptions>
	<Material>
<Column name="Default Thickness" unit="um" />	</Material>
	<Metal>
		<Column name="Temperature" unit="C" />
		<Column name="Conductivity" unit="S/m" />
	</Metal>
	<Dielectric>
		<Column name="Temperature" unit="C" />
		<Column name="Frequency" unit="MHz" />
		<Column name="Relative Permittivity" />
		<Column name="LossTangent" />
	</Dielectric>
	<Thermal>
		<Column name="Temperature" unit="C" />
		<Column name="Conductivity" unit="W/(m*K)" />
		<Column name="Density" unit="kg/m^3" />
		<Column name="Specific heat" unit="J/(kg*K)" />
	</Thermal>
	<Magnetic>
		<Column name="Frequency" unit="MHz" />
		<Column name="ur(real)" />
		<Column name="ur(-imag)" />
	</Magnetic>
	<SurfaceRoughness>
	<Huray>
		<Column name="Surface Ratio" />
		<Column name="Snowball Radius" unit="um" />
	</Huray>
	<ModHammerstad>
		<Column name="Roughness Factor" />
		<Column name="RMS value" unit="um" />
	</ModHammerstad>
	</SurfaceRoughness>
	<structural>
	<Elasticity>
		<Column name="Temperature" unit="C" />
		<Column name="Youngs Modulus" unit="Pa" />
		<Column name="Poissons Ratio" />
	</Elasticity>
	<CTE>
		<Column name="Reference Temperature" unit="C" />
		<Column name="Temperature" unit="C" />
		<Column name="CTE" unit="1/C" />
	</CTE>
	</structural>
</DataDescriptions>
ADD_MATERIAL
</Cadence_Material_Lib>
"""

	extracta_view = r"""
# netlist
LOGICAL_PIN
	NET_NAME != ''
	NET_NAME_SORT
	NET_NAME
	REFDES_SORT
	REFDES
	PIN_NUMBER_SORT
	PIN_NUMBER
	FUNC_DES_SORT
	FUNC_DES
	PIN_NAME
END
# comppin
COMPONENT_PIN
	REFDES  != ''
	REFDES_SORT
	REFDES
	PIN_NUMBER_SORT
	PIN_NUMBER
	COMP_DEVICE_TYPE
	PIN_TYPE
	PIN_NAME
	NET_NAME
END
# bom
COMPONENT
	REFDES_SORT
	REFDES
	PART_NUMBER
	COMP_DEVICE_TYPE
	COMP_VALUE
	COMP_TOL
	COMP_PACKAGE
	SYM_X
	SYM_Y
	SYM_ROTATE
	SYM_MIRROR
	BOM_IGNORE
	ALTERNATE_SYMBOLS                    
END
#	 fcndes, pinnam, fcntyp, slotnam, pinuse, refdes, pinnmr, netnam
LOGICAL_PIN
	FUNC_DES != ''
	FUNC_DES_SORT
	FUNC_DES
	PIN_NAME
	FUNC_TYPE
	FUNC_SLOT_NAME
	PIN_TYPE
	REFDES
	FUNC_REF_DES_FOR_ASSIGN
	PIN_NUMBER
	NET_NAME
END
"""


class tcl:

	def __init__(self, brd=None, spd=None, mux=None):
		self.brd = brd
		self.spd = spd
		self.dns = {}
		self.mux = mux
		self.r0 = "000-30577-00"

	def set(self, k, v):
		exec(f"self.{k} = {v}")

	def powerdc(self, cell=None):
		pass

	def apply_sinks(self, df=None, saveas=''):
		fspd = pstr(self.spd)
		if not df or not fspd.isfile:
			return
		
		def find_block(buffer, start, end):
			m = n = None
			for i, line in enumerate(buffer):
				if m is None and line.strip().startswith(start):
						m = i
				elif m is not None and line.strip().startswith(end):
						n = i
						break  # Stop after finding the first complete .VRM block
			return m, n			
			
		buffer = []
		ftmp = fspd.ext('.tmp')
		marker='* PdcElem description lines'
		with open(str(fspd), 'r') as src_file, open(str(ftmp), 'w') as dst_file:
			for line in src_file:
				dst_file.write(line)
				if line.strip() == marker:
					break
			buffer = src_file.readlines()
		m,n = find_block(buffer,'.Sink ', '.EndSink')
		
		m,n = find_block(buffer,'.VRM ', '.EndVRM')

		
		fovr = str(saveas) if(saveas and pstr(pstr(saveas).path()).isdir) else str(fspd)
		shutil.move(str(ftmp), fovr)					
		
	def many_grounds(self, str_pins='', str_gnds=''):
		ports, shorts = {}, {}
		gnd_nets = set(str_gnds.split(','))
		if len(gnd_nets)<2:
			return ports, shorts
		
		for u,(nets,pins) in self.spd.shorts.items():
			if len(set(nets)) == 2 and all( net in gnd_nets for net in nets):
				shorts[u] = nets
		
		u_grounds = {}
		for pin in str_pins.split(','):
			u = _u(pin)
			if u in u_grounds:
				continue
			u_gnd_pin_nodes = self.spd.gnd_pin_nodes(u)
			nets = [x.rpartition('::')[-1] if '::' in x else None for x in u_gnd_pin_nodes]
			nets = set(filter( lambda net: net in gnd_nets, nets))
			if nets:
				hit = re.compile(rf"{'|'.join(nets)}$",re.I)
				nodes = [s for s in u_gnd_pin_nodes if hit.search(s)]
				u_grounds[u] = (nets,nodes)
		if u_grounds:
			u_max =  max(u_grounds, key=lambda k: len(u_grounds[k][0]))
			(u_max_nets, u_max_nodes) = u_grounds[u_max]
			ports = {u_max: u_max_nodes}
			if len(u_max_nets) == len(gnd_nets):
				return ports, {}
			if len(u_max_nets) > 1:
				return ports, shorts
		
		return ports, shorts	

	def powerac(self, cell=None):
		if not cell:
			return
		spd_file = pstr(self.spd.spd_file)
		dns_str = ",".join([x for x in self.brd.dns if self.brd.dns[x]])
		dns_line = rex(template.dns_comma_str.lstrip()).map({"_NOT_STUFF_": dns_str}) if dns_str else ""
		rails = dict(
				(cell["xnet"][i], i)
				for (i, x) in enumerate(cell["grp"])
				if x.startswith("rail") and cell["simulate"][i] == "TRUE"
		)
		for rail, i in rails.items():
			nets = re.sub(",", " ", cell["nets"][i])
			soc, pmic, sns = cell["soc"][i], cell["pmic"][i], cell["sense"][i]
			pmic = pmic.split(",")[0]
			tcl = dns_line

			# add ports on soc
			if soc in self.spd.data["complyr"]:
				l1 = self.spd.data["complyr"][soc]
				tcl += (
						"sigrity::add port -circuit {"
						+ soc
						+ "} -layer {"
						+ l1
						+ "} -select {pin} -RefNet {GroundNets} -row {40} -col {40} -RefNetNodeTake -OnPin {!}\n"
				)

			# add ports on pmic
			if pmic in self.spd.data["complyr"]:
				# l2 = self.spd.data['complyr'][soc]
				# tcl += 'sigrity::add port -circuit {'+pmic+'} -layer {'+l2+'} -select {pin} -RefNet {GroundNets} -row {1} -col {1} -RefNetNodeTake -OnPin {!}'
				tcl += "sigrity::add port -circuit {" + pmic + "} {!}\n"

			# add ports on sense
			if bool(sns):
				pins = list(eval(sns).values()) if sns.startswith("{") else sns.split(",")
				if pins and self.spd.sig_pin_node(pins[0]):
					prt, num = pins[0].split(".")
					tcl += "sigrity::add port -name sense_port {!}\n"
					tcl += "sigrity::hook -port sense_port -circuit " + prt + " -PositiveNode " + num + " {!}\n"
					if len(pins) > 1:
						prt, num = pins[1].split(".")
						if prt[0] in "cC":  # sense port- not on pin, for DC flow
							pp, nn, d = self.spd.reference_node(pins[0], nearest=2, cut_margin=5)
							sn = self.spd.sig_pin_node(pins[1])
							ref = [x for x in nn if not x in sn]
							tcl += "sigrity::hook -port sense_port -nn " + ref[0] + " {!}\n" if ref else ""
						else:
							tcl += "sigrity::hook -port sense_port -circuit " + prt + " -NegativeNode " + num + " {!}\n"
					else:
						pp, nn, d = self.spd.reference_node(pins[0], nearest=True, cut_margin=5)
						tcl += "" if d < 0 else "sigrity::hook -port sense_port -nn " + nn + " {!}\n"
				else:
					prompt = f"cannot found node for {pins[0]}" if pins else "sense pin is blank"
					print(f"tcl.powerac(): {prompt}")

			# short the inductors
			inductors = [z.split(".")[0] for z in re.findall(r"(L\d+\.\d)", cell["shunt"][i])]
			for ind in inductors:
				tcl += "sigrity::update circuit -model {" + self.r0 + "} {" + ind + "} {!}\n"
			pos, neg = (
					re.split("/+", cell["nets"][i])[:2] if "/" in cell["nets"][i] else (cell["nets"][i], cell["ground"][i])
			)
			pwr, gnd = _cplit(pos), _cplit(neg)
			tcl += "sigrity::delete area -Net " + _curly(pwr) + " {!}\n"
			# write .tcl file

			file = pstr(spd_file.sub(f"/Scripts/{rail}.tcl"))
			txt = rex(template.tcl_psi_rails.lstrip()).map(
					{
							"_FILE_NAME_": spd_file.base(),
							"_BOARD_PATH_": spd_file.path(),
							"_SIMULI_PATH_": spd_file.sub("/SimFiles"),
							"_EXTENDED_NET_": rail,
							"_GROUND_NETS_": _tabby(gnd),
							"_POWER_NETS_": _tabby(pwr),
							"_SIGNAL_NETS_": rail,
							"_PORTS_TCL_": tcl,
					}
			)
			file.write(txt)
			if file.recently():
				print(f"  tcl created: {file}")

	def clocks(self, cell=None):
		app = self.spd._exe if self.spd else None
		if (not app) or (not isinstance(cell, pd.DataFrame)) or (cell.empty):
			return
		if re.search("Clarity3DLayout.exe",app, re.I):
			return self.clr_clocks(_simulate(cell))
		elif re.search("PowerSI.exe", app, re.I):
			return self.psi_clocks(_simulate(cell))
		else:
			print(f"? templates.tcl.clocks(): tcl for {app} not defined")

	def serdes(self, cell=None):
		if self.spd is None:
			return {}
		if cell is None:
			return {}
		app = self.spd._exe
		if re.search("Clarity3DLayout.exe", app, re.I):
			return self.clr_serdes(cell)
		elif re.search("PowerSI.exe", app, re.I):
			return self.psi_serdes(cell)
		else:
			print(f"? templates.tcl.clocks(): tcl for {app} not defined")
			return {}

	def psi_chips(self, cell=None):

		if not cell:
			return None

		sel = [x == "TRUE" for x in cell["simulate"]]
		cell = {k: [x for x, s in zip(v, sel) if s] for k, v in cell.items()}

		def _pin_n(u, p):
			n_pins = []
			p_pins = sorted(re.split(r"[\s,]+", p))
			for pin in p_pins:
				pp, nn, d = self.spd.reference_node(f"{u}.{pin}", nearest=True, onpin=True, cut_margin=5)
				n_pins += ["" if d < 0 else re.split(r"[!:]+", nn)[1]]
			return " ".join(sorted(set(n_pins)))

		def _pin_port(name, p, n, refz=""):
			u1, nodes_p = p[0], " ".join(_cplit(p[1]))
			u2, nodes_n = n[0], " ".join(_cplit(n[1]))
			s = f"sigrity::add port -name {name} {{!}}\n"
			s += f"sigrity::hook -port {name} -circuit {u1} -PositiveNode {nodes_p} {{!}}\n"
			s += f"sigrity::hook -port {name} -circuit {u2} -NegativeNode {nodes_n} {{!}}\n"
			if refz:
				s += f"sigrity::update port -name  {name} -RefZ {{{refz}}} {{!}}\n"
			return s

		def _nod_port(name, u, nd, refz=""):
			s = f"sigrity::add port -name {name} {{!}}\n"
			s += f"sigrity::hook -port {name} -circuit {u} -PositiveNode {nd[0]} {{!}}\n"
			s += f"sigrity::hook -port {name} -nn {nd[1]} {{!}}\n"
			if refz:
				s += f"sigrity::update port -name  {name} -RefZ {{{refz}}} {{!}}\n"
			return s

		def _pmic_port(pmic_string):
			if "/" in pmic_string:
				x = re.split(r"\s*/\s*", pmic_string)
				return _pin_port("pmic", x[0].split(","), x[1].split(","))
			w = pmic_string.split(",")
			if len(w) < 2:
				return "sigrity::add port -circuit {" + pmic_string + "} {!}\n"
			u, p = w[0], w[1]
			return _pin_port("pmic", [u, p], [u, _pin_n(u, p)])

		spd_file = pstr(self.spd.spd_file)

		port_pair = {
				"port_p": ["" for _ in cell["port_p"]],
				"port_n": ["" for _ in cell["port_p"]],
		}
		strdns = " ".join(x for x in self.dns if self.dns[x]) if self.dns else ""
		dns_line = "sigrity::update circuit -manual {disable} " + strdns + " {!}\n" if strdns else ""
		USE_RAIL_NAME_FOR_FB = True
		buck_rail_name = {}

		rails = {cell["xnet"][i]: i for (i, x) in enumerate(cell["grp"]) if x.startswith("rail")}
		for rail, i in rails.items():
			soc = cell["soc"][i]
			if soc in self.spd.data["complyr"] is False:
				print(f"error adding ports for rail {rail} to soc {soc} (not placed?) ")
				break

			if USE_RAIL_NAME_FOR_FB:
				buck_rail_name.update({cell["buck"][i]: rail})
			# port_grps = cell['port_p'][i]
			# if speed in [1,'1'] and not re.split(r"[~\r\n]+", port_grps):
			# 	continue	#skip extraction for grouped ports while no groups given
			pmic_string, sns = cell["pmic"][i], cell["sense"][i]
			nets = re.sub(",", " ", cell["nets"][i])
			pmic = pmic_string.split(",")[0]
			tcl = dns_line
			# add ports on soc
			# l1 = self.spd.data['complyr'][soc]
			soc_ps = [x.split(".")[1] for x in cell["io"][i].split(",") if x.startswith(soc)]
			port_ps = re.split(r"[~\r\n]+", cell["port_p"][i]) if cell["port_p"][i] else []
			mem_rail = cell["speed"][i] != "0"
			p_pins = port_ps if (port_ps and mem_rail) else sorted(soc_ps, key=_lynsort)
			for j, pin_p in enumerate(p_pins):
				name = f"port_{j:03d}" if port_ps else f"port_{j:03d}_{soc}_{pin_p}"
				if "/" in p_pins[j]:
					pin_p, pin_n = re.split(r"[/\s]+", p_pins[j])
				else:
					pin_n = _pin_n(soc, pin_p)
				if any(x.strip() == "" for x in (pin_p, pin_n)):
					print(f"tcl.psi_chips(): node not found for pin_p={pin_p}, pin_n={pin_n}")
				else:
					port_pair["port_p"][i] += pin_p.replace(" ", ",") + "\n"
					port_pair["port_n"][i] += pin_n.replace(" ", ",") + "\n"
					tcl += _pin_port(name, [soc, pin_p], [soc, pin_n])

			speed = cell["speed"][i]
			# add ports on pmic ( if speed in 0, before sense)
			if speed in [0, "0"] and pmic in self.spd.data["complyr"]:
				tcl += _pmic_port(pmic_string)
			# add ports on sense
			if bool(sns):
				sns_pins_dict = eval(sns)
				sns_pins = list(sns_pins_dict.keys())
				prt, num = sns_pins[0].split(".")
				tcl += "sigrity::add port -name sense_port {!}\n"
				tcl += "sigrity::hook -port sense_port -circuit " + prt + " -PositiveNode " + num + " {!}\n"
				if len(sns_pins) < 2:  # se sense
					pn, nn, d = self.spd.reference_node(sns_pins[0], nearest=True, cut_margin=5)
					tcl += "" if d < 0 else "sigrity::hook -port sense_port -nn " + nn + " {!}\n"
				else:  # diff sense
					prt, num = sns_pins[1].split(".")
					if prt[0] in "cC":  # sense port- not on pin, for DC flow
						pn, nn, d = self.spd.reference_node(sns_pins[0], nearest=2, cut_margin=5)
						sn = self.spd.sig_pin_node(sns_pins[1])
						ref = [x for x in nn if not x in sn]
						tcl += "sigrity::hook -port sense_port -nn " + ref[0] + " {!}\n" if ref else ""
					else:
						tcl += "sigrity::hook -port sense_port -circuit " + prt + " -NegativeNode " + num + " {!}\n"
			# add ports on pmic ( if speed in 1, after sense)
			if speed in [1, "1"] and pmic in self.spd.data["complyr"]:
				tcl += _pmic_port(pmic_string)

			if "/" in nets:
				tcl += "sigrity::delete area -Net " + _curly(nets.split("/")) + " {!}\n"
			else:
				tcl += "sigrity::delete area -Net {PowerNets} " + _curly(nets.split()) + " {!}\n"

			# short the inductors
			# inductors = [z.split('.')[0] for z in re.findall('(L\d+\.\d)',cell['shunt'][i])]
			inductors = re.findall(r"(L\d+)", cell["series"][i])
			for ind in inductors:
				# change comp model to self.r0 but may fail
				tcl += "sigrity::update circuit -model {" + self.r0 + "} {" + ind + "} {!}\n"

			# write .tcl file
			gnd = cell["ground"][i].replace(",", "\n\t")
			pwr = cell["nets"][i].replace(",", "\n\t")
			if "/" in cell["nets"][i]:
				pwrs, refs = re.split("/+", cell["nets"][i])[:2]
				gnd = refs.replace(",", "\n\t")
				pwr = pwrs.replace(",", "\n\t")
			file = pstr(spd_file.sub(f"/Scripts/{rail}.tcl"))
			txt = rex(template.tcl_psi_rails.lstrip()).map(
					{
							"_FILE_NAME_": spd_file.base(),
							"_BOARD_PATH_": spd_file.path(),
							"_SIMULI_PATH_": spd_file.sub("/SimFiles"),
							"_GROUND_NETS_": gnd,
							"_POWER_NETS_": pwr,
							"_SIGNAL_NETS_": "",
							"_EXTENDED_NET_": rail,
							"_PORTS_TCL_": tcl,
					}
			)
			file.write(txt)
			if file.recently():
				print(f"  tcl created: {file}")

		senses = {cell["xnet"][i]: i for (i, x) in enumerate(cell["grp"]) if x.startswith("sensep")}
		for sense, i in senses.items():
			if not cell["sense"][i]:
				continue
			rail_name = buck_rail_name[cell["buck"][i]] if cell["buck"][i] in buck_rail_name else ""
			nets = cell["nets"][i].split(",")
			xnet = min(nets, key=lambda x: len(x))
			pins = eval(cell["sense"][i])
			tcl = dns_line
			vrm_pins, xw_pins = pins[::2], pins[1::2]
			if len(vrm_pins) != len(xw_pins):
				print("wrong pin nubmer")
				continue
			v = ".".join(vrm_pins).split(".")
			x = ".".join(xw_pins).split(".")
			if len(xw_pins) == 2:
				tcl += _pin_port("sense_port", x[0:2], x[2:4], refz=100)
				tcl += _pin_port("pmic_port", v[0:2], v[2:4], refz=100)
			elif len(xw_pins) == 1:
				pp, xn, d = self.spd.reference_node(f"{xw_pins[0]}", nearest=True, onpin=True, cut_margin=5)
				tcl += _nod_port("sense_port", x[0], [x[1], xn], refz=50)
				pp, vn, d = self.spd.reference_node(f"{vrm_pins[0]}", nearest=True, onpin=True, cut_margin=5)
				tcl += _nod_port("pmic_port", v[0], [v[1], vn], refz=50)
			if "sigrity::add port" in tcl:
				exnet = rail_name + "_fb" if rail_name else re.sub("_p$", "", xnet, flags=re.I)
				tcl += "sigrity::delete area -Net " + _curly(nets) + " {!}\n"
				gnd = cell["ground"][i].replace(",", "\n\t")
				sig = cell["nets"][i].replace(",", "\n\t")
				file = pstr(spd_file.sub(f"/Scripts/{exnet}.tcl"))
				txt = rex(template.tcl_psi_rails.lstrip()).map(
						{
								"_FILE_NAME_": spd_file.base(),
								"_BOARD_PATH_": spd_file.path(),
								"_SIMULI_PATH_": spd_file.sub("/SimFiles"),
								"_EXTENDED_NET_": exnet,
								"_GROUND_NETS_": gnd,
								"_POWER_NETS_": "",
								"_SIGNAL_NETS_": sig,
								"_PORTS_TCL_": tcl,
						}
				)
				file.write(txt)
				if file.recently():
					print(f"  tcl created: {file}")

		return port_pair

	def psi_vsys(self, row={}, ports=None, discretes=None):

		if not row or ports is None:
			return None
		
		spd_file = pstr(self.spd.spd_file)
		# dns_str = ",".join([x for x in self.brd.dns if self.brd.dns[x]])
		# dns_line = rex(template.dns_comma_str.lstrip()).map({"_NOT_STUFF_": dns_str}) if dns_str else ""
		series = z.split(',') if(z:=row.get('series','')) else []
		shunts = eval(z) if(z:=row.get('shunt','')) else {} 
		u_shunts = [pin.partition('.')[0] for pin in shunts]
		tcl = "if {true} {\n\tsigrity::update circuit -manual {disable} -all {!}\n"
		if len(series):
			tcl += "\tsigrity::update circuit -manual {enable} " + _spacy(series) + " {!}\n"
		if len(shunts):
			tcl += "\tsigrity::update circuit -manual {enable} " + _spacy(u_shunts) + " {!}\n"

		
		def _pin_port(name, p, n, refz=""):
			u1, pins1 = p.split('.')
			u2, pins2 = n.split('.')
			s = f"\tsigrity::add port -name {name} {{!}}\n"
			s += f"\tsigrity::hook -port {name} -circuit {u1} -PositiveNode {pins1} {{!}}\n"
			s += f"\tsigrity::hook -port {name} -circuit {u2} -NegativeNode {pins2} {{!}}\n"
			if refz:
				s += f"\tsigrity::update port -name  {name} -RefZ {{{refz}}} {{!}}\n"
			return s

		# port tcl
		for i, ln in enumerate(ports.itertuples()):
			name = f"port_{ln.type}_{i:03d}" 
			pins = [s.replace(',', ' ') for s in ln.port]
			tcl += _pin_port(name, pins[0], pins[1], 0.1)
		tcl += "\tsigrity::delete area -Net {PowerNets} " + _curly(row.get('nets','').split(',')) + " {!}\n}\n"

		# shorts and discretes
		for ln in discretes.itertuples():
			if ln.use == '0':
				tcl += "sigrity::update circuit -model {" + self.r0 + "} {" + ln.refdes + "} {!}\n"
				continue
			if ln.use == ln.dcr and ln.dcr:
				#TODO: make sure gpn has right partical circuit with right DCR
				continue
			if ln.use == ln.acz and ln.acz:
				#TODO make sure acz models can be handled by powersi
				continue
		# write .tcl file
		gnd = row.get('ground','').replace(",", "\n\t")
		pwr = row.get('nets','').replace(",", "\n\t")
		rail =re.sub(r'/\d*$','', row.get('xnet','vsys'))
		ftcl = pstr(spd_file.sub(f"/Scripts/{rail}.tcl"))
		txt = rex(template.tcl_psi_rails.lstrip()).map(
				{
						"_FILE_NAME_": spd_file.base(),
						"_BOARD_PATH_": spd_file.path(),
						"_SIMULI_PATH_": spd_file.sub("/SimFiles"),
						"_GROUND_NETS_": gnd,
						"_POWER_NETS_": pwr,
						"_SIGNAL_NETS_": "",
						"_EXTENDED_NET_": rail,
						"_PORTS_TCL_": tcl,
				}
		)
		ftcl.write(txt)
		return f'{ftcl}'
	
	def pdc_irdrop(self, argd={}):
		fspd = pstr(self.spd.spd_file)
		df = argd.get('df', pd.DataFrame())
		if not fspd.isfile or df.empty:
			return
		outdir = argd.get('dir','')
		if outdir and not pstr(outdir).isdir:
			return
		fout =  pstr(outdir)+fspd.base() if outdir else fspd
		row_vsys = argd.get('row_vsys')
		power = re.sub(',', '\n\t', row_vsys['nets'])
		ground = re.sub(',', '\n\t', row_vsys['ground'])
		shunts = json.loads(row_vsys.get('shunt').replace("'", '"'))
		series = row_vsys.get('series')
		components = []
		for e in shunts:
			components.append(e.partition('.')[0])
		if series:
			components += series.split(',')

		d_vrm = df[df['port'].str.contains('\n')].to_dict(orient='records')
		d_sink = df[~df['port'].str.contains('\n')].to_dict(orient='records')

		txt_vrm_sinks = template.tcl_vrm_sinks.lstrip().splitlines()
		sources=[]
		for d in d_vrm:
			lines = d['port'].splitlines()
			refdes, pins = lines[0].split('.')[-2:]
			vrm = f'VRM_{refdes}_{pins.split(',')[0]}'
			current = d.get('current(mA)','1')
			current = f'{float(str(current)) * 1e-3:.6f}'
			sense_volt = d.get('voltage','3.8')
			voltage = d.get('voltage','3.8')
			#vrm name lines
			sources.append( rex(txt_vrm_sinks[0]).map(
				{"<VRM>":vrm, "<CURRENT>":current, "<SENSE_VOLT>":sense_volt, "<VOLTAGE>":voltage}
			))
			#vrm pos pin lines
			for pin in pins.split(','):
				sources.append( rex(txt_vrm_sinks[1]).map(
					{"<VRM>":vrm, "<REFDES_P>":refdes, "<PIN_P>":pin }
				))
			#vrm neg pin lines
			for refdes, pins in d.get('gnd',{}).items():
				for pin in pins.split(','):
					sources.append( rex(txt_vrm_sinks[2]).map(
						{"<VRM>":vrm, "<REFDES_N>":refdes, "<PIN_N>":pin }
					))
			
			if len(lines) <3:
				continue
			
			#vrm pos sense pin lines
			refdes, pins = lines[1].split('.')[-2:]
			for pin in pins.split(','):
				sources.append( rex(txt_vrm_sinks[3]).map(
					{"<VRM>":vrm, "<REFDES_P>":refdes, "<PIN_P>":pin }
				))
			#vrm neg pin lines
			refdes, pins = lines[2].split('.')[-2:]
			for pin in pins.split(','):
				sources.append(  rex(txt_vrm_sinks[4]).map(
					{"<VRM>":vrm, "<REFDES_N>":refdes, "<PIN_N>":pin }
				))

		for d in d_sink:
			refdes, pins = d['port'].split('.')[-2:]
			sink = f'SINK_{refdes}_{pins.split(',')[0]}'
			current = d.get('current(mA)','1')
			current = f'{float(str(current)) * 1e-3:.6f}'
			voltage = d.get('voltage','3.8')
			#sink name lines
			sources.append( rex(txt_vrm_sinks[5]).map(
				{"<SINK>":sink, "<CURRENT>":current, "<VOLTAGE>":voltage}
			))
			#sink pos pin lines
			for pin in pins.split(','):
				sources.append( rex(txt_vrm_sinks[6]).map(
					{"<SINK>":sink, "<REFDES_P>":refdes, "<PIN_P>":pin }
				))
			#sink neg pin lines
			for refdes, pins in d.get('gnd',{}).items():
				for pin in pins.split(','):
					sources.append( rex(txt_vrm_sinks[7]).map(
						{"<SINK>":sink, "<REFDES_N>":refdes, "<PIN_N>":pin }
					))
								
		txt = rex(template.tcl_pdc_irdrop.lstrip()).map(
				{
						"<OPEN_SPD>"		: str(fspd),
						"<SAVE_SPD>"		: str(fout),
						"<GROUND_NETS>"	: ground,
						"<POWER_NETS>"	: power,
						"<ENABLE>"			: _spacy(components),
						"<SINK_LINES>"	:	'\n\t'.join(sources),
				}
		)		
		
		file = fspd.ext('.irdrop.tcl')
		file.write(txt)
		if file.recently():
			print(f"  tcl created: {file}")
			return str(file)
		return None
	
	def psi_irdrop(self, argd={}):
		fspd = pstr(self.spd.spd_file)
		df = argd.get('df', pd.DataFrame())
		if not fspd.isfile or df.empty:
			return
		outdir = argd.get('dir','')
		if outdir and not pstr(outdir).isdir:
			return
		
		def str_port(name, pos, neg, refz=""):
			s = [f"sigrity::add port -name {name} {{!}}"]
			for u, pins in pos.items():
				for pin in pins.split(','):
					s += [f"sigrity::hook -port {name} -circuit {u} -PositiveNode {pin} {{!}}"]
			for u, pins in neg.items():
				for pin in pins.split(','):
					s += [f"sigrity::hook -port {name} -circuit {u} -NegativeNode {pin} {{!}}"]
			if refz:
				s += [f"sigrity::update port -name  {name} -RefZ {{{refz}}} {{!}}"]
			return s

		fout =  pstr(outdir)+fspd.base() if outdir else fspd
		row_vsys = argd.get('row_vsys')
		power = re.sub(',', '\n\t', row_vsys['nets'])
		ground = re.sub(',', '\n\t', row_vsys['ground'])
		shunts = json.loads(row_vsys.get('shunt').replace("'", '"'))
		series = row_vsys.get('series')
		components = []
		for e in shunts:
			components.append(e.partition('.')[0])
		if series:
			components += series.split(',')

		d_vrm = df[df['port'].str.contains('\n')].to_dict(orient='records')
		d_sink = df[~df['port'].str.contains('\n')].to_dict(orient='records')

		tcl = []
		discretes = argd.get('discretes', pd.DataFrame())
		# shorts and discretes
		for ln in discretes.itertuples():
			if ln.use == '0':
				tcl += ["sigrity::update circuit -model {" + self.r0 + "} {" + ln.refdes + "} {!}"]
				continue
			if ln.use == ln.dcr and ln.dcr:
				#TODO: make sure gpn has right partical circuit with right DCR
				continue
			if ln.use == ln.acz and ln.acz:
				#TODO make sure acz models can be handled by powersi
				continue
				
		tcl = []
		for d in d_sink:
			refdes, pins = d['port'].split('.')[-2:]
			sink = f'SINK_{refdes}_{pins.split(',')[0]}'
			pos = {refdes: pins}
			neg = d.get('gnd',{})
			tcl +=  str_port( sink, pos, neg, 0.1)

		for d in d_vrm:
			lines = d['port'].splitlines()
			refdes, pins = lines[0].split('.')[-2:]
			vrm = f'VRM_{refdes}_{pins.split(',')[0]}'
			pos = {refdes: pins}
			neg = d.get('gnd',{})
			tcl +=  str_port( vrm, pos, neg, 0.1)
			if len(lines) <3:
				continue
			
			#vrm pos sense pin lines
			refdes, pins = lines[1].split('.')[-2:]
			sense = f'SNS_{refdes}_{pins.split(',')[0]}'
			pos = {refdes: pins}
			refdes, pins = lines[2].split('.')[-2:]
			neg = {refdes: pins}
			tcl +=  str_port( sense, pos, neg, 0.1)

		txt = rex(template.tcl_psi_irdrop.lstrip()).map(
				{
						"<OPEN_SPD>"		: str(fspd),
						"<SAVE_SPD>"		: str(fout),
						"<GROUND_NETS>"	: ground,
						"<POWER_NETS>"	: power,
						"<ENABLE>"			: _spacy(components),
						"<SINK_LINES>"	:	'\n\t'.join(tcl),
				}
		)		
		
		file = fspd.ext('.acdrop.tcl')
		file.write(txt)
		if file.recently():
			print(f"  tcl created: {file}")
			return str(file)
		return None
	
	def resipins(self, cell=None):
		if not cell:
			return None
		rails = dict(
				(cell["xnet"][i], i)
				for (i, x) in enumerate(cell["grp"])
				if x.startswith("rail") and cell["simulate"][i] == "TRUE"
		)
		power_nets, ground_nets = {}, {}
		soc, pmic, discretes = {}, {}, {}
		netlist = self.brd.netlist()
		dns = {}
		csv = template.csv_pdcresi
		for rail, i in rails.items():
			for k in cell["nets"][i].split(","):
				if k:
					power_nets.update({k: 1})
			for k in cell["ground"][i].split(","):
				if k:
					ground_nets.update({k: 1})
			for k in cell["soc"][i].split(","):
				if k:
					soc.update({k: 1})
			for k in cell["pmic"][i].split(","):
				if k:
					pmic.update({k: 1})
			if "{" in cell["sense"][i]:
				for k in eval(cell["sense"][i]):
					if k.endswith("_P"):
						power_nets.update({k: 1})
					elif k.endswith("_N"):
						ground_nets.update({k: 1})
			pins = []
			for p in cell["pins"][i].split(","):
				refdes, pin = p.split(".")
				for k in cell["dns"][i].split(","):
					if k:
						dns.update({k: 1})
				if refdes in dns:
					continue
				if not refdes in (soc | pmic):
					discretes.update({refdes: 1})
				pins.append(p)
			soc1st = list(soc.keys())[0]
			for x in pins:
				if x.startswith(soc1st):
					pins.insert(0, pins.pop(pins.index(x)))
					break

			for j, p in enumerate(pins[1:]):
				csv["Resistance Name"].append(f"{rail}_pin_{j:03d}")
				csv["Model"].append("L2L")
				csv["Positive Pin"].append(pins[0])
				csv["Negative Pin"].append(p)
				csv["ShortVRM"].append("1")
				csv["OtherCKT"].append("1")
		shorts = sorted([k for k in discretes if any(k.startswith(x) for x in ["L", "XW"])])
		caps = [k for k in discretes if any(k.startswith(x) for x in ["C"])]
		# add ground pins of soc, pmic, discretes
		components = caps + list(soc.keys()) + list(pmic.keys())
		ground_nets = list(ground_nets.keys())
		ground_pins = []
		for net in ground_nets:
			if net in netlist:
				for pin in netlist[net]:
					if any(pin.startswith(x) for x in components):
						ground_pins.append(pin)
		# add csv ground pins
		if True:
			for j, p in enumerate(ground_pins[1:]):
				csv["Resistance Name"].append(f"{ground_nets[0]}_pin_{j:03d}")
				csv["Model"].append("L2L")
				csv["Positive Pin"].append(ground_pins[0])
				csv["Negative Pin"].append(p)
				csv["ShortVRM"].append("1")
				csv["OtherCKT"].append("1")
		text = [",".join(list(csv.keys()))]
		for i in range(1, len(csv["Resistance Name"])):
			text.append(",".join([csv[k][i] for k in csv]))
		# write csv
		spd_file = pstr(self.spd.spd_file)
		csv_path = pstr(spd_file.sub(f"/SimFiles/{spd_file.file()}_resi"))
		csv_path.mkdir()
		csv_file = pstr(f"{csv_path}/{spd_file.file()}.resi.csv")
		csv_file.write("\n".join(text))

		# write .tcl file
		file = pstr(spd_file.sub(f"/Scripts/{spd_file}_resi.tcl"))
		txt = rex(template.tcl_pdc_resi.lstrip()).map(
				{
						"_FILE_NAME_": spd_file.base(),
						"_BOARD_PATH_": spd_file.path(),
						"_SIMULI_PATH_": spd_file.sub("/SimFiles"),
						"_FILE_CSV_": csv_file.base(),
						"_NOT_STUFF_": _spacy(dns),
						"_GROUND_NETS_": _tabby(ground_nets),
						"_POWER_NETS_": _tabby(power_nets),
						"_XW_COMPONENT_": _tabby(shorts),
						"_XW_MODEL_": self.r0,
				}
		)

		file.write(txt)
		if file.recently():
			print(f"  tcl created: {file}")
		return True

	def psi_clocks(self, cell):
		spd_file = pstr(self.spd.spd_file)
		switches = {**self.mux["mux"], **self.mux["spst"]} if self.mux else {}
		draft = template.tcl_psi_clock.lstrip()
		z = re.search(r"-MarginForCutByNet {(\d+) mm}", draft)
		cut_margin = int(z.group(1)) if z else 5
		all_shorts= {}		# all ground net bridging components
		
		for ln in cell.itertuples():
			xnet = re.sub(r'/(\d+)?$', '', ln.xnet)
			port_pins, port_num, max_dist = [], 0, 0
			tcl = "if {true} {\n\tsigrity::update circuit -manual {disable} -all {!}\n\t"
		# ios
			bigport, shorts = self.many_grounds(ln.io, ln.ground)
			if shorts:
				str_shorts = _spacy([ e for e in shorts])
				tcl += "sigrity::update circuit -manual {enable} " + str_shorts + " {!}\n\t"
				all_shorts.update({e:True for e in shorts})

			for cmppin in ln.io.split(","):
				if cmppin.count('.') != 1:
					print(f'bad component pin: {cmppin}')
					continue
				p_node, n_node, dist = self.spd.reference_node(cmppin, onpin=True, cut_margin=cut_margin)
				if not (p_node and n_node):
					print(f"tcl.psi_clocks(): node or reference_node not found for pin {cmppin}")
					continue
				if dist > max_dist:
					max_dist = dist
				port_name = cmppin.replace('.','_')
				tcl += "sigrity::add port -name {" + port_name + "}\n\t"
				tcl += "sigrity::hook -port {" + port_name + "} -PositiveNode {" + p_node + "}\n\t"
				n_nodes = _curly(bigport.get(_u(cmppin), n_node))
				tcl += "sigrity::hook -port {" + port_name + "} -NegativeNode "+ n_nodes +"\n\t"
				port_pins.append(port_name)
				port_num += 1

		# shunts
			shunt_dict = eval(ln.shunt) if ln.shunt else {}
			for pin, ref_net in shunt_dict.items():
				shunt,num = pin.split(".")[:2]
				if ref_net.lower() in ["gnd", "ground"]:
					tcl += "sigrity::add port -circuit " + shunt + " {!}\n\t"
					port_pins.append(f'{shunt}_{num}')
					port_num += 1
				else:
					net_pin = [x for x in ln.pins.split(",") if shunt in x]
					p_node, n_node, dist = self.spd.reference_node(net_pin[0], cut_margin=cut_margin)
					# if dist<0: continue
					if any(x.strip() == "" for x in (p_node, p_node)):
						print(f"tcl.psi_clocks(): node or reference_node not found for pin {net_pin[0]}")
					else:
						print(f"  pull up pin {net_pin[0]} is {dist} from reference {n_node}")
						port_name = f"{shunt}_{ref_net}"
						tcl += "sigrity::add port -name {" + port_name + "}\n\t"
						tcl += "sigrity::hook -port {" + port_name + "} -PositiveNode {" + p_node + "}\n\t"
						tcl += "sigrity::hook -port {" + port_name + "} -NegativeNode {" + n_node + "}\n\t"
						port_pins.append(f'{shunt}_{num}')
						port_num += 1

		# series
			mux_visited = {}
			for series in ln.series.split(","):
				if (not series) or (series in mux_visited):
					continue
				pins = re.findall(rf"{series}\.(\w+)", ln.pins)
				for pad in pins:
					p_node, n_node, dist = self.spd.reference_node(f"{series}.{pad}", onpin=True, cut_margin=cut_margin)
					# if dist<0: continue
					if dist > max_dist:
						max_dist = dist
					print(f"mux pin {series}.{pad} is {dist} from reference {n_node}")
					port_name = f"{series}_{pad}"
					tcl += "sigrity::add port -name {" + port_name + "}\n\t"
					tcl += "sigrity::hook -port {" + port_name + "} -PositiveNode {" + p_node + "}\n\t"
					tcl += "sigrity::hook -port {" + port_name + "} -NegativeNode {" + n_node + "}\n\t"
					port_pins.append(port_name)
					port_num += 1

				if series in switches:
					mux_visited.update({series: True})

		# dns: always create port on 1st TP event dns'ed
			for part in ln.dns.split(","):
				if re.search(r"^TP\d+", part, re.I) and part in ln.pins:
					tp_pin = re.findall(rf"({part}\.\w+)", ln.pins)
					p_node, n_node, dist = self.spd.reference_node(tp_pin[0], cut_margin=cut_margin)
					# if dist<0: continue
					if dist > max_dist:
						max_dist = dist
					print(f"  test pin {tp_pin[0]} is {dist} from reference {n_node}")
					port_name = tp_pin[0].replace(".", "_")
					tcl += "sigrity::add port -name {" + port_name + "}\n\t"
					tcl += "sigrity::hook -port {" + port_name + "} -PositiveNode {" + p_node + "}\n\t"
					tcl += "sigrity::hook -port {" + port_name + "} -NegativeNode {" + n_node + "}\n\t"
					port_pins.append(port_name)
					port_num += 1
					break  # add just 1 TP
		# dns: always create 2nd port on dns'ed part if just one io, to avoid dcfitting problem of s1p
			if ln.dns and port_num < 2:
				for part in ln.dns.split(",")[:1]:
					if part in ln.pins:
						tp_pin = re.findall(rf"({part}\.\w+)", ln.pins)
						p_node, n_node, dist = self.spd.reference_node(tp_pin[0], cut_margin=cut_margin)
						# if dist<0: continue
						if dist > max_dist:
							max_dist = dist
						print(f"  probe pin {tp_pin[0]} is {dist} from reference {n_node}")
						port_name = tp_pin[0].replace(".", "_")
						tcl += "sigrity::add port -name {" + port_name + "}\n\t"
						tcl += "sigrity::hook -port {" + port_name + "} -PositiveNode {" + p_node + "}\n\t"
						tcl += "sigrity::hook -port {" + port_name + "} -NegativeNode {" + n_node + "}\n\t"
						port_pins.append(port_name)
						port_num += 1
						break  # add just 1 probe

		# flush tcl file
			if xnet:
				nets = ln.nets.split(",")
				grounds = ln.ground.split(",")
				margin = f"{0.5 + (max_dist if cut_margin < max_dist else cut_margin)}"
				tcl += "sigrity::update option -MarginForCutByNet {" + margin + " mm} {!}\n\t"
				tcl += "sigrity::delete area -Net " + _curly(nets) + " {!}\n}"
				tcl += "\n#"+','.join(port_pins)
				txt = rex(draft).map(
						{
								"_FILE_NAME_": spd_file.base(),
								"_BOARD_PATH_": spd_file.path(),
								"_SIMULI_PATH_": spd_file.sub("/SimFiles"),
								"_EXTENDED_NET_": xnet,
								"_GROUND_NETS_": "\n\t".join(grounds),
								"_POWER_NETS_": "",
								"_SIGNAL_NETS_": "\n\t".join(nets),
								"_PORTS_TCL_": tcl,
						}
					)
				file = pstr(spd_file.sub(f"/Scripts/{xnet}.tcl"))
				file.write(txt)
				if file.recently():
					print(f"  tcl created: {file}")
		if len(all_shorts):
			self.spd.short_grounds(all_shorts)			
		return {}

	def clr_clocks(self, cell):

		spd_file = pstr(self.spd.spd_file)
		rex_gnd = re.compile(self.spd.gnd_rex, re.I)
		port_tcls = {}
		all_shorts = {}
		draft = template.tcl_clr_clock.lstrip()
		z = re.search(r"-MarginForCutByNet {(\d+) mm}", draft)
		cut_margin = int(z.group(1)) if z else 5
		for ln in cell.itertuples():
			xnet = ln.xnet
			nets = ln.nets.split(",")
			tcl = "if {true} {\n\tsigrity::update circuit -manual {disable} -all {!}\n"
			bigport, shorts = self.many_grounds(ln.io, ln.ground)
			if shorts:
				str_shorts = _spacy([ e for e in shorts])
				tcl += "sigrity::update circuit -manual {enable} " + str_shorts + " {!}\n\t"
				all_shorts.update({e:True for e in shorts})
			gndnet, ports, port_num = [], [], 1
			io_gndnets = {}
			max_dist = 0
			#io
			for io_pin in ln.io.split(","):
				port_p, gnodes, dist = self.spd.reference_node(io_pin, nearest=False, cut_margin=cut_margin)
				# if dist<0: continue
				port_n = gnodes[0] if type(gnodes) == list else gnodes
				port_name = f"Port{port_num}_" + io_pin.replace(".", "_") + "::" + port_p.split(":")[-1]
				ports.append([port_name, port_p, port_n])
				gndnet.append(port_n.split(":")[-1])
				#
				gnd_nodes = gnodes if type(gnodes) == list else [gnodes]
				gnd_nets = {}
				for w in [x.split(":")[-1] for x in gnd_nodes]:
					if rex_gnd.match(w):
						gnd_nets.update({w: 1})
				io_gndnets.update({port_num: list(gnd_nets.keys())})
				port_num += 1

			coaxial = -1
			if len(set(gndnet)) > 1:
				print("ports referencing multiple grounds:" + ",".join(gndnet))
				coaxial = max(io_gndnets.keys(), key=lambda x: len(io_gndnets[x]))
				# the ic with most varity of grounds get a coaxial port!

			#shunts
			ref_dict = eval(ln.shunt) if ln.shunt else {}
			for ref_pin, ref_net in ref_dict.items():
				shunt = ref_pin.split(".")[0]
				net_pin = list(filter(lambda x: shunt in x, ln.pins.split(",")))[0]
				if rex_gnd.match(ref_net):
					port_p, port_n = (
							self.spd.sig_pin_node(net_pin),
							self.spd.sig_pin_node(ref_pin),
					)
					port_name = f"Port{port_num}_" + net_pin.replace(".", "_") + "::" + port_p.split(":")[-1]
				else:
					port_p, port_n, dist = self.spd.reference_node(net_pin, cut_margin=cut_margin)
					# if dist<0: continue
					if dist > max_dist:
						max_dist = dist
					print(f"  pull up: {net_pin} is {dist} from reference {port_n}")
					port_name = f"Port{port_num}_{shunt}_{ref_net}"
				ports.append([port_name, port_p, port_n])
				port_num += 1
			
			#series
			for series in ln.series.split(","):
				pads = re.findall(rf"{series}\.(\w+)", ln.pins)
				for pad in pads:
					port_name = f"Port{port_num}_{series}_{pad}"
					port_p, gnodes, dist = self.spd.reference_node(f"{series}.{pad}", nearest=True, cut_margin=cut_margin)
					# if dist<0: continue
					port_n = gnodes[0] if type(gnodes) == list else gnodes
					port_name += "::" + port_p.split(":")[-1]
					ports.append([port_name, port_p, port_n])
					# gndnet.append(port_n.split(':')[-1])
					port_num += 1

			for count, (port_name, port_p, port_n) in enumerate(ports):
				if count + 1 == coaxial:
					print(f"  port: {port_name} is coxial")
					ic = ln.io.split(",")[count].split(".")[0]
					tcl += "\tsigrity::add 3DFEMPort -circuit {" + ic + "} -PortType {coaxial} -AddSolderBallBump {1} {!}\n"
					tcl += "\tset fem_port [sigrity::querydetails port -index {" + f"{count}" + "}]\n"
					tcl += "\tset fem_name [lindex $fem_port 0]\n\t"
					tcl += "\tsigrity::update port -name $fem_name -NewName {" + port_name + "} {!}\n"
					continue
				tcl += "\tsigrity::add port -name {" + port_name + "}\n"
				tcl += "\tsigrity::hook -port {" + port_name + "}  -PositiveNode {" + port_p + "}\n"
				tcl += "\tsigrity::hook -port {" + port_name + "}  -NegativeNode {" + port_n + "}\n"
				tcl += "\tsigrity::update port -name {" + port_name + "} -RefZ {50} {!}\n"
				tcl += "\tsigrity::update port -name {" + port_name + "} -disabled 0 {!}\n"

			new_margin = 0.5 + (max_dist if cut_margin < max_dist else cut_margin)
			tcl += "\tsigrity::update option -MarginForCutByNet {" + f"{new_margin}" + " mm} {!}\n"
			tcl += "\tsigrity::delete area -Net " + _curly(nets) + " {!}\n}"
			xgnd = ln.ground.replace(",", "\n\t")
			# xpwr = cell['power'][i].replace(',','\n\t')
			clks = ln.nets.replace(",", "\n\t")
			# port_tcls[xnet] = [xgnd, xpwr, clks, tcl[:-1]]
			port_tcls[xnet] = (xgnd, clks, tcl)

		for xnet in port_tcls:
			file = pstr(spd_file.sub(f"/Scripts/{xnet}.tcl"))
			# xgnd, xpwr,clks,ports= port_tcls[xnet]
			xgnd, clks, tcl = port_tcls[xnet]
			txt = rex(draft).map(
					{
							"_FILE_NAME_": spd_file.base(),
							"_BOARD_PATH_": spd_file.path(),
							"_SIMULI_PATH_": spd_file.sub("/SimFiles"),
							"_EXTENDED_NET_": xnet,
							"_GROUND_NETS_": xgnd,
							"_POWER_NETS_": "",
							"_SIGNAL_NETS_": clks,
							"_PORTS_TCL_": tcl,
					}
			)
			file.write(txt)
			if file.recently():
				print(f"  tcl created: {file}")

		if len(all_shorts):
			self.spd.short_grounds(all_shorts)			
		return {}

	def psi_serdes(self, cell=None):

		def _fl(p):
			return re.match(r"^(F)?L", p, re.I)  # cmc starts with L or FL

		spd_file = pstr(self.spd.spd_file)
		rex_gnd = re.compile(self.spd.gnd_rex, re.I)

		switches = {**self.mux["mux"], **self.mux["spst"]} if self.mux else {}
		all_series = (",".join(cell["series"])).split(",")
		cmc = dict((k, True) for k in all_series if _fl(k) or k.startswith("U"))
		switches.update(cmc)

		draft = template.tcl_psi_serdes.lstrip()
		z = re.search(r"-MarginForCutByNet {(\d+) mm}", draft)
		cut_margin = int(z.group(1)) if z else 5
		buses = {grp: dsn for (grp, dsn) in zip(cell["grp"], cell["design"])}
		for bus, design in buses.items():
			rows = [x for x in range(len(cell["xnet"])) if bus == cell["grp"][x]]
			max_dist = 0
			file = pstr(spd_file.sub(f"/Scripts/{bus}_{design}.tcl"))
			print(f"  tcl.psi_serdes(): {file}", end="")
			series, shunts, supply, nets, ic_pins, jumpers = [], [], {}, [], [], {}
			for i in rows:
				nets += _cplit(cell["nets"][i])
				ser_prts = _cplit(cell["series"][i])
				pins = _cplit(cell["pins"][i])

				if HS_CREATE_JUMER_PORTS and switches:
					for p in filter(lambda x: x in switches, series):
						j = [x.split(".")[-1] for x in pins if x.startswith(f"{p}.")]
						jumpers[p] = jumpers[p] + j if p in jumpers else j
					series += [x for x in ser_prts if x not in jumpers]
				else:
					series += ser_prts

				if cell["shunt"][i]:
					d = eval(cell["shunt"][i])
					for k, v in d.items():
						shunts.append(_u(k))
						supply.update({v: True})
				if cell["ground"][i]:
					supply.update({e: True for e in _cplit(cell["ground"][i])})

				# add dns or tp if io is 1
				io = sorted(_cplit(cell["io"][i]))
				if len(io) < 2:
					dns_pins = [x for x in pins if x.startswith(tp)] if (tp := cell["dns"][i]) else []
					tp_pins = [x for x in pins if re.search(r"^tp", x, re.I)]
					io += dns_pins[:1] if dns_pins else (tp_pins[:1] if tp_pins else [])
				# always have attached pins in right ports
				u = {_u(x): True for x in io}
				if len(u) != 2:
					print(f"\n  ? bus {bus} has {len(u)} parts, expecting 2")
				end = []
				for att in _cplit(cell["attached"][i]):
					end.append(".".join(att.split(".")[1:]))
				if not end:
					end = io[-1:]
				ic_pins += [[e for e in io if e not in end], end] if (end and len(io) > 1) else [io]

			decaps = []
			gnds = [e for e in supply if rex_gnd.match(e)]
			power = [e for e in supply if e not in gnds]
			if len(gnds) and len(power):
				for c in self.spd.data["compval"]:
					pat = re.compile("^" + c + ".", re.I) if c.startswith("C") else None
					if pat is None:
						continue
					cpin = [e for e in self.spd.data["comppin"] if pat.match(e)]
					if cpin and all([self.spd.data["comppin"][e]["net"] in supply for e in cpin]):
						decaps.append(c)

			tcl = "if {true} {\n\tsigrity::update circuit -manual {disable} -all {!}\n"
			if len(series):
				tcl += "\tsigrity::update circuit -manual {enable} " + _spacy(series) + " {!}\n"
			if len(shunts):
				tcl += "\tsigrity::update circuit -manual {enable} " + _spacy(shunts) + " {!}\n"
			if len(decaps):
				tcl += "\tsigrity::update circuit -manual {enable} " + _spacy(decaps) + " {!}\n"

			port_num = 0
			for pins in ic_pins:
				p_nodes, n_nodes = [], []
				for pin in map(lambda x: x, filter(None, pins)):
					pp, nn, dist = self.spd.reference_node(pin, nearest=2, cut_margin=5)
					# if dist<0: continue
					if dist > max_dist:
						max_dist = dist
					p_nodes.append(pp)
					n_nodes.extend(nn)
				if not (len(p_nodes) and len(n_nodes)):
					print(
							f"\n  ?ports not created for bus {bus} pins {_spacy(ic_pins)}",
							end="",
					)
					continue
				port_num += 1
				port_name = f"Port{port_num}_{pins[0]}::" + p_nodes[0].split(":")[-1]
				tcl += "\tsigrity::add port -name {" + port_name + "}\n"
				for p_node in set(p_nodes):
					tcl += "\tsigrity::hook -port {" + port_name + "} -PositiveNode {" + p_node + "}\n"
				for n_node in set(n_nodes):
					tcl += "\tsigrity::hook -port {" + port_name + "} -NegativeNode {" + n_node + "}\n"
				# print(f'\t{port_name}')
			for part, pins in jumpers.items():
				for pin in pins:
					p_node, n_nodes, dist = self.spd.reference_node(f"{part}.{pin}", nearest=4, cut_margin=5)
					# if dist<0: continue
					if dist > max_dist:
						max_dist = dist
					port_num += 1
					port_name = f"Port{port_num}_{part}_{pin}::" + p_node.split(":")[-1]
					tcl += "\tsigrity::add port -name {" + port_name + "}\n"
					tcl += "\tsigrity::hook -port {" + port_name + "} -PositiveNode {" + p_node + "}\n"
					for n_node in set(n_nodes):
						tcl += "\tsigrity::hook -port {" + port_name + "} -NegativeNode {" + n_node + "}\n"
					# print(f'\t{port_name}')

			new_margin = 0.5 + (max_dist if cut_margin < max_dist else cut_margin)
			tcl += "\tsigrity::update option -MarginForCutByNet {" + f"{new_margin}" + " mm} {!}\n"
			if len(power):
				tcl += "\tsigrity::delete area -Net {PowerNets} " + _curly(nets) + " {!}\n}"
			else:
				tcl += "\tsigrity::delete area -Net " + _curly(nets) + " {!}\n}"

			txt = rex(draft).map(
					{
							"_FILE_NAME_": spd_file.base(),
							"_BOARD_PATH_": spd_file.path(),
							"_SIMULI_PATH_": spd_file.sub("/SimFiles"),
							"_EXTENDED_NET_": file.name(),
							"_POWER_NETS_": _tabby(power),
							"_GROUND_NETS_": _tabby(gnds),
							"_SIGNAL_NETS_": _tabby(nets),
							"_PORTS_TCL_": tcl,
					}
			)
			file.write(txt)
			if file.recently():
				print(f" - created")

		return {}

	def clr_serdes(self, cell=None):

		COAXIAL_PORT = False
		fem_opt = "sigrity::update option -Wave3DSettingsolutionfreq {1.8e+10} -Wave3DFreqBand {{1e+08 2.5e+10 linear 1e+07}{0 1e+07 linear 1e+06}} -Wave3DSettingDCRefinement {1} -PowerNetImpedance {0.1} -ResultFileHasBnp {1} -ResultFileFreqUnit {38} -Wave3DSettingminimumAdaptiveIterations {2} -Wave3DSettingminimumConvergedIterations {2} {!}\n"
		spd_file = pstr(self.spd.spd_file)
		rex_gnd = re.compile(self.spd.gnd_rex, re.I)
		draft = template.tcl_psi_serdes.lstrip()

		buses = {grp: dsn for (grp, dsn) in zip(cell["grp"], cell["design"])}
		for bus, design in buses.items():
			rows = [x for x in range(len(cell["xnet"])) if bus == cell["grp"][x]]
			max_dist = 0
			file = pstr(spd_file.sub(f"/Scripts/{bus}_{design}.tcl"))
			print(f"  tcl.clr_serdes(): {file}", end="")
			series, shunts, supply, nets, ic_pins = [], [], {}, [], []
			for i in rows:
				nets += _cplit(cell["nets"][i])
				series = _cplit(cell["series"][i])
				pins = _cplit(cell["pins"][i])
				if cell["shunt"][i]:
					d = eval(cell["shunt"][i])
					for k, v in d.items():
						shunts.append(_u(k))
						supply.update({v: True})
				if cell["ground"][i]:
					supply.update({e: True for e in _cplit(cell["ground"][i])})

				# add dns or tp if io is 1
				io = _cplit(cell["io"][i])
				if len(io) < 2:
					dns_pins = [x for x in pins if x.startswith(tp)] if (tp := cell["dns"][i]) else []
					tp_pins = [x for x in pins if re.search(r"^tp", x, re.I)]
					io += dns_pins[:1] if dns_pins else (tp_pins[:1] if tp_pins else [])
				io = sorted(io)
				# always have attached pins in right ports
				u = {_u(x): True for x in io}
				if len(u) != 2:
					print(
							f"\n  ? interface `{cell['grp'][i]}` has {len(u)} parts, expecting 2",
							end="",
					)
				end = []
				for att in cell["attached"][i].split(","):
					end.append(".".join(att.split(".")[1:]))
				if not end:
					end = io[-1:]
				ic_pins += [[e for e in io if e not in end], end] if end and len(io) > 1 else [io]

			decaps = []
			gnds = [e for e in supply if rex_gnd.match(e)]
			power = [e for e in supply if e not in gnds]
			if len(gnds) and len(power):
				for c in self.spd.data["compval"]:
					pat = re.compile("^" + c + ".", re.I) if c.startswith("C") else None
					if pat is None:
						continue
					cpin = [e for e in self.spd.data["comppin"] if pat.match(e)]
					if cpin and all([self.spd.data["comppin"][e]["net"] in supply for e in cpin]):
						decaps.append(c)
			tcl = fem_opt if COAXIAL_PORT else ""
			tcl += "if {true} {\n\tsigrity::update circuit -manual {disable} -all {!}\n"
			if len(series):
				tcl += "\tsigrity::update circuit -manual {enable} " + _spacy(series) + " {!}\n"
			if len(shunts):
				tcl += "\tsigrity::update circuit -manual {enable} " + _spacy(shunts) + " {!}\n"
			if len(decaps):
				tcl += "\tsigrity::update circuit -manual {enable} " + _spacy(decaps) + " {!}\n"
			# if len(gnds):
			# 	tcl+= '\tsigrity::move net {GroundNets} ' + _curly(gnds) + ' {!}\n'
			# 	tcl+= '\tsigrity::update net selected 1 ' + _curly(gnds) + ' {!}\n'
			# if len(power):
			# 	tcl+= '\tsigrity::move net {PowerNets} ' + _curly(power) + ' {!}\n'
			# 	tcl+= '\tsigrity::update net selected 1 ' + _curly(power) + ' {!}\n'
			# if len(nets):
			# 	tcl+= '\tsigrity::move net {} ' + _curly(nets) + ' {!}\n'
			# 	tcl+= '\tsigrity::update net selected 1 ' + _curly(nets) + ' {!}\n'

			if bool(COAXIAL_PORT):
				devices = {_u(e): True for icpin in ic_pins for e in icpin}
				for part in devices:
					tcl += f"\tsigrity::add 3DFEMPort -circuit {part} " + "-PortType {coaxial}"
			else:
				port_num = 0
				for pins in ic_pins:
					p_nodes, n_nodes = [], []
					for pin in map(lambda x: x, filter(None, pins)):
						pp, nn, dist = self.spd.reference_node(pin, nearest=1, cut_margin=5)
						# if dist<0: continue
						if dist > max_dist:
							max_dist = dist
						p_nodes.append(pp)
						n_nodes.extend(nn)
					if not (len(p_nodes) and len(n_nodes)):
						print(
								f"\n   ? ports not created for bus {bus} pins {_spacy(ic_pins)}",
								end="",
						)
						continue
					port_num += 1
					port_name = f"Port{port_num}_{pins[0]}::" + p_nodes[0].split(":")[-1]
					tcl += "\tsigrity::add port -name {" + port_name + "}\n"
					for p_node in set(p_nodes):
						tcl += "\tsigrity::hook -port {" + port_name + "} -PositiveNode {" + p_node + "}\n"
					for n_node in set(n_nodes):
						tcl += "\tsigrity::hook -port {" + port_name + "} -NegativeNode {" + n_node + "}\n"
					# print(f'\t{port_name}')

			tcl += "\tsigrity::delete area -Net " + _curly(nets) + " {!}\n}"

			txt = rex(draft).map(
					{
							"_FILE_NAME_": spd_file.base(),
							"_BOARD_PATH_": spd_file.path(),
							"_SIMULI_PATH_": spd_file.sub("/SimFiles"),
							"_POWER_NETS_": _tabby(power),
							"_GROUND_NETS_": _tabby(gnds),
							"_SIGNAL_NETS_": _tabby(nets),
							"_EXTENDED_NET_": file.name(),
							"_PORTS_TCL_": tcl,
					}
			)
			file.write(txt)
			if file.recently():
				print(f"   - created")

		return {}
