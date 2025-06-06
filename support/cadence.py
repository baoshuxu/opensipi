import os
import re
import time
import glob
import shutil
import textwrap
import pandas as pd
from .templates import template
from .folders import folders, logme, dos, scheduler, pstr, assert_os_variable
import xml.etree.ElementTree as ET

PIN_TYPES_REGEX = {
		"dsc": r"^(r|l|c|fb|d|q|xw)\d+",  # discretes
		"cmc": r"^(fl)\d+",  							# cmc
		"rsns": r"^(rs)\d+",  						# sense resistors
		"tp": r"^(m)?t(p)?",  						# test points
		"ic": r"^(u|y|vcm)",  						# drivers receivers
		"io": r"^(m|j|p|cn|con)",  				# connectors
}


class engnum:
	scale = {
			"f": 1e-15,
			"p": 1e-12,
			"n": 1e-9,
			"u": 1e-6,
			"m": 1e-3,
			"1": 1e0,
			"K": 1e3,	'k': 1e3,
			"M": 1e6,	'X': 1e6,
			"G": 1e9, "g": 1e9,
			"T": 1e12,
			"P": 1e15,
		}

	def __init__(self, number=""):
		#regex = re.compile(r"([-+]?[\d\.]*([eE][-+]?\d+)?)([A-Za-z]*)")
		regex =  re.compile(r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)([a-zA-Z]*)")
		self.number = number
		self.value = None
		if isinstance(number, (int, float)):
			self.value = number
		elif isinstance(number, str):
			# num, eng, unit = regex.search(number).groups()
			# if num:
			if( z:= regex.fullmatch(number.strip()) ):
				num, unit = z.groups()
				u = unit[0] if unit else None
				self.value = float(num) * engnum.scale.get(u, 1.0)
			else:
				self.value = float("nan")


rex_numbers = re.compile(r"[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?", re.VERBOSE)
rex_part_numbers = re.compile(r"^(G?)(\d{3}-\d{5}-\d{2})", re.I)

def cdns_xnet_mode():
	# cadence xnet generation by xnet_pin or dml
	if not assert_os_variable('CDS_XNET_STATE_UI','1'):
		return 'dml'
	return 'xpin'
	

def scaled_int(str, ROUND_UP_SCALE=1):
	v = rex_numbers.search(str).group()
	return int(float(v) * ROUND_UP_SCALE)


def csv2dict(csv, add_key=""):
	if not os.path.exists(csv):
		return None
	with open(csv, "r") as f:
		text = f.read().splitlines()
	text = [x.strip() for x in text]
	header = [x.strip() for x in text[0].split(",")]
	if add_key:
		header.append(str(add_key).strip())
		text = [x + "," for x in text]
	dictobj = dict(zip(header, [[] for _ in range(len(header))]))
	for line in text[1:]:
		for i, v in enumerate(line.split(",")):
			dictobj[header[i]].append(v.strip())
	return dictobj


def read_stackup(cell=[]):
	"""report .cmx and .csv string for spd given text cell"""

	def metal(mat_info):
		temp_mat = '<Material name="MAT_NAME">\n\t<Metal>\n\t\t<Model>\
			\nTEMP CONDUCTIVITY\n\t\t</Model>\n\t</Metal>\n</Material>\n'
		temp_mat = temp_mat.replace("MAT_NAME", mat_info[0])
		temp_mat = temp_mat.replace("TEMP", "20")
		temp_mat = temp_mat.replace("CONDUCTIVITY", mat_info[2])
		return temp_mat

	def diel(mat_info):
		temp_mat = '<Material name="MAT_NAME">\n\t<Dielectric>\n\t\t<Model>\
			\nFreq Diek Disf\n\t\t</Model>\n\t</Dielectric>\n</Material>\n'
		temp_mat = temp_mat.replace("MAT_NAME", mat_info[0])
		temp_mat = temp_mat.replace("Freq", mat_info[3])
		temp_mat = temp_mat.replace("Diek", mat_info[4])
		temp_mat = temp_mat.replace("Disf", mat_info[5])
		return temp_mat

	cmx, csv = ("", "")
	if bool(cell):
		i, j = (0, 0)  # index bounding meterial data, after j is stackup info
		for k, line in enumerate(cell):
			if line.startswith("StackUp"):
				csv = "\n".join(cell[k + 1 :])
				j = k if not j else j
				break
			if line.startswith("Material"):
				i = k + 1 if not i else i
			if line.startswith(","):
				j = k if not j else j

		material = ""
		for line in cell[i + 1 : j]:
			row = line.split(",")
			if row[1] == "Dielectric":
				material += diel(row)
			elif row[1] == "Metal":
				material += metal(row)
		cmx = template.material_cmx.lstrip()
		cmx = cmx.replace("ADD_MATERIAL", material)
	return cmx, csv


def run_skill(app, brd, il):
	"""app- the allegro.exe path
	brd- the brd file path
	il - a skill string joined with \n or a file
	"""
	pattern = r'\s+(?=(?:[^"]*"[^"]*")*[^"]*$)'
	exe = re.split(pattern, app)[0]
	if not all(pstr(x).isfile for x in (exe, brd)):
		print(f"cadence.run_skill(): {exe} or {brd} not accessable")
		return
	if pstr(il).isfile:
		fil, cwd = il, os.path.dirname(il)
	else:
		cwd = os.path.dirname(brd)
		fil = f"{cwd}/extract.il"
		pstr(fil).write(il)

	# copy brd file if currently open in allegro
	lck = brd + ".lck" if pstr(brd + ".lck").isfile else ""
	tgt = lck + ".brd" if lck else brd
	if lck:
		try:
			shutil.copy(brd, tgt)
		except Exception as e:
			print(f"cadence.run_skill(): shutil error- {e}")
			return
	if not all(pstr(x).isfile for x in (tgt, fil)):
		print(f"cadence.run_skill(): {tgt} or {fil} not found")
		return

	# create script file which calls the skill file fil
	scr = pstr(fil + ".scr")
	scr.write(f'open "{tgt}"\nskill load("{fil}")\nexit\n')
	if not scr.recently:
		print(f"cadence.run_skill(): {scr} not created")
		return

	# actually batch command
	command = f'{app} -nograph -p "{cwd}" -s "{scr}"'
	dos(command, cwd=cwd, timeout=600)

	# clean up
	clr = [scr] if pstr(il).isfile else [scr, pstr(fil)]
	clr += [pstr(tgt)] if lck else []
	for f in clr:
		f.remove()

	return


def spd_workflow(tool, file):
	# modify spd file to agree with workflow of the tool,otherwise got popup windows
	codes = {
		"powersi": "0x100000067", 
		"clarity": "0x10000006b",
		"powerdc": "0x1000000067"
		}
	if "Sigrity" in tool and "PowerSI.exe" in tool:
		code = codes["powersi"]
	elif "Sigrity" in tool and "Clarity3DLayout.exe" in tool:
		code = codes["clarity"]
	elif "Sigrity" in tool and "PowerDC.exe" in tool:
		code = codes["powerdc"]
	else:
		return None

	file = str(file)
	# fastest method of modifying only one line of a file is write/rename
	f_temp = file + ".tmp"
	with open(file, "r") as f_input:
		line1 = next(f_input, None)
		line2 = None
		if line1 and line1.startswith("Title WorkflowKey"):
			z = re.search(r"^Title WorkflowKey = ([\S]+) .*", line1)
			if z and z.group(1) != code:
				line2 = re.sub(z.group(1), code, line1, count=0)
		if line2:
			with open(f_temp, "w") as f_output:
				f_output.write(line2 + "\n")  # Write the new first line
				for line in f_input:
					f_output.write(line)  # Copy the rest of the lines
	if all(map(os.path.isfile, [file, f_temp])):
		os.remove(file)
		os.replace(f_temp, file)
	return file

class ammx:
	def __init__(self, f_ammx=''):
		self.xml = pstr(f_ammx)
		self.r = {}		# {gpn: (pin1, pin2, dcr, cktdef)}
		self.c = {}   # {gpn: (cap, volt, cktdef) }
		self.l = {}   # {gpn: (dcr, inductance, maxurrent)}
		self.b = {}		# {gpn: (ibs, components)}
		self.x = {} 	# {gpn: (model_name, cir_file)}
		self.s = {}		# {gpn: (num_ports, ts_file)}, to be implemented
		self.pp = {}  # {gpn: (1 3)(2 4)}, thur pin pair

	def resistor(self):
		return( self.r or self.parse_tree('r') )
	
	def inductor(self):
		return( self.l or self.parse_tree('l') )

	def capacitor(self):
		return( self.c or self.parse_tree('c') )
	
	def integrated(self):
		return( self.b or self.parse_tree('b') )
	
	def subcircuit(self):
		return( self.x or self.parse_tree('x') )
	
	def touchstone(self):		# TBD
		return {}
		#return( self.s or self.parse_tree('s') )
	
	def parse_tree(self, o='c'):
		if not self.xml.isfile:
			return {}
		tree = ET.parse(str(self.xml))
		root = tree.getroot()
		for e in root.findall('.//IC'):
			gpn = e.attrib.get('ModelName')
			io = e.find('IO_MODEL')
			if io is not None:
				self.b.update({gpn: (io.attrib.get('IBIS'),  io.attrib.get('Components'))})
		
		for e in root.findall('.//SPICE'):
			gpn = e.attrib.get('ModelName')
			self.x.update({gpn: (e.attrib.get('Model'), e.attrib.get('SPICE_File'))})	
		
		for e in root.findall('.//Inductor'):
			gpn = e.attrib.get('ModelName')

			self.l.update({gpn: (e.attrib.get('Resistance'), e.attrib.get('Inductance'), e.attrib.get('MaxCurrent'))})	
			thru = e.find('.//ThroughInfo')
			if thru is not None:
				dcr = ( thru.attrib.get('Pin1'), thru.attrib.get('Pin2'), thru.attrib.get('Resistance'))	
			else:
				dcr=('1','2', engnum(e.attrib.get('Resistance','0')).value )
			pair1 = [re.split(r'[\s,;]+',s)[0] for s in dcr[:2]]
			if all( e for e in pair1): # avoid pin ==''
				self.pp[gpn] = self.pp.get(gpn,'') + f'({pair1[0]} {pair1[1]})'

		for e in root.findall('.//Capacitor'):
			gpn = e.attrib.get('ModelName')
			self.c.update({gpn: (e.attrib.get('Cnom'), e.attrib.get('Volt'), e.attrib.get('CktDef'))})		
		
		for e in root.findall('.//Resistor'):
			gpn = e.attrib.get('ModelName')
			if gpn=='G480-01227-00':
				print('ho')			
			ckt = e.attrib.get('CktDef')
			thru = e.find('.//ThroughInfo')
			if thru is not None:
				dcr = ( thru.attrib.get('Pin1'), thru.attrib.get('Pin2'), thru.attrib.get('Resistance'))
			else:
				dcr=('1','2', engnum(e.attrib.get('Resistance','-1')).value )
			self.r.update({gpn: dcr+(ckt,)})
			pair1 = [re.split(r'[\s,;]+',s)[0] for s in dcr[:2]]
			if all( e for e in pair1):  # avoid pin ==''
				self.pp[gpn] = self.pp.get(gpn,'') + f'({pair1[0]} {pair1[1]})'
		
		return getattr(self, o) if hasattr(self,o) else None

	@property
	def xpin(self):
		""" report thru pins ( xpin fromat)"""
		pp = self.pp or self.parse_tree('pp')
		return ','.join(f'{k}={v}' for (k,v) in pp.items()) if pp else ''

class skill:

	def __init__(self, app, brd):
		self.app = app
		self.brd = brd

	def run(self, il):
		run_skill(self.app, self.brd, il)

class extnet:

	def __init__(self, nets={}, cpn={}, inc_pwr=False):
		self.nets = nets
		self.cpn = cpn
		self.seen = {}
		self.flat = []
		self.inc_pwr = inc_pwr

	def flatten(self, nested):
		"""flatten the nested list"""
		for e in nested:
			if isinstance(e, list):
				self.flatten(e)
			else:
				self.flat.append(e)
		return self.flat

	def use(self, pin):
		return (pin["volt"] != "0") if self.inc_pwr else (pin["volt"] == "")

	def bubble(self, net=""):
		"""find xnet with recursive method"""
		pins = []
		for a in self.nets.get(net,[]):
			if self.seen[a] or self.use(self.cpn[a]) is False:
				continue
			pins.append(a)
			self.seen[a] = True
			# (thru pin) and (not nc) and (not voltage)
			if (num:=self.cpn[a].get('thru')):
				b = f"{a.partition('.')[0]}.{num}"
				if self.cpn[b].get('net') and self.use(self.cpn[b]):
					pins.append(self.bubble(self.cpn[b]['net']))
		return pins

	def find(self, net=""):
		self.seen = {k: False for k in self.cpn}
		pins = self.bubble(net)
		self.flat = []
		flat = self.flatten(pins)
		return pins, flat

class functionpin:

	def __init__(self, fpn=None, dns=None, xpin=None):
		self.nets = {}  # netlist
		self.cpn = {}  # component pin
		self.xnet = {}  # xnets
		self.vnet = {}  # voltage nets
		self.gpn = {}  # google partno

		if isinstance(fpn, (dict, pstr, str)):  # initiate with given cpn
			self._cpn(fpn)
		else:
			return None

		self._thru()

		if dns is not None:
			self._dns(dns)
		if xpin is not None:
			self._xpin(xpin)

	def trace(self, dc={}, flt={}, inc_pwr=False, xpin=None):
		if xpin is not None:
			self._xpin(xpin)

		def nonet(k):
			return k in self.vnet and inc_pwr is False

		fltrd = dict((e, True) for e in self.nets)

		if bool(flt):
			for net in fltrd:
				pat = [e for e in flt if re.match(e, flt, re.I)]
				if len(pat) < 1:
					fltrd[net] = False

		if bool(dc):
			self.vnet.update(self._dc(dc))

		if bool(self.vnet):
			seen = dict((k, True if nonet(k) else False) for k in self.nets)
			e = extnet(self.nets, self.cpn, inc_pwr)
			for net, do in fltrd.items():
				if do and seen[net] is False:
					_, pins = e.find(net)
					nets = set([self.cpn[p]["net"] for p in pins])
					for n in nets:
						seen[n] = True
					self.xnet.update({",".join(sorted(nets)): ",".join(sorted(pins))})
		else:
			self.xnet = dict((k, ",".join(sorted(v))) for (k, v) in self.nets.items())

		return self.xnet

	def kinds(self, pins):
		def _u(p):
			return p.split(".")[0]  # refdes R1000

		def _n(p):
			return self.cpn[p]["thru"]  # thru pin number 2

		def _b(p):
			return _u(p) + "." + _n(p)  # thur pin	R1000.2

		def v(p):
			return self.cpn[p]["volt"]

		def tp(p):
			return self.cpn[p]["type"] in ["tp"]

		def io(p):
			return self.cpn[p]["type"] in ["ic", "io"]

		def rc(p):
			return self.cpn[p]["type"] in ["rsns", "dsc", "mux", "cmc"]
		
		def dnp(p):
			return self.cpn[p]["thru"] is None

		d = dict((k, {}) for k in "ser par pwr gnd g2g dns tp".split())
		# list series components and shunt
		rc_pin = [x for x in pins if rc(x) and not dnp(x)]
		for a in rc_pin:
			u, n, b = _u(a), _n(a), _b(a)
			if n and v(b):
				d["par"].update({b: self.cpn[b]["net"]})
			elif n:
				d["ser"].update({u: True})
		# list power(s), grounds(s) on all parts involved
		prt = [_u(x) for x in pins]
		for p, e in self.cpn.items():
			if _u(p) in prt and e["volt"]:
				if e["volt"] == "0":
					d["gnd"].update({e["net"]: True})
				else:
					d["pwr"].update({e["net"]: True})
		# list devices connecting 2 ground nets:
		pin0v = [x for x in pins if v(x) == "0"]
		for a in pin0v:
			u, n, b = _u(a), _n(a), _b(a)
			if n and b in pin0v:
				d["g2g"].update({u: True})

		d["io"] = dict((x, True) for x in pins if io(x) and not dnp(x))
		d["tp"] = dict((_u(x), True) for x in pins if tp(x) and not dnp(x))
		d["dns"] = dict((_u(x), True) for x in pins if dnp(x))

		return d

	def regex(self, inc=None, exc=None, prt=None, exact=True):
		def cat(x):
			return ",".join(x)

		def lst(x):
			return x.split(",")

		def _sort(x):
			return ",".join(sorted(x.split(",")))

		exclude_whole_xnet = True
		include_exact_nets = True
		d = {}
		if isinstance(inc, re.Pattern):
			s = [e for e in self.xnet if any(inc.search(x) for x in lst(e))]
			if exact:  # include_exact_nets
				for e in s:
					k = [x for x in lst(e) if inc.search(x)]
					v = [cat(self.nets[x]) for x in k]
					d.update({cat(k): cat(v)})
			else:  # not include_exact_nets
				d = {k: self.xnet[k] for k in s}
			if isinstance(exc, re.Pattern):  # xnets to be excluded
				if exact:  # not exclude_whole_xnet
					s = list(d.keys())
					d = {}
					for e in s:
						k = [x for x in lst(e) if not exc.search(x)]
						v = [cat(self.nets[x]) for x in k]
						d.update({cat(k): cat(v)})
				else:  # exclude_whole_xnet
					s = [e for e in d if not exc.search(e)]
					d = {k: d[k] for k in s}
			if isinstance(prt, re.Pattern):  # xnets must on refes
				d = {k: v for (k, v) in d.items() if prt.search(v)}
		return dict((_sort(k), _sort(v)) for (k, v) in d.items() if k)

	def _cpn(self, fpn):
		if isinstance(fpn, dict):  # initiate with given cpn
			self.cpn = fpn
		elif pstr(fpn).isfile:
			df = pd.read_csv(str(fpn), skiprows=4, dtype=str)
			for ln in df.itertuples():
				partno = re.findall(r"-(\d{3}-\d{5}-\d{2})", ln.FUNC_TYPE)
				self.gpn.update({ln.REFDES: partno[0] if partno else ""})
				pin = f"{ln.REFDES}.{ln.PIN_NUMBER}"
				self.cpn[pin] = {
						"name": ln.PIN_NAME,
						"net": ln.NET_NAME,
						"type": "",
						"thru": "",
						"volt": "",
				}
			# for i in range(len(df)):
			# 	pin = f'{df["REFDES"][i]}.{df["PIN_NUMBER"][i]}'
			# 	net = df["NET_NAME"][i]
			# 	self.cpn[pin] = {'name': df["PIN_NAME"][i], 'net': net, 'type': '', 'thru': '', 'volt': ''}
		for pin, d in self.cpn.items():
			if d["net"] in self.nets:
				self.nets[d["net"]].append(pin)
			else:
				self.nets[d["net"]] = [pin]
	
	def get_gnd_pins(self, gnd_rex):
		if not self.cpn:
			return {}
		rex = re.compile(gnd_rex, re.I)
		by_u = {}
		for pin, d in self.cpn.items():
			if rex.match(d['net']):
				u, num = pin.split('.')
				by_u.setdefault(u, []).append(num)
		return by_u
	
	def _dc(self, volts={}):
		# volts={'.*(vcc)*':3.3, '^vcc(\d)[vp](\d+).*':'\2.\3'}
		vnets = {}
		for pin, attr in self.cpn.items():
			net = attr["net"].lower()
			pat = [e for e in volts if re.match(e, net, re.I)]
			dcs = [volts[k] for k in pat] if pat else []
			notv = any(x.upper() == "NAN" for x in dcs) if pat else True
			if notv:
				attr["volt"] = ""  # always give self.cpn a 'volt' property
			else:
				dc = dcs[-1]  # pick last voltage
				attr["volt"] = re.sub(pat[-1].lower(), dc, net, count=0) if ("\\" in dc) else dc
				vnets[attr["net"]] = attr["volt"]
		self.vnet.update(vnets)

		return vnets

	def _thru(self):
		# specials = {'Q1.3':'Q1.4', 'Q1.2':'Q1.1}
		# cpn['C427.1'] = {'name':'1','net':'LD_PPVAR_VSYS_PWR_RFFE','type':'dsc', 'thru':'2'}
		# S-D, A-K, 1-2
		fets = dict(zip(list("sSdDaAkK12"), list("dDsSkKaA21")))
		numbers, names = {}, {}
		for pin, attr in self.cpn.items():
			for t, pat in PIN_TYPES_REGEX.items():
				if re.match(pat, pin, re.I):
					attr["type"] = t
					break
			self.cpn[pin] = attr
			prt, num = pin.split(".")
			if prt in numbers:
				numbers[prt].update({num: attr["name"]})
			else:
				numbers[prt] = {num: attr["name"]}
		for prt, d in numbers.items():
			names[prt] = dict((v, k) for (k, v) in d.items())

		for pin, attr in self.cpn.items():
			if attr["name"] and attr["type"] in ["dsc", "rsns"]:
				prt, num = pin.split(".")
				
				# change 4 term Reistor to rsense
				if prt[0] in 'rR' and len(numbers[prt]) ==4:
					attr["type"] = "rsns"
				
				x = attr["name"]  # (x,y) are pin_names of a thru pinpair
				# discetes with 2 terminals: swap the pin number( a,b)
				if 2 == len(numbers[prt]):
					a, b = numbers[prt].keys()
					attr["thru"] = a if (num == b) else b
				# 4 terminal rsns
				elif attr["type"] == "rsns" and num in ["1", "2"]:
					attr["thru"] = "2" if num == "1" else "1"
				# D/Qs
				elif (prt[0] in "dDqQ") and (x[0] in fets):
					# example dual fets: pin=1/2/3/4/5/6, name=s1/d1/g1/s2/d2/g2,--> pinpair=1-2,4-5
					y = fets[x[0]] + x[1:]
					if y in names[prt]:
						b = names[prt][y]
						attr["thru"] = b if b in self.cpn else ""
				# self.cpn[pin] = attr

	def _dns(self, dns):
		if isinstance(dns, str) and dns:
			refdes = re.split(r"\s*,\s*", re.sub(r'[\'"]', "", dns))
			dns = {}
			for rdes in refdes:
				if rdes in self.gpn:
					dns.update({rdes: "user"})
				else:
					z = re.findall(r"^[gG]?(\d{3}-\d{5}-\d{2})", rdes)
					if z:
						dns.update({u: no for (u, no) in self.gpn.items() if u == z[0]})
		for pin, attr in self.cpn.items():
			u = pin.split(".")[0]
			if u in dns and bool(dns[u]):
				attr["thru"] = None
		pass

	def _xpin(self, xpin):
		if isinstance(xpin, str):
			s = re.sub(r"\s*[=,]\s*", ",", xpin, count=0).split(",")
			s = s[0 : (2 * (len(s) // 2))]
			xpn = dict(zip(s[0::2], s[1::2])) if s else {}
		elif isinstance(xpin, dict):
			xpn = xpin  # {'RS2':'(1 2)(3 4)'}
		else:
			return

		rex_gpn = re.compile(r'^G?(\d{3}-\d{5}-\d{2})$', re.I)
		comps = set( e.split(".")[0] for e in self.cpn )
		for k, v in xpn.items():
			if(z:= rex_gpn.match(k) ):
				refdes = [ e for e in comps if self.gpn.get(e,'') == z.group(1)]
			else:
				refdes = [e for e in comps if re.match(k, e, re.I)]
			items = re.split(r"[\s\)\(]+", v)
			if len(items) % 2 == 0 and len(items) > 3:
				for u in refdes:
					pin = [f"{u.upper()}.{e}" for e in items[1:-1]]
					for p1, p2 in zip(pin[0::2], pin[1::2]):
						if all(e in self.cpn for e in [p1, p2]):
							self.cpn[p1]["thru"] = p2.split(".")[-1]
							self.cpn[p2]["thru"] = p1.split(".")[-1]

	def _mux(self, sw):
		pass

class spdlinks:

	def __init__(self, converter="", app="", raw="", spd="", amm="", precut="", stk=[]):
		# raw file, e.g. .tgz, .brd
		self.raw = pstr(raw)
		# app, e.g. pwoersi, clarity, for setting up workflow
		self.app = pstr(app)
		# traslator, e.g. powersi, spdlink
		self.converter = pstr(converter) if converter else self.app
		# .spd
		self.spd = pstr(spd) if spd else pstr(self.raw.file() + ".spd")
		self.amm = pstr(amm)
		self._precut = precut
		self._stk = stk

	@property
	def translated(self):
		if self.raw == self.spd:
			return True
		return self.spd.after(self.raw)

	def translate(self, options=None):
		src = self.raw
		if src.isfile is False:
			print(f"file {src} not found")
			return None
		if True:
			# if self.converter != self.app:
			ext = src.ext().lower()
			if ext == ".brd":
				src = self._brd2spd()
			elif ext == ".tgz":
				src = self._tgz2spd()
		src = spd_workflow(self.app, str(src))
		src = self._spd2spd(src)
		if options:
			src = self._process(options)
		return src

	def _brd2spd(self, wait=1800):  # 30 minutes to finishe transilation
		run_spdlinks = f"{self.converter} -brd {self.raw} {self.spd}"
		dos(run_spdlinks, cwd=self.raw.path(), timeout=wait)
		return self.spd

	def _tgz2spd(self, wait=120):
		run_spdlinks = f"{self.converter} -zip {self.raw} {self.spd}"
		dos(run_spdlinks, cwd=self.raw.path(), timeout=wait)
		return self.spd

	def _spd2spd(self, datbase, wait=180):
		database = pstr(datbase)
		cwd = database.path()
		if bool(self._stk):
			cmx, csv = read_stackup(self._stk)
			pstr(f"{cwd}/material.cmx").write(cmx)
			pstr(f"{cwd}/stackup.csv").write(csv)
		ltcl = [r"sigrity::import stackup $stackup {!}"]
		swap = r"  sigrity::update layer layer_name {NEW} {OLD} {!}"
		if self.app.isexe:
			f_tcl = pstr(f"{database.file()}.tcl")
			tcstr = template.tcl_brd_spd.lstrip()
			tcstr = re.sub(r"DATABASE", datbase, tcstr, count=1, flags=0)
			tcstr = re.sub(r"LIBRARY", str(self.amm), tcstr, count=1, flags=0)
			if bool(self._precut):
				tcstr = re.sub(r"PRECUT", self._precut, tcstr, count=1, flags=0)
			if os.path.exists(f"{cwd}/material.cmx"):
				print(f"  applying meterial file {cwd}/material.cmx")
				tcstr = re.sub(r"MATERIAL", f"{cwd}/material.cmx", tcstr, count=1, flags=0)
			if os.path.exists(f"{cwd}/stackup.csv"):
				print(f"  applying stackup file {cwd}/stackup.csv")
				lmap = self._stack_up(f"{cwd}/stackup.csv")
				ltcl += [swap.replace("OLD", k).replace("NEW", v) for (k, v) in lmap.items()]
				tcstr = re.sub(r"STACKUP", f"{cwd}/stackup.csv", tcstr, count=1, flags=0)
				tcstr = tcstr.replace(ltcl[0], "\n".join(ltcl))
			f_tcl.write(tcstr)

			command = f'{self.app} -tcl "{f_tcl}"'
			dos(command, cwd=cwd, timeout=wait)
		else:
			print(f"splinks._spd2spd(): error find executable {self.app}")
		f_spd = self.spd
		if not f_spd.isfile:
			print(f"Error translating {datbase} to spd file\n")
			return ""
		else:  # add abs path to s-parameter models if any
			rex_smodel = re.compile(
					r"^\.model\s+\S+\s+S\s+tstonefile=[\'\"]?(.+\\)?(\S+\.s\d+p)[\'\"]?",
					re.I,
			)
			has_amm_path = lambda p: self.amm.path() in p.replace("\\", "/")
			f_spd_1 = pstr(f"{self.spd},1")
			if f_spd.copyto(f_spd_1):
				text = f_spd.read().splitlines()
				slines = {i: z.groups() for (i, ln) in enumerate(text) if (z := rex_smodel.match(ln))}
				I = {k: m for k, (p, m) in slines.items() if p is None}
				J = {m: k for k, (p, m) in slines.items() if p is not None and has_amm_path(p)}
				for i, m in I.items():
					if m in J:
						text[i] = text[J[m]]
						for j in filter(lambda k: text[k][0] in "sS", range(i + 1, len(text))):
							mname = text[i].split()[1]
							text[j] = re.sub(r"mname=(.+)", f"mname={mname}", text[j], count=0)
							break
						# for j in filter(lambda k: text[k].startswith('.SUBCKT'), range(J[m]-2, len(text))):
						# 	for n in range(j, len(text)):
						# 		text[n] = '*'+ text[n]
						# 		if re.search('.ENDS$',text[n].strip()):
						# 			break
						# 	break

				f_spd.write("\n".join(text))
				if f_spd.after(f_spd_1):
					f_spd_1.remove()
				for i, m in I.items():
					pstr(f_spd.path() + "/" + m).remove()
			print(f"  spd file created: {self.spd}")
		return self.spd
	
	def _process(self, options):
		f = self.spd
		if options.get('rm_diff_channel', False) and f.isfile:
			f.write(re.sub(r'\nDiff_Channel', '\n*Diff_Channel',f.read()))
			return f

	def _stack_up(self, csv="stackup.csv"):
		"""inserts/updates dilectrical layers an return new-old layer mapping"""
		lines = self.spd.readlines()
		stk_new = pd.read_csv(str(csv))

		def modify_layers(lines, df_stk):
			stk_layers = df_stk["Layer #"].notna().sum()
			layer_map = {}
			# signal layers
			re_conductor = re.compile(r"^(Signal|Plane)", re.I)
			ln_conductors = [ln for ln in lines if re_conductor.search(ln)]
			if len(ln_conductors) != stk_layers:
				return lines, layer_map
			# patch/shape lines
			re_patch = re.compile(r"^(Patch)", re.I)
			# stackup lines
			i, skt_lines = 0, []
			for index, row in df_stk.iterrows():
				if pd.isna(row["Layer #"]):  # dielectric
					skt_lines.append(f"{row['Layer Name']} Thickness = 1mm Material = {row['Material']}")
				else:  # conductor
					skt_lines.append(ln_conductors[i])
					layer_map.update({ln_conductors[i].split()[0]: row["Layer Name"]})
					i += 1
			skt_lines += [ln for ln in lines if re_patch.search(ln)]
			swap_map = {k: v for (k, v) in layer_map.items() if k != v}
			return skt_lines, swap_map

		re_start = re.compile(r"^(Medium|Signal|Plane)", re.I)
		re_end = re.compile(r"^Node\d+", re.I)

		start_idx, end_idx = None, None
		for i, line in enumerate(lines):
			line = line.strip()
			if re_start.search(line) and start_idx is None:
				start_idx = i
			elif re_end.search(line) and start_idx is not None:
				end_idx = i
				break

		if start_idx is not None and end_idx is not None:
			stk_lines = []
			for line in lines[start_idx:end_idx]:
				ln = re.sub(r"^\*.*", "", line.strip())
				if not ln:
					continue
				if ln.startswith("+"):
					stk_lines[-1] += re.sub(r"\++", "", ln)
				else:
					stk_lines.append(ln)
			new_lines, layer_map = modify_layers(stk_lines, stk_new)
			lines[start_idx:end_idx] = [s + "\n" for s in new_lines]
			self.spd.writelines(lines)
			return layer_map
		else:
			print("Pattern not found or incorrect order of patterns")
			return {}

class component:
	
	default_component_value = {re.compile(r'^XW\d+$', re.I): 1e-6}

	def __init__(self, cmp=None, nondns="", dns="", dcr=""):
		cols = "refdes partno value layer dns dcr type model"
		self.data = pd.DataFrame(columns=cols.split())
		self.dns = {}
		self.value = {}
		self.gpn = {}
		self.dcr = {}
		self.model = {}

		if isinstance(cmp, dict):
			self._from_dict(cmp)
		elif pstr(cmp).isfile:
			self._from_cmpfile(str(cmp))
		#add default part value is empty
		no_value = lambda u: self.value.get(u,'').strip()==''
		for refdes, default_value in component.default_component_value.items():
			parts = [ u for u in self.value if refdes.match(u) ]
			for part in filter(no_value, parts):
				self.value[part] = default_value

		if nondns:
			self.dns_tag(nondns, "")
		if dns:
			self.dns_tag(dns, "user")
		if dcr:
			self.dns_dcr(dcr)

	def _from_cmpfile(self, file):
		with open(file) as f:
			for line in f:
				refdes, user, eda = line.split("!")
				props = "" if user == "nil" else user[1:-1]
				props += "" if eda == "nil" else eda[1:-1]

				z = re.search(r'\(BOM_IGNORE "([^\"]+)"\)', props, re.I)
				self.dns[refdes] = z.group(1) if z else ""

				z = re.search(r'\(VALUE "([^\"]+)"\)', props, re.I)
				self.value[refdes] = engnum(z.group(1)).value if z else ""
				z = re.search(r'\(PART_NUMBER "([^\"]+)"\)', props, re.I)
				self.gpn[refdes] = z.group(1) if z else ""
				# z= re.search(r'\(dcr "(.+?)"\).*',props,re.I)
				z = re.search(r'\(dcr "([^\"]+)"\)', props, re.I)
				self.dcr[refdes] = engnum(z.group(1)).value if z else ""

	def _from_dict(self, data):
		self.dns = data.get("dns", {})
		self.value = data.get("compval", {})
		self.gpn = data.get("partno", {})
		self.dcr = data.get("dcr", {})
		self.model = data.get("model", {})

	def dns_tag(self, entry, tag=""):
		# strip quotes:
		entry = re.sub(r"[\'\"]", "", entry.strip(), count=0)
		entries = re.split(r"\s*,\s*", entry)
		gpn_map = {
				e: (rex_part_numbers.sub(r"^G?\2", e, count=0) if rex_part_numbers.match(e) else None)
				for e in entries
		}
		for e, n in gpn_map.items():
			parts = [] if n else [u for u in self.dns if re.match(rf"^{e}$", u, re.I)]
			parts += [u for (u, v) in self.gpn.items() if re.search(n, v, re.I)] if n else []
			for u in set(parts):
				self.dns[u] = tag

	def dns_dcr(self, limits, amm=None):
		""" dsn if dcr is larg, for power nets """
		# Set default dcr values for keys starting with 'R' if they don't have dns/dcr values
		for u, v in self.value.items():
			if not (self.dns.get(u) or self.dcr.get(u)) and u[0].upper() == 'R':
				self.dcr[u] = v
		
		# update dcr value with those from amm library
		if isinstance(amm, ammx):
			r_amm, l_amm = amm.resistor(), amm.inductor()
			x_amm, s_amm = amm.subcircuit(), amm.touchstone()
			for u, gpn in ((u, self.gpn.get(u)) for u in self.dcr if self.gpn.get(u)):
				if gpn in r_amm:
					self.dcr[u] = engnum(r_amm[gpn][2]).value
				elif gpn in l_amm:
					self.dcr[u] = engnum(l_amm[gpn][0]).value
				if gpn in x_amm:
					self.model[u] = x_amm[gpn][1]
				elif gpn in s_amm:
					self.model[u] = s_amm[gpn]

		# chekc against r/l limits
		rl_limits = {k: v for (k,v) in eval(limits).items() if k in {"r_max", "l_max"} }
		for k, v in rl_limits.items():
			num = engnum(v).value
			isu = lambda e: e[0].lower() == k[0]
			for u in filter(isu, self.dns):
				if self.dns[u] in ["user"]:
					continue
				if self.dcr.get(u):
					self.dns[u] = "dcr" if self.dcr[u] > num else ""
	
	def to_jsonf(self, file):
		""" dump to json file"""
		import math

		def is_num(e):
			return isinstance(e, (int, float)) and not (isinstance(e, float) and math.isnan(e))

		data = {
				'gpn': self.gpn,
				'dns': {k: v for k, v in self.dns.items() if v},
				'dcr': {k: v for k, v in self.dcr.items() if is_num(v)},
				'value': {k: v for k, v in self.value.items() if is_num(v)},
				'model': {k: v for k, v in self.model.items() if pstr(v).isfile},
		}

		pstr(file).dump(data)
		
class allegro:

	def __init__(self, brd="", editor="allegro.exe"):
		self.brd = pstr(brd)
		self.fpn = self.brd.ext(".fpn")
		self.cmp = self.brd.ext(".cmp")
		self.app = str(editor)
		self._nets = {}
		self._fpns = {}

		if self.fpn.after(self.brd) and self.cmp.after(self.brd):
			return
		if self.brd.isfile:
			self.extract(self.app)
		if not(self.fpn.isfile and self.cmp.isfile):
			print(f"?? error extracting allegro database. missing .fpn or .cmp file")

	def _read(self, fpn_file):
		if pstr(fpn_file).isfile:
			df = pd.read_csv(str(fpn_file), skiprows=4, dtype=str)
			for row in df.itertuples(index=False):
				pin = f"{row.REFDES}.{row.PIN_NUMBER}"
				net = row.NET_NAME
				self._fpn[pin] = {
						"name": row.PIN_NAME,
						"net": net,
						"type": "",
						"thru": "",
						"volt": "",
				}
				self._nets.setdefault(net, []).append(pin)
				# if net in self.netlist:
				# 	self._nets[net].append(pin)
				# else:
				# 	self._nets[net] = [pin]

	def extract(self, exe):
		cwd = pstr(self.brd.path())
		brd = self.brd.file()
		
		il = cwd + "extract.il"
		scr = cwd+ "extract.scr"
		
		il.write(template.il_fpn_cmp.lstrip())
		scr.write(f'open "{brd}"\nskill load("extract.il")\nexit\n')
		
		if il.recently() and scr.recently():
			cmd = f'{exe} -nograph -p "{cwd}" -s extract.scr'
			dos(cmd, cwd=cwd, timeout=600)
			il.remove()
			scr.remove()

	def netlist(self):
		if not self._nets:
			self._read(self.fpn)
		return self._nets

	def funcpin(self):
		if not self._fpns:
			self._read(self.fpn)
		return self._fpns

	def unrouted(self, topo):
		nc = {"rpt": "", "net": {}, "ind": []}
		ncp = self.brd.ext('.ncp')

		if not ncp.after(self.brd):
			il = f'axlReportGenerate("Unconnected Pins Report" nil "{ncp}")'
			skill(self.app, str(self.brd)).run(il)
			if not ncp.recently(60):
				print(f"allegro.unrouted(): error creating unrouted pins report {ncp}")
				return {}
			
		s = ncp.read()
		if not s:
			return {}

		t = topo[topo["simulate"] == "TRUE"]
		ncp_txt = pstr(re.sub(r"brd\.ncp$", "ncp.txt", str(ncp), count=0))
		txt = {}

		for row in t.itertuples():
			nets = re.split(r"[\s,]+", row.nets)
			for net in nets:
				if( cut:=re.search(rf"\n{net}\n((From:[^\n]+\n)+)\n", s) ):
					txt.update({net: cut.group(1)})
					nc["ind"] += [row.Index]
					nc["net"].update({",".join(nets): True})
					# fr, to = {}, {}
					# for line in cut.group(1).split("\n"):
					# 	pins = re.search(rf"From: (\S+).+ To: (\S+).+", line)
					# 	if pins and 2 == len(pins.groups()):
					# 		fr.update({pins.group(1): True})
					# 		to.update({pins.group(2): True})
		if txt:
			flat_txt  = [e for pair in txt.items() for e in pair]
			ncp_txt.write("\n".join(flat_txt ))
			nc["rpt"] = str(ncp_txt)
			return nc

		return {}

	def set(self, k, v):
		exec(f"self.{k} = {v}")

class sigrity:

	gnd_rex =  r".*gnd(\d)?$"   # regex pattern for GND nets

	def __init__(self, app="", amm="", spd="", precut="", stk=[]):
		self._exe = str(app)  # powersi.exe
		self._amm = str(amm)  # path to .amm libary
		self.spd_file = pstr(spd)
		self._precut = precut

		# clean/one-liner text. do not write to file or sigrity error reading long lines
		self._text = []
		self._nodes = {}
		self._data = {}   # {netlist,comppin,dns,partno,compval}
		self._shorts = {} # {refdes : ( gnd_nets, pins) } for gnds shorting element

		self.ROUND_UP_SCALE = 1e4

		if bool(self._exe):
			exe, sw = re.search(r"(.+exe)(\s-.+)?", self._exe).groups()
			if os.path.isfile(exe) is False:
				print(f"Sigrity executable {exe} not found \n")
			else:
				self.spd_file = pstr(spd_workflow(self._exe, str(self.spd_file)))

	def set(self, k, v):
		r"""example self.set('gnd_rex', r'.*gnd(\d)?$')"""
		exec(f"self.{k} = {v}")

	@property
	def text(self):
		if not self._text:
			s = self.spd_file.read()
			s = re.sub(r"\n\+\s+", " ", s, count=0)
			s = [re.sub(r"\s+", " ", x.strip(), count=0) for x in s.splitlines()]
			# skip comment line
			self._text = [x for x in s if x and x.startswith("*") is False]
		return self._text
	
	@property
	def data(self):
		"""read spd file directly for netlist, comppin, partno, dsn, compval"""
		if self._data:
			return self._data

		nodes_lyr = {}
		for s in self.text:
			if s.startswith("Node"):
				z = re.search(r"(^node[\S]+).*layer\s*=\s*(\S+)", s, re.I)
				if z and len(z.groups()) == 2:
					nodes_lyr[z.group(1)] = z.group(2)

		comppin, netlist, partno, compval, complyr, dns = {}, {}, {}, {}, {}, {}
		for s in self.text:
			if s.startswith(".Component"):
				z = re.search(r".Component (\S+).*StartLayer = (\S+)", s)
				if not z:
					pass  # try complyr found by node
					# print(f'? mount layer not found for {s[1:]}')
				else:
					refdes, layer = z.groups()
					complyr.update({refdes: layer})

		inBlock = False
		for i, s in enumerate(self.text[:-1]):
			if s.startswith(".EndC"):
				inBlock = False
			elif s.startswith(".Connect"):
				inBlock = True
				refdes, gpn = s.split()[1:3]
				partno.update({refdes: gpn})
				if self.text[i + 1].startswith(".EndC"):
					dns.update({refdes: gpn})
				else:
					dns.update({refdes: ""})
			elif inBlock:
				if not "::" in s:
					pin, node, pin_name = re.search(r"(.+)\s+(.+)!!(.+)", s).groups()
					net = ""
				else:
					pin, node, pin_name, net = re.search(r"(.+)\s+(.+)!!(.+)::(.+)", s).groups()
				if refdes not in complyr:
					node = re.findall("(node.+)", s, re.I)
					if len(node):
						complyr[refdes] = nodes_lyr[node[0]]
				pin2 = f"{refdes}.{pin}"
				netlist.update({net: netlist[net] + [pin2] if net in netlist else [pin2]})
				comppin.update({pin2: {"name": pin_name, "net": net, "type": "", "thru": ""}})

		partval = {}
		inBlock = False
		for s in self.text:
			if s.startswith(".EndPartialCkt"):
				inBlock = False
			elif s.startswith(".PartialCkt"):
				inBlock = True
				if not "ExtNode" in s:
					gpn, nodes = s.split()[1], ""
				else:
					gpn, _, nodes = re.search(r"\.PartialCkt\s+(\S+)\s+ExtNode\s+=(\s+)?(.*)", s).groups()
				partval.update({gpn: ""})
			elif inBlock:
				words = s.split()
				if len(nodes.split()) == 2 and len(words) == 4:
					partval.update({gpn: words[-1]})
		for refdes, gpn in partno.items():
			if gpn in partval:
				compval.update({refdes: partval[gpn]})
		cpn = functionpin(comppin, dns).cpn

		self._data = {
				"netlist": netlist,
				"comppin": cpn,
				"complyr": complyr,
				"dns": dns,
				"partno": partno,
				"compval": compval,
		}

		return self._data

	@property
	def nodes(self):
		if not (self._nodes and self._nodes["all"]):
			gnd_net = re.compile(r":" + sigrity.gnd_rex, re.I)
			nd_circuit = re.compile(r"(\S+).*Package\.(Node.+\S)")

			all_nodes, ref_nodes = {}, {}
			# print('  all_nodes: reading nodes info')
			for line in filter(lambda x: x.startswith("Node"), self.text):
				node = line.split()[0]
				d = dict(re.findall(r"\s*(\S+)\s*=\s*(\S+)\s*", line))
				d["X"] = scaled_int(d["X"], self.ROUND_UP_SCALE)
				d["Y"] = scaled_int(d["Y"], self.ROUND_UP_SCALE)
				# d['Net'] = nd.split(':')[-1]
				all_nodes.update({node: d})
				if gnd_net.search(node):
					lyr = d["Layer"]
					ref_nodes.setdefault(lyr, {}).update({node: True})
			pin_nodes, gnd_pins, in_circuit = {}, {}, False
			# print('  circuit_nodes: reading circuit nodes')
			for line in self.text:
				if line.startswith(".EndC"):
					in_circuit = False
				elif line.startswith(".Connect"):
					in_circuit = True
					u = re.split(r"\s+", line)[1]
					gnd_pins.update({u: []})
				elif in_circuit:
					pin, node = nd_circuit.match(line).groups()
					pin_nodes.update({f"{u}.{pin}": node})
					if gnd_net.search(node):
						gnd_pins[u].append(node)

			self._nodes = {
					"all": all_nodes,
					"pin": pin_nodes,
					"ref": ref_nodes,
					"gnd": gnd_pins,
			}
		return self._nodes

	@property
	def shorts(self):
		if self._shorts:
			return self._shorts
		isgnd = re.compile(sigrity.gnd_rex,re.I)
		for u in self.data['compval']:
			pins = [ p for p in self.data['comppin'] if u==p.split('.')[0] ]
			nets = [self.data['comppin'][p].get('net',None) for p in pins] if len(pins)==2 else []
			if all(nets) and all(isgnd.match(n) for n in nets):
				self._shorts[u] = (nets, pins)
		return self._shorts

	def write(self, spd=""):
		max_width = 200
		fspd = pstr(spd) if spd else self.spd_file
		lines = []
		for line in self.text:
			if len(line) < max_width:
				lines.append(line)
			else:
				chunks = textwrap.wrap(line, max_width)
				lines.append(chunks[0])
				chunks = ["+ " + x for x in chunks[1:]]
				lines.extend(chunks)
		fspd.write("\n".join(lines))

	def apply_subckt(self, src_ckt={}, saveas=''):
		""" apply subckt model and save as other spd if directed"""
		#src_ckt ={ gpn: (path/value, Ture(ac)/(dc) )}
		SET_DCR = True
		circuits = {gpn: '' for gpn in src_ckt}
		for gpn in circuits:
			path_val, ac = src_ckt[gpn]
			if ac is False and SET_DCR:
				nodes = '1 2'
				if path_val.lower().startswith('inf'): 
					path_val = '1e12'
				circuits[gpn] = (
					f".PartialCkt {gpn}  ExtNode = {nodes}\n"
					f"R1 {nodes} {path_val}\n"
					f".EndPartialCkt\n"
				)
				continue
			#snp:
			#.PartialCkt G180-04942-00 ExtNode = 1 2 3 4
			#.model NFG0QHB542HS2 S tstonefile='G:\My Drive\lab\models\cmc\NFP0QHB542HS2.s4p
			#S1 1 2 4 3 0 mname=NFG0QHB542HS2
			#.EndPartialCkt

			if( N:= pstr(path_val).issnp() ):
				nodes = ' '.join(str(i) for i in range(1, N + 1))	
				name = pstr(path_val).name()
				circuits[gpn] = (
					f".PartialCkt {gpn}  ExtNode = {nodes}\n"
					f".model {name} S tstonefile='{path_val}'\n"
					f"S1 {nodes} 0 mname={name}\n"
					f".EndPartialCkt\n"
				)
				continue
			# ckt:
			# .PartialCkt G150-11185-00 ExtNode = 1 2
			# xcall 1 2 sub_GRM033R61A105ME44
			# .SUBCKT sub_GRM033R61A105ME44 port1 port2
			# ....
			# .ENDS
			# .EndPartialCkt

			if pstr(path_val).isfile:
				lines= [s.strip() for s in pstr(path_val).readlines() if not s.strip().startswith('*')]
				lines = '\n'.join([re.sub(r'^\+\s*',' ',s) for s in lines]).splitlines()
				if lines and lines[0].lower().startswith('.subckt'):
					words = lines[0].split()
					nodes = ' '.join(str(i) for i in range(1, len(words)-1))	
					circuits[gpn] = (
						f".PartialCkt {gpn}  ExtNode = {nodes}\n"
						f"xcall {nodes} {words[1]}\n"
						f"{'\n'.join(lines)}\n"
						f".EndPartialCkt\n"
					)

		temp_path = f'{self.spd_file}.tmp'
		with open(f'{self.spd_file}', 'r') as infile, open(temp_path, 'w') as tmpfile:
			in_block = False
			skip_plus = False #'+ mcp lines after endpartialckt'
			for line in infile:
				if not in_block:
					if( z:= re.match(r'\.PartialCkt\s+(\S+)', line.strip()) ):
						gpn = z.group(1)
						if circuits.get(gpn,''):
							in_block = True
							tmpfile.writelines(circuits[gpn].splitlines(keepends=True))
							continue  
					if skip_plus and line.strip().startswith('+'):
						continue
					skip_plus = False
					tmpfile.write(line)
				else:
					if line.strip().startswith('.EndPartialCkt'):
						in_block = False
						skip_plus = True

		out_path = str(saveas) if(saveas and pstr(pstr(saveas).path()).isdir) else str(self.spd_file)
		shutil.move(temp_path, out_path)					
			
	def sig_pin_node(self, pin):
		return self.nodes["pin"].get(pin,"")
	
	def gnd_pin_nodes(self, part):
		u = part.split('.')[0]
		return self.nodes["gnd"].get(u,[])

	def reference_node(self, pin, nearest=True, onpin=False, cut_margin=5):
		part, number = pin.split(".")
		gnd_pins = self.nodes["gnd"]
		all_nodes = self.nodes["all"]
		ref_nodes = self.nodes["ref"]
		grounds = gnd_pins[part] if (part in gnd_pins) else []
		signal = self.sig_pin_node(pin)
		if not signal:
			return "", "", -1
		gnd_pin, gnd_dist, ref_node, ref_dist = [], [], [], []
		# measure ref node distance
		if True:  # always search for nearst ref node
			lyr, x, y = [all_nodes[signal][k] for k in "Layer X Y".split()]
			src_node, ref_node = (signal, list(ref_nodes[lyr].keys()))
			dist = list(
					map(
							lambda z: (all_nodes[z]["X"] - x) ** 2 + (all_nodes[z]["Y"] - y) ** 2,
							ref_node,
					)
			)
			i, ref_dist = zip(*sorted(enumerate(dist), key=lambda x: x[1]))
			ref_node = [ref_node[j] for j in i]
			mm_ref_gap = ref_dist[0] ** (0.5) / self.ROUND_UP_SCALE
		# measure gnd pin distance
		tgt_node, tgt_dist = ref_node, list(ref_dist)
		if grounds:
			lyr, x, y = [all_nodes[signal][k] for k in "Layer X Y".split()]
			src_node, gnd_pin = (signal, grounds)
			dist = list(
					map(
							lambda z: (all_nodes[z]["X"] - x) ** 2 + (all_nodes[z]["Y"] - y) ** 2,
							gnd_pin,
					)
			)
			i, gnd_dist = zip(*sorted(enumerate(dist), key=lambda x: x[1]))
			gnd_pin = [gnd_pin[j] for j in i]
			mm_gnd_gap = gnd_dist[0] ** (0.5) / self.ROUND_UP_SCALE
			# prefer refrence node on pin if within cut margin
			if mm_gnd_gap < cut_margin:
				if gnd_dist[0] < (2 * ref_dist[0]) and gnd_pin[0] != ref_node[0]:
					tgt_node.insert(0, gnd_pin[0])
					tgt_dist.insert(0, gnd_dist[0])
			# else other nodes if they are closer
			elif mm_ref_gap < mm_gnd_gap:
				onpin = False

		min_dist = gnd_dist[0] if onpin and grounds else ref_dist[0]
		if type(nearest) == bool and nearest:
			tgt_node = gnd_pin[0] if onpin and len(gnd_pin) else tgt_node[0]
			min_dist = gnd_dist[0] if onpin and len(gnd_pin) else tgt_dist[0]
		elif type(nearest) == int:
			n = min(max(1, nearest), len(tgt_node))
			tgt_node = gnd_pin[:n] if onpin and len(gnd_pin) else tgt_node[:n]
			min_dist = min(gnd_dist[:n]) if onpin and len(gnd_pin) else min(tgt_dist[:n])

		d = int(0.5 + min_dist ** (0.5) / self.ROUND_UP_SCALE)
		return src_node.strip(), tgt_node.strip(), d

	def short_grounds(self, parts):
		text = [ln.strip() for ln in self.spd_file.readlines()]
		gpns = set(filter(None, [self.data['partno'].get(p, None) for p in parts]))
		rex_start = re.compile(rf"^\.PartialCkt\s+({'|'.join(gpns)})")
		rex_stop = re.compile(rf"^\.EndPartialCkt")
		start = [(i,line) for i, line in enumerate(text) if rex_start.match(line)]
		stop = [i for i, line in enumerate(text) if rex_stop.match(line)]
		inserted = False
		for i,line in reversed(start):
			m, n = i+1, min((j for j in stop if j > i), default=i)
			if m>n:
				print(f"sigrity::short_grounds(), error paring partialckt around line\n\t --{line}")
				continue
			node = line.rpartition("=")[-1].strip()
			if 2 != len(re.split(r'\s+',node)):
				print(f"sigrity::short_grounds(), not 2 terminal in line {line}")
			else:
				text[m:n] = [f"R {node} 0"] + [f"* {e}" for e in text[m:n]]
				inserted = True
		if inserted:
			self.spd_file.write('\n'.join(text))
			print(f'sigrity::short_grounds(): spd file overitten with shorts {str(gpns)}')

	def bnp2snp(self, net):
		sim_folder = self.spd_file.path().replace("/SimBrd", "/SimFiles")
		nets = net if type(net) == list else [net]
		queque = []
		DC_S1P_OPEN_SHORT_IS_FINE = True
		for net in nets:
			cwd = pstr(f"{sim_folder}/{net}")
			# power si: _dcfittted s*p
			dcfitted = cwd.glob("*dcfitted*.s*p")
			if len(dcfitted):
				continue
			# clarity: single zone, _FIT.s*p
			dcfitted = cwd.glob("*_FIT.s*p")
			if len(dcfitted):
				src = max(dcfitted, key=lambda f: os.path.getmtime(f))
				ext = os.path.splitext(src)[-1]
				tgt = re.sub(rf"_FIT{ext}$", f"_dcfitted{ext}", src, count=0)
				open(tgt, "wb").write(open(src, "rb").read())
				continue

			# clarity: multizune: _DCFittedManually.bnp
			dcfitted = cwd.glob("*dcfitted*.bnp")
			if not len(dcfitted):
				continue
			src = max(dcfitted, key=lambda f: os.path.getmtime(f))
			snp = re.sub(r"_DCFittedManually\.bnp$", "", src, count=0)
			dc = [x.replace("\\", "/") for x in glob.glob(f"{snp}_DC.s*p")]
			if len(dc) != 1:
				continue
			ext = os.path.splitext(dc[0])[-1]

			tgt = re.sub(r"_DCFittedManually\.bnp$", f"_dcfitted{ext}", src)
			ftcl = pstr(os.path.dirname(src) + "/powersi_bnp2snp.tcl")
			stcl = template.tcl_bnp_snp.lstrip()
			str = re.sub(
					"{SRC}",
					"{" + src + "}",
					stcl.replace("{TGT}", "{" + tgt + "}"),
					count=0,
			)
			ftcl.write(str)
			queque.append(ftcl)

		if bool(queque):
			q = [f'-tcl "{x}"' for x in queque]
			lics = 3
			s = scheduler(self._exe, lics, q)
			s.run()
			timeout = time.time() + len(q) * (600 / lics)
			while not s.complete():
				time.sleep(1)
				if time.time() > timeout:
					s.kill("all")
					break

	def run(self, licenses=4, overwrite=True):
		brd_folder = self.spd_file.path()
		sim_folder = brd_folder.replace("/SimBrd", "/SimFiles")
		res_folder = brd_folder.replace("/SimBrd", "/Result")

		# find jobs to run
		parts = brd_folder.replace("\\", "/").split("/")
		wrk = folders("/".join(parts[:-2]), parts[-2])

		simulated = wrk.simulated(santize=(2 if overwrite else 1))
		netfolder = [x for x in os.listdir(sim_folder) if os.path.isdir(f"{sim_folder}/{x}")]
		self.bnp2snp(netfolder)
		wrk.s1p_fitted()
		wrk.copy_snp(newcopy=True)
		# run the job
		q = [f'-tcl "{x}"' for x in simulated if not simulated[x]]
		s = scheduler(self._exe, licenses, q)
		logme.info(f"  total {len(q)} jobs in queque,  running {min(licenses,len(q))} in parallel")
		s.run()
		timeout = time.time() + len(q) * (21600 / licenses)
		complete = []
		while not s.complete():
			time.sleep(1)
			files = glob.glob(f"{sim_folder}/*.done")
			for f in [re.sub(r"\.done$", "", x, count=0) for x in files]:
				netdone = os.path.basename(f)
				if not (netdone in complete):
					complete.append(netdone)
					self.bnp2snp(netdone)
					wrk.s1p_fitted()
					wrk.copy_snp(netdone)
					logme.info(f"  simulated: file[{len(complete)}/{len(q)}]: {netdone}.spd")
			if time.time() > timeout:
				s.kill("all")
				break

		wrk.copy_snp()
		return len(glob.glob(f"{sim_folder}/*.done")) - len(glob.glob(f"{res_folder}/*dcfitted*.s*p"))

	def pdc(self, exe="", tcl=""):
		sim_folder = self.spd_file.path().replace("/SimBrd", "/SimFiles")

		if not exe:
			exe = self._exe
		if not os.path.exists(exe.split()[0]):
			print(f"executable not find -{exe}")
			return "", None
		if not os.path.exists(tcl):
			print(f"tcl file not find- {tcl}")
			return "", None
		if True:
			q = [f'-tcl "{tcl}"']
			licenses = 1
			s = scheduler(exe, licenses, q)
			s.run()
			timeout = time.time() + len(q) * (3600 / licenses)
			while not s.complete():
				time.sleep(1)
				if time.time() > timeout:
					s.kill("all")
					break
		resi = os.path.splitext(os.path.basename(tcl))[0]
		resdirs = [x for x in os.listdir(f"{sim_folder}/{resi}") if os.path.isdir(f"{sim_folder}/{resi}/{x}")]
		resdir = max(resdirs, key=lambda f: os.path.getmtime(f"{sim_folder}/{resi}/{f}"))
		rescsv = f"{sim_folder}/{resi}/{resdir}/CSVFolder/Resis.csv"

		portcsv = pstr(f"{sim_folder}/{resi}").latest("*.resi.csv")
		rtable = csv2dict(portcsv, add_key="Value")
		rvalue = csv2dict(rescsv)
		if bool(rtable) and bool(rvalue):
			for k, v in zip(rvalue["Name"], rvalue["Value(Ohm)"]):
				if not k in rtable["Resistance Name"]:
					continue
				rtable["Value"][rtable["Resistance Name"].index(k)] = v if v else "Z"
		return portcsv, rtable

class touchstone:

	def __init__(self, tsfile):
		self.tsfile = str(tsfile)
		self._data = {}

	@property
	def data(self):
		if bool(self._data):
			return self._data
		# read snp file to self.data[]
		text = []
		with open(self.tsfile, "r") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				if re.search(r"^[#!]", line):
					text.append(line)
				elif re.search(r"^\d", line):
					break
		tstr = "\n".join(text)
		port_lines = []
		# sigrity tools
		if any(x in tstr for x in ["Layout Workbench"]):
			for line in text:
				z = re.match(r"^! Port\s+(\d+)\s*=\s*(\S+)-(\S+)\s+(\S+)$", line)
				if z:
					port_lines.append(z.groups())
		# ansys tools
		elif any(x in tstr for x in ["ElectronicsDesktop", "HFSS"]):
			for line in text:
				z = re.match(r"^! Port\[(\d+)\]\s*=\s*(\S+)\.(\S+)\.(\S+)$", line)
				if z:
					port_lines.append(z.groups())

		if not port_lines:
			print(f"?? {self.tsfile} missing header from sigrity/ansys")
			return {}
		netlist, comppin, partno, complyr, compval, dns = {}, {}, {}, {}, {}, {}
		for q in port_lines:
			refdes = q[1].upper()
			for suffix in ["_SBUMP", "_SBALL"]:
				if refdes.endswith(suffix):
					refdes = refdes.replace(suffix, "")
			net, pin = q[-1], f"{refdes}.{q[2]}"
			netlist.update({net: netlist[net] + [pin] if net in netlist else [pin]})
			comppin.update({pin: {"name": q[2], "net": net, "type": "", "thru": ""}})
			partno.update({refdes: "undef"})
			complyr.update({refdes: "unknown"})
			compval.update({refdes: ""})
			dns.update({refdes: ""})
		self._data = {
				"netlist": netlist,
				"comppin": comppin,
				"complyr": complyr,
				"dns": dns,
				"partno": partno,
				"compval": compval,
		}
		return self._data

class concepthdl:

	def __init__(self, **kwargs):
		self.cfg = {"cdslib": "C:/Users/baoshu/ee/cds/cdslib/misc"}

		for key, value in kwargs.items():
			self.cfg.update({key: value})

	def getchip(self, sym="laguna_v30p5"):
		cdslib = self.cfg.get("cdslib", input("path of the csdlib is:"))
		if not bool(cdslib):
			print("cdslib not known")
			return {}

		string = ""
		if os.path.isdir(f"{cdslib}/{sym}"):
			chips = f"{cdslib}/{sym}/chips/chips.prt"
			if os.path.exists(chips):
				with open(chips, "r") as f:
					string = f.read()
		string = string.strip()
		if not string:
			print(f"empty file {chips}")
			return {}

		pins = {}
		z = re.findall(r"\'(.+)\'(<\d+>)?:\s*\n\s*PIN_NUMBER=\'\((.+)\)\';\s*\n", string)
		for name, rep, pages in z:
			for i, num in enumerate(pages.split(",")):
				if not "0" in num:
					pins.update({name + rep: [num, i]})
					break
		return pins  # {pin_name, [ball_number,symbol_page]}


if __name__ == "__main__":
	# lga = concepthdl().getchip()
	# brd_file = "C:/Users/baoshu/Downloads/proj/P25/deepspace/mb/hs_20231025_1608/SimBrd/G651-13008-01.brd"
	# brd = allegro(brd_file)
	# netlist = brd.netlist()
	obj = ammx(r'G:\My Drive\lab\models\Google_SIPI_Library.ammx')
	r= obj.resistor()
	print("cool")
