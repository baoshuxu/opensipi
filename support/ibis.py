import re
import os
import string
import time


def trailing(i, x):
	return (i, x[x.index(i) + 1 :])


class pkg:

	def __init__(self, dot_pkg_file):
		pass


class ebd:

	def __init__(self, dot_ebd_file):
		pass


class ibis:

	buffer_io_types = dict(
		[
				(e, 'i') for e in ['input', 'terminator']
		] + [
				(e, 'i/o') for e in ['i/o', 'input_output', 'i/o_open_drain', 'i/o_open_sink', 'i/o_open_source']
		] + [
				(e, 'o') for e in ['output', 'open_drain', 'open_sink', 'open_source', 'tristate', '3-state']
		])
	
	def __init__(self, file, read="short"):
		text, brackets = ibs_text(file)
		self.file = file
		self.header = ibs_header(text)
		self.component = ibs_component(text, brackets)
		self.selector = ibs_selector(text, brackets)
		if read in ["short"]:
			self.models = ibs_model_short(text, brackets)
		else:
			self.models = ibs_model(text, brackets)
			self.submodel = ibs_submodel(text, brackets)
			self.circiut = ibs_external_circuit(text, brackets)
			self.testdata = ibs_test_data(text, brackets)
			self.testload = ibs_test_load(text, brackets)
			self.pkgmodel = ibs_package_model(text, brackets)

		# add io type to component pins
		for comp_name, comp in self.component.items():
			for pin_name, pin_dict in comp["Pin"].items():
				pin_dict.update({"type": ""})
				model_or_selctor = self.pin(pin_name, comp_name)["model_name"]
				if isinstance(model_or_selctor, dict):
					mselector = list(model_or_selctor.values())[0]
					model = list(mselector.keys())[0]  # take the frist model in selctor for type detection
				else:
					model = model_or_selctor
				if model and not (model.upper() in ["NC", "GND", "POWER"]):
					if model not in self.models:
						print(f"pin {pin_name} has model {model} missing in {self.file}")
					else:
						spec = self.spec(model)
					if not spec:
						continue
					btype = spec["type"].lower()
					pin_dict.update({"type": ibis.buffer_io_types.get(btype)})

	def pin(self, pinName, comp=""):
		c = list(self.component.keys())[0] if not bool(comp) else comp
		if not pinName in self.component[c]["Pin"]:
			return None
			# raise Warning(f'{pinName} not found in {self.file}')
		p = self.component[c]["Pin"][pinName]
		m = p["model_name"]
		if isinstance(m, str) and m in self.selector:
			p["model_name"] = {m: self.selector[m]}
		return p

	def comp(self):
		return list(self.component.keys())

	def spec(self, bufferName):
		type, vdd, vih, vil, vmeas, ven = ("", "", "", "", "", "")

		if not bufferName in self.models:
			return {}
		buffer = self.models[bufferName]
		if "Model Spec" in buffer:
			text = buffer["Model Spec"].strip().lower().splitlines()
			for str in text:
				if "vinh" in str:
					vih = str.split()[1]
				elif "vinl" in str:
					vil = str.split()[1]
				elif "vmeas" in str:
					vmeas = str.split()[1]
		if "Model" in buffer:
			text = buffer["Model"].strip().lower().splitlines()
			for str in text:
				str = re.sub(r"\s*=\s*", " ", str, count=0)
				if (not vih) and ("vinh" in str):
					vih = str.split()[1]
				elif (not vil) and ("vinl" in str):
					vil = str.split()[1]
				elif (not vmeas) and ("vmeas" in str):
					vmeas = str.split()[1]
				elif "enable" in str:
					ven = "1" if ("active-high" in str) else "0"
				elif "model_type" in str:
					type = str.split()[1].lower()
		if "Receiver Thresholds" in buffer:
			pass
		# print(f'{bufferName} from {os.path.basename(self.file)}')
		if "Voltage Range" in buffer.keys():
			vdd = buffer["Voltage Range"].strip().lower().split()[0]
		elif "Pullup Reference" in buffer.keys():
			vdd = buffer["Pullup Reference"].strip().lower().split()[0]

		return {
				"io": ibis.buffer_io_types.get(type),
				"vdd": vdd,
				"vih": vih,
				"vil": vil,
				"vmeas": vmeas,
				"ven": ven,
				"type": type,
		}


# ibis specification version 6.0 key words
def kwv6(key=None):
	top_level_key = "[Component],[Model Selector],[Model],[Submodel],[External Circuit],[Test Data],[Test Load],[Define Package Model],[End]".split(
			","
	)
	model_key = []
	submodel_key = []
	if key is None:
		return top_level_key
	if key == "[Model]":
		return model_key
	if key == "[Submodel]":
		return submodel_key


def ibs_key(line, key):
	if not key in line:
		raise Warning(f'{key} not in "{line}"')
	return line.replace(key, "").strip()


def ibs_piece(text, enclose):
	pieces = {}
	start = enclose[0]
	end = [enclose[1]] if isinstance(enclose[1], str) else enclose[1]
	m, b = (0, False)
	for i, s in enumerate(text):
		if s.startswith(start):
			if b and i > m:
				kw = ibs_key(text[m], start)
				pieces[kw] = {"text": text[m:i]}
			m, b = (i, True)
		elif any(s.startswith(x) for x in end):
			if b:
				kw = ibs_key(text[m], start)
				pieces[kw] = {"text": text[m:i]}
			m, b = (i, False)
	return pieces


def ibs_brackets(text, enclose, bracket=None):
	pieces = {}
	start = enclose[0]
	end = [enclose[1]] if isinstance(enclose[1], str) else enclose[1]
	m, b = (0, False)
	if bracket is None:
		bracket = range(len(text))
	for i in bracket:
		s = text[i]
		if s.startswith(start):
			if b and i > m:
				kw = ibs_key(text[m], start)
				pieces[kw] = {"text": text[m:i]}
			m, b = (i, True)
		elif any(s.startswith(x) for x in end):
			if b:
				kw = ibs_key(text[m], start)
				pieces[kw] = {"text": text[m:i]}
			m, b = (i, False)
	return pieces


def ibs_text(file):
	tic = time.time()

	bracket = {}
	comment_char = r"\|"  # Default comment character

	with open(file, "r") as f:
		lines = [line.strip() for line in f]

	# Faster `startswith` check
	bracket = {i: True for i, line in enumerate(lines) if line.startswith("[")}

	# Find the comment character efficiently without regular expressions
	for i in bracket:
		if lines[i].startswith("[Comment Char]"):
			# Faster way to extract comment_char using split()
			parts = lines[i].split()
			if len(parts) >= 3:
				comment_char = re.escape(parts[2][0])  # Extract the first character after `]`
			break

	comment_pattern = re.compile(rf"{comment_char}.*")
	space_pattern = re.compile(r"\s+")
	lines = [space_pattern.sub(" ", comment_pattern.sub("", x)) for x in lines]

	# Optimized loop to capitalize text inside brackets
	for i in bracket:
		line = lines[i]

		# Use pre-compiled patterns and chain replacements in a single operation
		line = re.sub(
				r"\[(.*?)\]",
				lambda x: f"[{string.capwords(x.group(1).strip())}]",
				line,
				count=0,
		)
		line = re.sub(r"\[(Power|Gnd|Ibis*?) ", lambda x: f"[{x.group(1).upper()} ", line, count=0)
		line = re.sub(r" (Mosfet*?\])", lambda x: f" {x.group(1).upper()}", line, count=0)
		line = re.sub(
				r"([\[\s](Emi|Oem*?)[\s\]])",
				lambda x: f"{x.group(1).upper()}",
				line,
				count=0,
		)
		line = re.sub(r"(\[Isso (Pu|Pd)\])", lambda x: f"{x.group(1).upper()}", line, count=0)

		lines[i] = line

	# Filter out empty lines
	text = [x for x in lines if x]

	# Update bracket indices for the filtered lines
	bracket = {i: True for i, line in enumerate(text) if line.startswith("[")}

	tictoc = time.time() - tic
	if tictoc > 0:
		print(f"ibis.ibs_text(): takes {tictoc:.2f} seconds parsing {os.path.basename(file)}")

	return text, bracket


def ibs_header(text):
	for i, s in enumerate(text):
		if s.startswith("[Component]"):
			return text[:i]
	return ""


def ibs_component(text, bracket=None):

	component = ibs_brackets(text, trailing("[Component]", kwv6()), bracket)

	for c, d in component.items():
		txt = d.pop("text", None)
		s = re.split(r"[\[\]]", "\n".join(txt[1:]))
		t = dict(zip(s[1::2], s[2::2]))
		for k, v in t.items():
			v = v.strip().splitlines()
			component[c].update({k: v})
		s = component[c].pop("Pin", None)
		h = f"pin {s[0].lower()}".split()
		p = {}
		for x in s[1:]:
			w = x.split()
			while len(w) < len(h):
				w.append("")
			p[w[0]] = dict(zip(h[1:], w[1:]))
		component[c].update({"Pin": p})

	return component


def ibs_selector(text, bracket=None):
	selector = ibs_brackets(text, ("[Model Selector]", "["), bracket)
	for k, d in selector.items():
		txt = d.pop("text", None)
		for str in txt[1:]:
			w = str.split()
			d.update({w[0]: " ".join(w[1:])})
		selector[k] = d
	return selector


def ibs_model(text, bracket=None):
	models = ibs_brackets(text, trailing("[Model]", kwv6()), bracket)
	inc = {
			"ext": (r"\[External Model]", r"\[End External Model\]"),
			"ami": (r"\[Algorithmic Model\]", r"\[End Algorithmic Model\]"),
			"emi": (r"\[Begin EMI Model\]", r"\[End EMI Model\]"),
	}
	vt = {
			"e1": (r"\[Rising Waveform\]", r"\[Composite Current\]", r"\["),
			"e2": (r"\[Falling Waveform\]", r"\[Composite Current\]", r"\["),
			"e3": (r"\[Rising Waveform\]", r"\[Composite Current\]", r"\["),
			"e4": (r"\[Falling Waveform\]", r"\[Composite Current\]", r"\["),
	}
	for c, d in models.items():
		txt = d.pop("text", None)
		str = "\n".join(txt[0:])
		# external model/ algrightmic model, emi model
		for m, (start, end) in inc.items():
			z = re.search(rf"({start}(.+){end}\n)", str)
			if (not z) or (len(z.groups()) != 2):
				continue
			models[c].update({m: z.groups()[1]})
			str = str.replace(z.groups()[0], "")

		# multiple vt curves
		models[c].update({"vt": []})
		str = str + "["
		for m, (start, option, end) in vt.items():
			z = re.search(rf"({start}.*?({option}.*?)?){end}", str, re.DOTALL)
			if not z:
				continue
			models[c]["vt"].append(z.groups()[0])
			str = str.replace(f"{z.groups()[0]}", "")
		str = str[:-1]

		# other keywords using
		s = re.split(r"[\[\]]", str)
		t = dict(zip(s[1::2], s[2::2]))
		for k, v in t.items():
			models[c].update({k: v.strip()})

	return models


def ibs_model_short(text, bracket=None):
	models = ibs_brackets(text, trailing("[Model]", kwv6()), bracket)
	for c, d in models.items():
		txt = d.pop("text", None)
		a = ibs_piece(txt, (r"[Model]", r"["))
		# a = ibs_brackets(txt,(r'[Model]',r'['), None)
		models[c].update({"Model": f"{c}\n" + "\n".join(a[c]["text"])})
	return models


def ibs_submodel(text, bracket=None):
	submodel = ibs_brackets(text, trailing("[Submodel]", kwv6()), bracket)
	return submodel


def ibs_external_circuit(text, bracket=None):
	circuit = ibs_brackets(text, trailing("[External Circuit]", kwv6()), bracket)
	# todo
	return circuit


def ibs_test_data(text, bracket=None):
	testdata = ibs_brackets(text, trailing("[Test Data]", kwv6()), bracket)
	return testdata


def ibs_test_load(text, bracket=None):
	testload = ibs_brackets(text, trailing("[Test Load]", kwv6()), bracket)
	return testload


def ibs_package_model(text, bracket=None):
	package_model = ibs_brackets(text, ("[Define Package Model]", "[End Package Model]"), bracket)
	return package_model


def plot_vt(buffer):
	pass


def plot_iv(buffer):
	pass


def calc_zin(buffer):
	pass


def calc_zout(buffer):
	pass


def sym2pin(sym, ibs):
	"""translate component info (allegro info comp) to ibis [pin] section"""
	with open(sym, "r") as f:
		text = f.read().splitlines()
	text = [x.strip() for x in text]
	text = [x for x in text if bool(x)]
	sections = []
	for i, line in enumerate(text):
		if line.startswith("Pin IO Information:"):
			sections = text[i + 2 :]
			break
	content = []
	content.append("[Pin]  signal_name      model_name           R_pin     L_pin      C_pin")
	ii, jj, kk = 8, 16, 24
	for i, line in enumerate(sections):
		w = line.split()
		pin = ""
		if len(w) < 2:
			pin += w[0].ljust(ii) + w[0].ljust(jj) + "NC"
		elif len(w) < 3:
			w.append("")
			if w[1].upper() in "POWER":
				w[2] = w[1]
			elif w[1].upper() in ["GROUND", "GND"]:
				w[2] = "GND"
			elif w[1].upper() in ["NC", "UNSPEC"]:
				w[2] = "NC"
			pin += w[0].ljust(ii) + w[0].ljust(jj) + w[2]
		elif len(w) < 4:
			if w[1] in ["POWER"]:
				w[1], w[2] = w[0], "POWER"
			elif w[1] in ["GROUND", "GND"]:
				w[1], w[2] = w[0], "GND"
			else:
				w[1], w[2] = w[2], "TBD"
			pin += w[0].ljust(ii) + w[0].ljust(jj) + w[2]
		else:
			pin += w[0].ljust(ii) + w[2].ljust(jj) + w[2]

		content.append(pin)

	with open(ibs, "w") as f:
		f.write("\n".join(content))


def formal(file):
	"""try make ibis files  more formal with basic syntax correction"""
	# make filename lower case
	fname = os.path.basename(file)
	fpath = os.path.dirname(file)
	file1 = fpath.replace("\\", "/") + "/" + fname.lower()
	if fname.lower() != fname:
		os.rename(file, file1)
	with open(file, "r") as f:
		text = f.read().splitlines()

	# make file name in ibis file same
	for i, s in enumerate(text):
		if s.startswith("[Component]"):
			text[i] = re.sub(r"(\]\s+)(\S*)", lambda x: x.group(1) + x.group(2).lower(), s, count=0)
		if re.match(r"^\[Voltage range\]", s, re.I):
			text[i] = re.sub(r"^\[Voltage range\]", "[Voltage Range]", s, count=0, flags=re.I)
		if re.match(r"^\[file name\]", s, re.I):
			text[i] = "[File Name] " + fname.lower()
	with open(file1, "w") as f:
		f.write("\n".join(text))


def ibis_test():
	file = "C:/proj/parts/pixel/lga_v10_a0_gpio.ibs"
	file = r"C:\proj\parts\modem\u24001.ibs"
	tic = time.time()
	ibs = ibis(file, "short")
	toc = time.time()
	print(f"{toc-tic}")
	if False:
		ibs = ibis(file)
		pin = ibs.pin("A4")
		# pin = ibs.pin('1')
		model = pin["model_name"]
		if isinstance(model, str):
			io = model
		elif isinstance(model, dict):
			selector, models = list(model.items())[0]
			io = list(models.keys())[0]
		spec = ibs.spec(io)
		plot_iv(io)
		plot_vt(io)
		zin = calc_zin(io)
		zout = calc_zout(io)
if __name__ == "__main__":
	ibis_test()
