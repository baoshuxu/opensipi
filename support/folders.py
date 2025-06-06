import re
import os
import sys
import json
import time
import glob
import winreg
import shlex
import shutil
import requests
import readline
import subprocess
import numpy as np
from collections.abc import Iterable


import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s", stream=sys.stdout)
# logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.WARNING)
# logging.getLogger('googleapiclient.http').setLevel(logging.WARNING)


def set_long_path():
	if os.name == "nt":
		try:
			key = winreg.OpenKey(
					winreg.HKEY_LOCAL_MACHINE,
					r"SYSTEM\CurrentControlSet\Control\FileSystem",
					0,
					winreg.KEY_SET_VALUE,
			)
			winreg.SetValueEx(key, "LongPathsEnabled", 1, winreg.REG_DWORD, 1)
			key.Close()
		except PermissionError as e:
			print("Please relaunch with administrator permissions. Exiting...")
			sys.exit(1)

def assert_os_variable(name, value):
	val = os.environ.get(name)
	if val == value:
		return True
	elif val is not None:
		print(f"Environment variable {name} is not set to {value}")
		return False
	else:
		print(f"Environment variable {name} does not exist.")
		return False
	
def debug_enabled():
	try:
		if sys.gettrace() is not None:
			return True
	except AttributeError:
		pass
	try:
		if sys.monitoring.get_tool(sys.monitoring.DEBUGGER_ID) is not None:
			return True
	except AttributeError:
		pass
	return False


def day_hour():
	t = time.localtime()
	day = f"{t.tm_year}{str(t.tm_mon).zfill(2)}{str(t.tm_mday).zfill(2)}"
	hour = f"{str(t.tm_hour).zfill(2)}{str(t.tm_min).zfill(2)}"
	return f"{day}_{hour}"


class logme:
	_configured = False  # class level variable

	def __init__(self, logfile=""):
		if not logme._configured:
			file = str(logfile)
			if file and pstr(os.path.dirname(file)).isdir:
				fh = logging.FileHandler(file)
				fh.setLevel(logging.WARNING)
				fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
				logging.getLogger().addHandler(fh)
			logme._configured = True

	@staticmethod
	def info(msg):
		logging.info(msg)

	@staticmethod
	def warning(msg):
		logging.warning(msg)


""" multiprocessing conflicts with fancy logging """
# class logme:
# 	_instance = None  # Singleton instance
# 	_logger = None    # Logger instance

# 	def __new__(cls, appname=None,logfile=None):
# 		if cls._instance is None:
# 			cls._instance = super(logme, cls).__new__(cls)
# 		return cls._instance

# 	def __init__(self, appname=None, logfile=None):
# 		# Logger setup only on the first instantiation
# 		if logme._logger is None:
# 			# Create the logger
# 			app_name = appname if appname else __name__
# 			logme._logger = logging.getLogger(app_name)
# 			logme._logger.setLevel(logging.DEBUG)
# 			formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 			# Create console handler for logging to the console
# 			ch = logging.StreamHandler()
# 			ch.setLevel(logging.DEBUG)
# 			ch.setFormatter(formatter)
# 			logme._logger.addHandler(ch)

# 			if logfile:
# 				# If a logfile is provided, create a file handler
# 				fh = logging.FileHandler(logfile)
# 				fh.setLevel(logging.DEBUG)
# 				fh.setFormatter(formatter)
# 				logme._logger.addHandler(fh)

# 	@staticmethod
# 	def info(message):
# 		if logme._logger is None:
# 			raise Exception("Logger not initialized! Call logme() first.")
# 		logme._logger.info(message)

# 	@staticmethod
# 	def error(message):
# 		if logme._logger is None:
# 			raise Exception("Logger not initialized! Call logme() first.")
# 		logme._logger.error(message)

# 	@staticmethod
# 	def debug(message):
# 		if logme._logger is None:
# 			raise Exception("Logger not initialized! Call logme() first.")
# 		logme._logger.debug(message)

# 	@staticmethod
# 	def warning(message):
# 		if logme._logger is None:
# 			raise Exception("Logger not initialized! Call logme() first.")
# 		logme._logger.warning(message)


class pstr(str):
	"""path-string, encapsulate some os operations"""

	def __new__(cls, f):
		f = str(f).replace("\\", "/")  # Normalize the path separator
		return super().__new__(cls, f)

	def __eq__(self, f):
		return super().__eq__(str(f).replace("\\", "/"))

	def __add__(self, suffix):
		return pstr(re.sub(r"/+", "/", f"{self}/{str(suffix)}"))

	def sub(self, rep, pat="/SimBrd/?[^/]*$", flags=0):
		return pstr(re.sub(pat, rep, str(self), flags=flags))

	@property
	def missing(self):
		return not os.path.exists(str(self))

	@property
	def isfile(self):
		return os.path.isfile(str(self))

	@property
	def isexe(self):
		arg0 = shlex.split(str(self))[0]
		return os.path.isfile(arg0) and arg0.endswith(".exe")

	@property
	def isdir(self):
		return os.path.isdir(str(self))

	@property
	def isurl(self):
		valid_url = False
		if re.search(r"^http(s)?:", str(self), re.I):
			try:
				r = requests.head(self, allow_redirects=True, timeout=5, verify=True)
				valid_url = r.status_code == 200
			except requests.exceptions.RequestException:
				valid_url = False
		return valid_url

	def issnp(self):
		if not os.path.isfile(str(self)):
			return False
		if (z:=re.search(r"\.s(\d+)p$", str(self), re.I)):
			return z.group(1)
		return False
	
	def which(self):
		return shutil.which(str(self))

	def recently(self, dt=60):
		return (time.time() - os.path.getmtime(str(self))) < dt if self.isfile else False

	def size(self):
		return os.path.getsize(str(self)) if self.isfile else None

	def split(self):
		path,base = os.path.split(str(self))
		name,ext = os.path.splitext(base)
		return path, name, ext

	def path(self):
		return self.split()[0]

	def name(self):
		return self.split()[1]

	def ext(self, e=None):
		return pstr(self.rpartition(".")[0] + str(e)) if e else self.split()[-1]

	def base(self):
		return "".join(self.split()[1:])

	def file(self):
		return "/".join(self.split()[:2])

	def after(self, f):
		# True if: 1) equal, 2) after, 3) no other
		# False if : 1) no this 2) not after
		if not self.isfile:
			return False
		elif self.lower() == f.lower():
			return True
		elif os.path.exists(str(f)):
			return os.path.getmtime(str(self)) > os.path.getmtime(str(f))
		return True

	def read(self, b=""):
		if not self.isfile:
			print(f"{self} is not a file, nothing read")
			return ""
		try:
			with open(str(self), "r" + b) as f:
				return f.read()
		except Exception as e:
			print(f"error reading file {self}: {e}")
		return ""

	def readlines(self, b=""):
		if not self.isfile:
			print(f"{self} is not a file, nothing read")
			return ""
		try:
			with open(str(self), "r" + b) as f:
				return f.readlines()
		except Exception as e:
			print(f"error reading file {self}: {e}")
		return ""

	def write(self, s, b=""):
		p = self.path()
		if not os.path.isdir(p):
			print(f"path for {self} does not exist")
			return None
		try:
			with open(str(self), "w" + b) as f:
				return f.write(s) if b else f.write(str(s))
		except Exception as e:
			print(f"error writing file {self}: {e}")
		return None

	def writelines(self, s):
		p = self.path()
		if not os.path.isdir(p):
			print(f"path for {self} does not exist")
			return None
		try:
			with open(str(self), "w") as f:
				return f.writelines(s)
		except Exception as e:
			print(f"error writing file {self}: {e}")
		return None

	def dump(self, d={}):
		''' json dump'''
		if isinstance(d, dict) and len(d):
			try:
				with open(str(self), "w") as f:
					json.dump(d, f, indent=2)
					return True
			except Exception as e:
				print(f"error writing file {self}: {e}")
		return None
	
	def load(self):
		''' json load '''
		if self.isfile:
			with open(str(self)) as fjson:
				try:
					return json.load(fjson)
				except json.JSONDecodeError:
					print("__init__(): json file error")
					return None
		return None
	
	def remove(self):
		if self.isfile:
			try:
				os.remove(self)
				return True
			except Exception as e:
				return f"error {e}"
		elif self.isdir:
			try:
				shutil.rmtree(self, ignore_errors=True)
				return True
			except Exception as e:
				return f"error {e}"
		else:
			return True

	def mkdir(self):
		try:
			os.makedirs(self, exist_ok=True)
			return self
		except Exception as e:
			return f"error {e}"

	def clear(self):
		if self.isdir:
			for f in self.listdir():
				(self+f).remove()

	def copyto(self, tgt):
		if self.isfile:
			try:
				shutil.copy(str(self), str(tgt))
				return True
			except Exception as e:
				print(f"error {e}")
				return False
		elif self.isdir and os.path.isdir(str(tgt)):
			try:
				shutil.copytree(str(self), str(tgt))
				return True
			except Exception as e:
				print(f"error {e}")
				return False
		else:
			return False

	def glob(self, wildcard="*.*"):
		files = glob.glob(f"{self}/{wildcard}", recursive=False)
		return [f.replace("\\", "/") for f in files]

	def latest(self, wildcard="*.*"):
		files= self.glob(wildcard)
		return max(files, key=lambda f: os.path.getmtime(f)) if files else None
	
	def earlist(self, wildcard="*.*"):	
		return min(self.glob(wildcard), key=lambda f: os.path.getmtime(f))

	def lstypes(self, types="*.spd;*.brd"):
		files = []
		for ext in re.split(r"[\s,;]+", types):
			files.extend(self.glob(ext))
		return files

	def listdir(self):
		return os.listdir(self)

	def tree(self, wildcard="*.*"):
		files = glob.glob(f"{self}/{wildcard}", recursive=True)
		return [f.replace("\\", "/") for f in files]

	def copyfrom(self, src):
		if os.path.isfile(str(src)):
			try:
				shutil.copy(str(src), str(self))
				return True
			except Exception as e:
				print(f"error {e}")
				return False
		elif os.path.isdir(str(src)) and self.isdir:
			try:
				shutil.copytree(str(src), str(self))
				return True
			except Exception as e:
				print(f"error {e}")
				return False
		else:
			return False

	def readline(self):
		with open(str(self), "r") as f:
			for line in f:
				yield line.strip()


class tsfile:

	def __init__(self, file):
		self.file = pstr(file)
		self.head = []
		self.n = 0
		self.f = None
		self.s = None
		self.origin = None

		ext = self.file.ext()
		self.n = int(z[0]) if (z := re.findall(r"s(\d+)p", ext)) else 0
		self.origin = self._origin()

	def _origin(self):
		line= self._header()[0]
		structures = ['.spd','.adet']
		source = [e for e in structures if line.endswith(e)]
		return source[0] if source else None

	
	def _header(self):
		if len(self.head) < 1 and self.file.isfile:
			hash_line = []
			for line in self.file.readline():
				if line.startswith("!"):
					self.head.append(line)
				elif line.startswith("#"):
					hash_line.append(line)
				elif re.search(r"^\d+", line):
					break
			self.head += hash_line[-1:]
		return self.head

	def _data(self):
		if self.n and self.f is None:
			row_size = self.n * self.n * 2 + 1
			fmt = self.format()
			lines = self.file.read()
			lines = re.sub(r"\n\s*\!.*?\n", "\n", lines, count=0, flags=re.DOTALL)
			lines = re.sub(r".*#.*?\n", "", lines)
			numbers = re.split(r"[\s\n]+", lines)
			rows = len(numbers) // (row_size)
			if len(numbers) == rows * row_size:
				print("tsfile.data(): erorr processing s-parameter data")
				return None
			f = np.zeros(rows)
			s = np.zeros((rows, self.n, self.n))
			for i in range(rows):
				row = numbers[row_size * i : row_size * (i + 1)]
				f[i] = float(row[0])
				v1 = np.array(row[1::2], dtype=float)
				v2 = np.array(row[2::2], dtype=float)
				if "db" in fmt["cord"]:
					s[i] = np.reshape((10 ** (-v1 / 20)) * (np.exp(1j * v2 / 180)), self.n, self.n)
				elif "ma" in fmt["cord"]:
					s[i] = np.reshape(v1 * np.exp(1j * v2 / 180), (self.n, self.n))
				else:
					s[i] = np.reshape(v1 + 1j * v2, (self.n, self.n))
			self.f, self.s = f, s
		return self.f, self.s

	def format(self):
		d = {"hz": 1, "syz": "s", "cord": "ri", "z0": "50"}
		if self._header():
			z = re.sub(r"^#\s+", "", self.head[-1]).split()
			scale = {"h": 1, "k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12}
			unit = z[0][0].lower()
			factor = scale[unit] if unit in scale else 1
			d = {"hz": factor, "syz": z[1].lower(), "cord": z[2].lower(), "z0": z[-1]}
		return d

	def ports(self):
		names = {}
		#! Port 1 = CON7-26 RMAIN_SCL
		#! Port[10] = T900_PCIE_LN1_TXN_SPHY3_16_T1
		rex_port = re.compile(r"^! Port[\[\s]+(\d+)[\]\s]*=\s*(.+)$", re.I)
		for line in self._header():
			if( z := rex_port.search(line) ):
				num, other = z.groups()
				names[num] = re.split(r'\s+', other.replace('-','_'))
		if names:
			return names
		
		#! Port6_J1_sball_RF1::FR2_IF01_C
		rex_port = re.compile(r"^! ([^:]+)::([^:]+)$")
		count = 1
		for line in self._header():
			if z := rex_port.search(line):
				other, net = z.groups()
				if ( y := re.search(r'^port(\d+)[-_]+(.+)', other, re.I)):
					num, name = y.groups()
				else:
					num, name = str(count), other
					count += 1
				names[num] = [name, net]
		return names
	
	def pins(self):
		if self.origin == '.spd':
			rex = re.compile(r'_sball|_sbump',re.I)
			return {rex.sub('',v[0]):k for k,v in self.ports().items()}
		return {v[0]:k for k,v in self.ports().items()}
	
	def freq(self):
		return self.f or self._data()[0]

	def sparameter(self):
		return self.s or self._data()[1]


class license:
	# lmutil lmstat -a -c %CDS_LIC_FIE%
	def __init__(self, lic="CDS_LIC_FILE"):
		self.lic = lic
		self.features = {
			"powersi": "PowerSI_II", 
			"clarity": "Clarity_3DSolverG.*",
			"powerdc": "PowerDC",
		}

	def free(self, tool):
		"""check available license, lmutil.exe and lmstat.exe should reside in ./support folder"""

		# tool can be a nickname or feature name
		if tool in self.features:
			feature = self.features[tool]
		elif any(re.match(x, tool, re.I) for x in self.features.values()):
			feature = tool
		else:
			print(f"{tool} not find, please choose one from {str(self.features)}")
			return 0

		lic_file = os.environ.get(self.lic)
		if not lic_file:
			print(f"{self.lic} environment variable is not set")
			return 0
		
		cwd = os.path.dirname(os.path.abspath(__file__))
		if not all(pstr(f"{cwd}/{exe}.exe").isfile for exe in ["lmutil", "lmstat"]):
			print(f"Cannot find lmutil/lmstat under {cwd}")
			return 0

		cmd = f"lmutil lmstat -a -c {lic_file}"
		result = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

		if result.returncode != 0:
			print(f"Error occurred querying license: {result.stderr}")
			return 0
		matches = re.findall(
			rf"^Users of {feature}:.*Total of (\d+).*issued;.*Total of (\d+).*in use\)",
			result.stdout, re.M | re.I
		)

		if matches:
			return sum(int(total) - int(used) for total, used in matches)

		print(f"No license info for {tool} found in %{self.lic_env}%")
		return 0

	def get(self, n_lics, prompt, toolstr):
		"""Determine usable license count based on availability."""
		n_lics = int(n_lics)
		match = re.search(r"\s+-PS(\w+)", toolstr)
		if not match:
				return None

		feature = match.group(1)
		available = self.free(feature)

		if available < 1:
			print(f"{prompt}: no license available for {feature}")
			return 0
		if available < n_lics:
			print(f"{prompt}: reduced to {available} parallel runs due to license limits for {feature}")
			return available

		return n_lics


cds_lic = license("CDS_LIC_FILE")


def wildcard_to_regex(patterns):
	regex = []
	for pat in re.split(r"[\s,;]+", patterns):
		p = re.escape(pat)
		p = p.replace(r"\*", ".*")
		p = p.replace(r"\?", ".")
		p = rf"{p.rpartition('.')[0]}\d+p" if p.endswith('.s.*p') else p
		regex.append(f"^{p}$")
	return "|".join(regex)


def dos(command, cwd="c:/tmp", timeout=300, fexit=None):
	"""run command till timeout or exist earlier if fexit() fullfilled"""
	proc = subprocess.Popen(command, cwd=str(cwd))
	for _ in range(timeout):
		time.sleep(1)
		if fexit and fexit():
			proc.terminate()
			proc.wait()
			return True
		if proc.poll() == 0:
			return proc.returncode == 0
	proc.kill()
	proc.wait()
	return None


class scheduler:

	def __init__(self, executable="", run_size=4, tcl_queque=[], seperate_dir=True):
		self.executable = executable
		self.running = [None] * run_size
		self.timeout = [None] * run_size
		self.queque = {t: 1 for t in tcl_queque}
		self.cwd = self.where(seperate_dir)
		self.timer = 21600  # default timout 6 hours for one extraction
		self.interval = 30  # wait 30s to start next job, or license overload
		print("jobs in queque: \n" + "\n".join(tcl_queque))

		self.appointed = time.time()

	def where(self, seperate_dir=True):
		"""each net has a individual folder"""
		folders = {t: os.getcwd() for t in self.queque}
		if seperate_dir:
			for t in folders:
				dir, tcl = os.path.split(t.split('"')[1])
				net = re.sub(r"\.tcl$", "", tcl)
				cwd = re.sub(r"/Scripts$", f"/SimFiles/{net}", dir)
				os.makedirs(cwd, exist_ok=True)
				folders[t] = cwd
		return folders

	def set_timer(self, seconds=3600):
		"""set timer in seconds"""
		self.timer = seconds
		pass

	def add(self, jobs=[]):
		"""add more jobs to the queque"""
		self.queue.update({job: 1 for job in jobs})
		self.run()
		pass

	def run(self):
		"""run multiple jobs"""
		for i, _ in filter(lambda exe: not exe[1], enumerate(self.running)):
			for j, _ in filter(lambda job: job[1], self.queque.items()):
				self.running[i], self.timeout[i] = self.start(j, self.cwd[j])
				self.queque[j] = 0
				break

	def start(self, argument, cwd):
		"""open process for just one job, delayed start"""

		time.sleep(max(0, self.appointed - time.time()))
		now = time.time()
		self.appointed = now + self.interval
		command = f"{self.executable} {argument}"
		proc = subprocess.Popen(command, cwd=cwd)

		return proc, now + self.timer

	def complete(self):
		"""poll subprocess to see if all job finised (return 1)"""
		self.poll()
		if any(self.queque.values()):
			self.run()
			return 0
		elif any(self.running):
			return 0
		return 1

	def poll(self):
		"""find information of jobs"""
		for i, exe in enumerate(self.running):
			exitcode = exe.poll() if exe else None
			if exitcode is not None:
				self.running[i] = None

	def kill(self, kind="timeout"):
		"""kill process, kind in ['all','timout']"""
		if kind == "all":
			for job in self.queque:
				self.queque[job] = 0
				for exe in filter(None, self.running):
					exe.kill()
		elif kind == "timeout":
			for i, exe in filter(lambda e: e[1], enumerate(self.running)):
				exitcode = exe.poll()
				if exitcode != 0 and time.time() > self.timeout[i]:
					exe.kill()
					print(f"job kill after {self.timeout[i]} seconds")
		else:
			pass


class folders:
	latest_snp_in_result_folder = True
	
	def __init__(self, parent="", child="", tag="run", ee=None):
		self.parent = pstr(str(parent).rstrip("/"))
		self.path = self.parent + (str(child) or f"{tag}_{day_hour()}")
		self.subdir = [
				"Logs", "Report", "Result", "Scripts",
				"SimBrd", "SimFiles", "Spice"
		]
		for name in self.subdir:
			setattr(self, name, self.path + name)

		self.Boards = self.parent + "Boards"
		self.ee = ee or {}

	@classmethod
	def setup( cls, row, prj, tag='pp'):
		''' setup up folder by a row in the board database table. return new simpath if necessary'''
		row_path = pstr(row.simpath.strip())
		
		if row_path.endswith("SimBrd"):
			root = pstr(row_path.path())
			parent = root.path()
			child = root.base()
			src_brd = pstr(f"{parent}/Boards/{row.file}")
		else:
			parent = row_path.sub("", r"/Boards/?$", flags=re.I)
			child = ""
			src_brd = pstr(f"{parent}/Boards/{row.file}")
			
			layout = prj.get("gitee", {}).get("layout", {})
			if "gitee" in prj and row.design in layout:
				usr_brd = pstr(f'{prj["gitee"]["project"]}/{layout[row.design]}')
			else:
				usr_brd = row_path + row.file

			pstr(src_brd.path()).mkdir()
			if src_brd.lower() != usr_brd.lower():
				src_brd.copyfrom(usr_brd)

		if not src_brd.isfile:
			return None
	
		cwd = cls(parent, child, tag=tag).create()
		simbrd = f"{cwd}/SimBrd"
		tgt_brd = pstr(f"{simbrd}/{row.file}")
		tgt_spd = tgt_brd.ext(".spd")
		
		if src_brd.after(tgt_brd) or tgt_brd.isfile is False:
			src_brd.copyto(tgt_brd)
		
		if not tgt_spd.after(tgt_brd):
			return simbrd
		
		return None

	def list(self):
		return {"parent": str(self.parent), "path": str(self.path)}

	def create(self, copybrd=None, copyee=False):
		self.parent.mkdir()
		if self.parent.isdir:
			self.Boards.mkdir()
		
		self.path.mkdir()
		if self.path.isdir:
			for attr in self.subdir:
				getattr(self, attr).mkdir()
		
		if copybrd:
			self.copy_brd(copybrd)
		
		if copyee and self.ee:
			self.copy_ee(self.ee)

		return self.path

	def s1p_fitted(self):
		"""rename _s.s1p to _dcfitted.s1p, in case sigrity failed fitting perfect open/short"""
		if not self.SimFiles.isdir:
			print(f"folders.s1p_fitted(): {self.SimFiles} is not a folder!")
			return

		for net in self.SimFiles.listdir():
			cwd = self.SimFiles + net
			if not cwd.isdir:
				continue

			dc_files = cwd.glob(f"{net}*_DC.s*p")
			if not dc_files:
				continue

			fdc = pstr(dc_files[0])
			if fdc.ext().lower() != ".s1p":
				continue

			dc_line = next((line for line in fdc.readlines() if line.startswith("0")), None)
			if not dc_line:
				continue

			z = re.split(r"\s+", dc_line)
			if len(z) < 2 or abs(float(z[1]) - 1) >= 1e-3:
				continue

			sources = cwd.glob(f"{net}*_S.s*p") + cwd.glob(f"{net}*_fit.s*p")
			if not sources:
				continue

			src = pstr(max(sources, key=os.path.getmtime))
			p, n, e = src.split()
			tgt = pstr(f"{p}/{re.sub(r'[^_]+$', 'dcfitted', n)}{e}")
			tgt.copyfrom(src)

	def simulated(self, ext="*_dcfitted*.s*p;*_fit.s*p", santize=1):
		# santize to clean existing simulation if
		# 2: newer database file is under /SimBrd
		# 1: exracted touchstone file has wrong number of ports
		# 0: do nothing
	
		def formal(names):
			return {s.lower().replace('-', '_') for s in names}

		simulated = {tcl: False for tcl in self.Scripts.glob("*.tcl") }
		databases = self.SimBrd.lstypes("*.spd,*.brd,*.tgz")
		last_brd = pstr(max(databases, key=os.path.getmtime))

		for tcl in simulated:
			net = pstr(tcl).name()
			simdir = self.SimFiles + net
			if not simdir.isdir:
				continue

			txt = pstr(tcl).read()
			np_expected = txt.count("sigrity::add port")
			pins_expected = formal(re.findall(r"sigrity::hook -port {(\w+)} -PositiveNode", txt))
			
			tsfiles = simdir.lstypes(ext)
			np_extracted, newer_brd, pins_extracted = 0, False, set()
			if tsfiles:
				last_ts = pstr(max(tsfiles, key=os.path.getmtime))
				np_extracted = int(last_ts.ext()[2:-1])
				newer_brd = last_brd.after(last_ts)
				pins_extracted = formal(tsfile(last_ts).pins())

			success = np_expected <= np_extracted  # and pins_expected.issubset(pins_extracted)
			failed = not success and santize > 0
			outdated = newer_brd and santize > 1

			if failed or outdated:
				for f in simdir.glob("*.*"):
					pstr(f).remove()

			# Refresh tsfiles after deletion
			tsfiles = simdir.lstypes(ext)
			done = self.SimFiles + f"{net}.done"
			
			if done.isfile and not tsfiles:
				done.remove()

			if tsfiles and not done.isfile:
				done.write("\n")

			simulated[tcl] = done.isfile

		return simulated

	def run_spd(self, spd, licenses=4):
		"""
		another version of batch run, each license open sequentially multiple baords,
		avoiding frequent license checkin/checkout
		"""
		_DEBUG = False
		self.s1p_fitted()
		nets = [x for x in self.SimFiles.listdir() if (self.SimFiles + x).isdir]
		spd.bnp2snp(nets)

		simulated = self.simulated(ext="*_dcfitted.s*p")
		previously = self.copy_snp(newcopy=True)
		to_simualte = {pstr(x).name(): x for x, v in simulated.items() if v is False}

		# create Run subfolders by number of license
		job_groups = {}
		for i in range(licenses):
			os.makedirs(self.Scripts + f"/Run{i}", exist_ok=True)
			job_groups.update({i: []})
		for i, (k, v) in enumerate(to_simualte.items()):
			job_groups[i % licenses].append(v)
		# assgin a batch Run tcl to each run sub folder
		for i, tcls in job_groups.items():
			lines = []
			run_tcl = self.Scripts + f"Run{i}/Run{i}.tcl"
			if run_tcl.isfile:
				run_tcl.remove()
			for tcl in tcls:
				s = pstr(tcl).read()
				s = re.sub(r"\nsigrity::exit -n {!}", "\n#sigrity::exit -n {!}\nsigrity::close document $spd_file {!}", s)
				lines.extend(s.splitlines())
			if len(lines):
				lines.append("sigrity::exit -n {!}")
				run_tcl.wirte("\n".join(lines))

		# now fire up the simulation
		jobs = {}
		for i in range(licenses):
			run_tcl = self.Scripts + f"Run{i}/Run{i}.tcl"
			if run_tcl.isfile:
				command = f'{spd._exe} -tcl "{run_tcl}"'
				job = subprocess.Popen(command, cwd=run_tcl.path())
				jobs.update({job: True})
				if _DEBUG:
					print(f"{run_tcl.base()}: {job}")
		if len(to_simualte):
			for i, net in enumerate(to_simualte):
				print(f"  nets {i+1}/{len(to_simualte)} in queque: {net}")
		else:
			print("  0 nets waiting to be extracted")
		# pull job status
		timeout = time.time() + len(to_simualte) * (21600 / licenses)  # 6 HOURS PER TCL
		i = 0
		while time.time() < timeout:
			time.sleep(1)
			self.s1p_fitted()
			for x, done in self.simulated(santize=False).items():
				t = pstr(x).name if done else ""
				if t and t in to_simualte and to_simualte[t]:
					to_simualte[t] = False
					print(f"  net {i+1}/{len(to_simualte)} extracted: {t}")
					i += 1
			for job, waiting in jobs.items():
				if waiting and job.poll() == 0:
					jobs[job] = False
					if _DEBUG:
						print(f"{job}")
			if any(jobs.values()) is False:
				break
		else:
			for job, waiting in jobs.items():
				if waiting:
					job.kill()

		# closing status
		nets = [x for x in self.SimFiles.listdir() if (self.SimFiles + x).isdir]
		spd.bnp2snp(nets)
		extracted = self.copy_snp(ext="*_dcfitted.s*p")
		return len(extracted) - len(previously) - len(to_simualte)

	def copy_brd(self, brdfile):
		# copy .brd or .spd, .brd has priority if both there
		srcd = pstr(brdfile).path()
		if len(srcd):
			src = pstr(brdfile)
			if not src.isfile:
				print(f"file {brdfile} not found")
				return 0
		else:
			if not self.Boards.isdir:
				print(f"folder does not exist: {self.Boards}")
				return 0
			if isinstance(brdfile, str):
				brds = [f"{self.Boards}/{brdfile}"]
				spds = []
			else:
				brds = self.Boards.glob("*.brd")
				spds = self.Boards.glob("*.spd")
			if (len(brds) < 1) and (len(spds) < 1):
				print(f"No .brd or .spd found under {self.Boards}")
				return 0
			# always pick the latest file as src for copy
			if bool(len(brds)):
				src = pstr(max(brds, key=lambda f: os.path.getmtime(f)))
			elif bool(spds):
				src = pstr(max(spds, key=lambda f: os.path.getmtime(f)))
		
		tgt = self.SimBrd + src.base()

		# no copy if alreay newer file in target folder
		files = [pstr(f).base() for f in self.SimBrd.glob("*.*")]
		if src.base() in files and tgt.after(src):
			return 0

		# copy and remove all other files in target folder
		tgt.copyfrom(src)
		for f in self.SimBrd.glob("*.*"):
			pf = pstr(f)
			if src.base() != pf.base():
				pf.remove()
		
		return 1

	def copy_snp(self, net="", newcopy=False, ext="*_dcfitted.s*p;*_FIT.s*p"):
		# copy snp to resutls folder and all names changed to _dicffitted
		copied = {}  # copied[tgt] = src or None( if already there)
		fitted_suffix  = re.compile(r"_dcfitted$", re.I)
		time_stamp = re.compile(r'(\w+)_(\d{6}_\d{6})_(.*)', re.I)

		def fitted_name(src):
			_, name, e = pstr(src).split()
			if fitted_suffix .search(name) is None:
				name = name.rpartition("_")[0] + "_dcfitted"
			return self.Result + f"{name}{e}"

		if newcopy:
			self.Result.clear()

		if net:  # copy just one file
			nets = [net] if (self.SimFiles + net).isdir else []
		else:
			nets = [n for n in self.SimFiles.listdir() if (self.SimFiles + n).isdir]
		for net in nets:
			snps  = (self.SimFiles + net).lstypes(ext)
			if len(snps ):
				src = max(snps , key=lambda f: os.path.getmtime(f))
				tgt = fitted_name(src)
				tgt.copyfrom(src)
				copied.update({str(tgt): str(src)})

		# keep just latest snp in Result folder for each net
		ts_by_net  = {}
		for f in self.Result.lstypes('*.s*p'):
			name  = pstr(f).name()
			net_name = z.group(1) if( z:=time_stamp.match(name )) else name 
			ts_by_net.setdefault(net_name, []).append(f)
		for files in ts_by_net.values():
			latest = max(files, key=lambda f: os.path.getmtime(f))  
			for ts in filter(lambda f: f != latest, files):
				pstr(ts).remove()

		return copied

	def archive(self, keep="*.spd;*dcfitted*.s*p;*_FIT.s*p"):

		if not self.path or not self.SimFiles.isdir:
			return

		regex  = wildcard_to_regex(keep.strip())
		retain = lambda f: re.search(regex , str(f), re.IGNORECASE)

		if self.Result.isdir:
			self.Result.clear()

		for item  in self.SimFiles.listdir():
			path = self.SimFiles + item 
			if path.isdir:  # this is a net folder
				for sub in filter(lambda f: (path + f).isdir, path.listdir()):
					(path + sub).remove()
				for f in filter(lambda f: not retain(f), path .glob("*.*")):
					pstr(f).remove()
			elif path.isfile:
				if not retain(path ):
					path.remove()
	
		print(f"only {keep} files left under {self.SimFiles}")
	
	def copy_ee(self, ee):
		src = pstr(f'{ee["project"]}{ee["layout"]}')
		tgt = pstr(self.Boards + "/" + src.base())
		tgt.copyfrom(src)
		if not tgt.isfile:
			print(f'failed copying file {ee["layout"]}')


class workflow:

	def __init__(self, func=None, title="", cli=None):
		if isinstance(cli, str):
			self.options = re.sub(r"(?m)^\s*|\n\s*\n", "", cli).splitlines()
		elif isinstance(cli, list):
			self.options = [x.strip() for x in cli]
		else:
			print("missing commandline options")
			return

		self.title = title or "none"
		self.func = func
		self.bulletins = [str(i) for i in range(len(self.options))]
		self.job = None

	def initiate(self, json_file):
		if not os.path.isfile(json_file):
			print("? Please provide a valid JSON file (Ctrl+V to paste)")
			return False

		print(f'"{os.path.abspath(json_file)}"')
		self.job = self.func(json_file)
		
		if self.job.gxls is None:
			print("? please check if json file has valid information")
			return False
		
		self.options.pop(0)
		print(f". Done {self.title} setup with {json_file}")
		return True
	
	# Example of reading input with tab completion
	def completer(self, text, state):
		matches = [opt for opt in self.options if opt.startswith(text)]
		return matches[state] if state < len(matches) else None

	def execute(self, step):
		# loptions = len(self.options)-1
		if not step:
			return
		
		choices = [opt for opt in self.options if opt.startswith(step[0])]
		if not choices:
			print(f"? Invalid step. Choose [0-{self.bulletins[-1]}]" if self.job else "? Choose step [0]")
			return
		
		if ":" in step:
			number, _, remainder = step.partition(":")
			command = remainder.strip()
			if number == "0":
				json_file = command.split("=")[-1].strip().strip("'\"")  # Handle quotes
				self.initiate(json_file)
			elif not self.job or self.job.gxls is None:
				print("? Step 0 is required or incomplete")
			elif number in self.bulletins:
				typo, fixed  = self.simple_fix(command)
				if typo is False:
					print(f"! {fixed }")
					eval("self.job." + fixed )
			else:
				print(f"? please choose a valid step [0-{self.bulletins[-1]}]:")
		elif self.job:
			command = ":".join(choices[0].split(":")[1:]).strip()
			print(f"! {self.title}.{command}")
			eval("self.job." + command)
		else:
			print(f"? please choose a valid step [0]:")
		
		self.reset_history()

	def simple_fix(self, s):
		"""boolean typo fix for user input cli"""
		raw_args = re.findall(r"=\s*([\"\'\w]*)[,\)]", s)
		args_class = {"str": [], "bool": [], "int": [], "none": [], "na": []}
		for val in raw_args:
			if val[0] in ('"', "'"):
				args_class["str"].append(val)
			elif val[0].upper() in ["T", "F"]:
				args_class["bool"].append(val)
			elif val[0].isdigit():
				args_class["int"].append(val)
			elif val.lower() == "none":
				args_class["none"].append(val)
			else:
				args_class["na"].append(val)

		s = re.sub(r"=\s*[tT]\w*([,\)])", r"=True\1", s)
		s = re.sub(r"=\s*[fF]\w*([,\)])", r"=False\1", s)
		typo = not raw_args or bool(args_class["na"])
		
		return typo, s

	def reset_history(self):
		readline.clear_history()
		for line in self.options:
			readline.add_history(line)

	def loop(self, args):
		"""Run the interactive loop, or accept command-line args."""
		# args = sys.argv
		# run example 1: clocks json_file_given
		if len(args) > 1:
			initiated = self.initiate(args[1])  # args[1] = json file
			if initiated and len(args) > 2:
				match = next((x for x in self.options if x.startswith(args[2])), None)
				if match:
					command = ":".join(match.split(":")[1:]).strip()
					print(f"! {self.title}.{command}")
					eval(f"self.job.{command}")
					sys.exit()
		# run example 3: clocks
		readline.set_completer(self.completer)
		readline.parse_and_bind("tab: complete")
		self.reset_history()
		try:
			while True:
				opts = "\n".join(["  " + re.sub(r"\(.+\)", "()", o) for o in self.options])
				print(f"> {self.title} workflow. up/down/esc/tab to edit, ctrl+d to quit:\n{opts}")
				step = input("> ")
				self.execute(step)
		except (EOFError, KeyboardInterrupt):
			print("terminated")

	def run(self, *args):
		"""Entry point for launching the workflow."""
		set_long_path()

		def flatten(e):
			for i in e:
				if isinstance(i, Iterable) and not isinstance(i, (str, bytes)):
					yield from flatten(i)  # Recursively flatten the item
				else:
					yield i

		input_args = tuple(flatten(args))
		if debug_enabled():
			self.loop(input_args)
		else:
			self.loop(sys.argv)

def proj_archive( keep ="*.spd,*.s*p,*.cir" ):
	"""trim simulation folder to leave only .spd|snp files"""
	# proj_root=[r'C:\Projects\P25\FL5\clock\mlb']
	root_folders = [r"c:\proj\p26"]
	proj_pat = re.compile(r"^(ck|hs|pp|run)(_\d{8}_\d{4})$", re.I)

	for root in filter(lambda f: pstr(f).isdir, root_folders):
		for dirpath, dirnames, _ in os.walk(root):
			for dirname in filter(proj_pat.match, dirnames):
				folders(parent=dirpath, child=dirname, tag="").archive(keep=keep)

if __name__ == "__main__":
	logme()
	print("main")
	proj_archive()
	input("press entre key...")
