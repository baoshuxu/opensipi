"""
document:
https://docs.google.com/document/d/1BTvaRC3c_gJUi9-sVTEFp-4wAWHW2jtR_uDU8vsHFFE/edit?resourcekey=0-xJz_s2OG2xdpDWneNeh1mg&tab=t.fj2mc5dqwfc1#heading=h.d48oab2eqobt
to run:
python.exe clocks.py ds_quasar_clocks.json 3
to compile: ( --onedir vs --onefile for faster boot)
pyinstaller --onedir  --uac-admin --add-data "support/nexxim_gui.py;support" --add-data "support/credentials.json;support" clocks.py --hidden-import openpyxl.cell._writer

"""

import os
import re
import json
import copy
import time
import signal
import itertools
import webbrowser
import pandas as pd
import networkx as nx
import support.templates as tmpl
from datetime import datetime
from support import gservices
from support.folders import folders, logme, dos, workflow, pstr, license
from support.ibis import ibis
from support.cadence import allegro, sigrity, component, functionpin, spdlinks, touchstone
import multiprocessing

fwd_slash = lambda s: s.replace("\\", "/")
rm_pattern = lambda p, s, f=0: re.sub(p, "", s, count=0, flags=f)


rex_gnd = re.compile(r".*gnd(\d)?$", re.I)  # regex pattern for GND nets
row_active = lambda row: not row.hide and "*" not in row[1][:1] + row[1][-1:]
# first memeber (e.g card, design) with * in it is equivlent as hiding


def seconds_to_hms(seconds):
  hours = seconds // 3600
  minutes = (seconds % 3600) // 60
  seconds = seconds % 60
  return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


class multiplex:

  def __init__(self, nodes={}, edges=[], switch=None):

    self.nodes = nodes  # a dictionary { row: (grp, card, pins)}, might just need to be { row: grp}
    self.edges = edges  # a list of tuples
    self.switch = switch  # a dataframe, the switch sheet
    self.G = self.base_graph()

    self._mux = {}
    self._mask = {}

  def _contends(self, seq):
    if len(self._mask):
      num = int("".join(map(str, seq)), 2)
      for mask in self._mask:
        if (num & mask) == mask:
          return True
    return False

  def _contest_mask(self, by=None):
    """set bit mask for contending mux combinations which will be skipped in searching
    by= None -> no mask
    by='left_right' -> a xnet cannot have pins on multiple mux at same side
    type(by)== list -> the list of (mux1,mux2,...muxn) tuples is given
    """
    if by is None:
      self._mask = {}
    elif by == "left_right":
      muxOrd = {u: i for i, (u, v) in enumerate(self._mux.items())}
      for ln, (grp, card, pins) in self.nodes.items():
        # all card refes and pin of this ln
        crdUs = {f"{card}.{u}": n for (u, n) in [e.split(".") for e in pins.split(",")]}
        # and if present on self._mux
        muxUs = {u: (n, self._mux[u]) for u, n in crdUs.items() if u in self._mux}
        # which mux has left pins in ln
        lUs = {}
        # which mux has right side pins in ln
        rUs = {}
        for u, (n, chs) in muxUs.items():
          p1 = [e for pos in chs for e in pos[0::2]]  # mux pins at left ports
          p2 = [e for pos in chs for e in pos[1::2]]  # mux pins at right ports
          if n in p1:
            lUs.update({u: muxOrd[u]})
          elif n in p2:
            rUs.update({u: muxOrd[u]})
        # if xnet ln has pins on same left side of multiple muxs:
        if len(lUs) > 1:
          bits = ["1" if u in lUs else "0" for u in self._mux]
          self._mask[int("".join(bits), 2)] = True
        # if xnet ln has pins on same right side of multiple muxs:
        if len(rUs) > 1:
          bits = ["1" if u in rUs else "0" for u in self._mux]
          self._mask[int("".join(bits), 2)] = True
    elif isinstance(by, list):
      for mux in by:
        hitUs = [u for u in mux if u in self._mux]
        if len(hitUs) > 1:
          bits = ["1" if u in hitUs else "0" for u in self._mux]
          self._mask[int("".join(bits), 2)] = True
      self._mask = {}

    return self._mask

  def _bind_mux(self, _mux, by="left_right"):
    """set bit mask for contending mux combinations which will be skipped in searching
    by= None -> no mask
    by='left_right' -> a xnet cannot have pins on multiple mux at same side
    type(by)== list -> the list of (mux1,mux2,...muxn) tuples is given
    """
    if by is None:
      self._mask = {}
      self._mux = {i: [[f"{u}.{p}" for p in ch] for ch in pos] for i, (u, pos) in enumerate(_mux.items())}
    elif isinstance(by, str):
      groups = {}
      muxOrd = {u: i for i, (u, v) in enumerate(_mux.items())}
      for ln, (inteface, card, pins) in self.nodes.items():
        # all card refes and pin of this ln
        crdUs = {f"{card}.{u}": n for (u, n) in [e.split(".") for e in pins.split(",")]}
        # and if present on _mux
        muxUs = {u: (n, _mux[u]) for u, n in crdUs.items() if u in _mux}
        # which mux has left pins in ln
        lUs = {}
        # which mux has right side pins in ln
        rUs = {}
        for u, (n, chs) in muxUs.items():
          p1 = [e for pos in chs for e in pos[0::2]]  # mux pins at left ports
          p2 = [e for pos in chs for e in pos[1::2]]  # mux pins at right ports
          if n in p1:
            lUs.update({u: muxOrd[u]})
          elif n in p2:
            rUs.update({u: muxOrd[u]})
        # if xnet ln has pins on same left side of multiple muxs:
        if len(lUs) > 1:
          bits = ["1" if u in lUs else "0" for u in _mux]
          self._mask[int("".join(bits), 2)] = True
          if "left" in by:
            saw = [gn for gn, gv in groups.items() if any(x in gv for x in lUs)] if len(groups) else []
            if saw:
              groups[saw[0]].update(lUs)
            else:
              groups.update({len(groups): lUs})
        # if xnet ln has pins on same right side of multiple muxs:
        if len(rUs) > 1:
          bits = ["1" if u in rUs else "0" for u in _mux]
          self._mask[int("".join(bits), 2)] = True
          if "right" in by:
            saw = [gn for gn, gv in groups.items() if any(x in gv for x in rUs)] if len(groups) else []
            if saw:
              groups[saw[0]].update(rUs)
            else:
              groups.update({len(groups): rUs})

      linked = []
      if len(groups):
        linked = [list(x.keys()) for k, x in groups.items()]
        print(f"multiplex._bind_mux(): linked multiplexers detected:\n {str(linked)}")
      ulink = [x for grp in linked for x in grp]
      self._mux = {}
      for u, pos in _mux.items():
        if u in ulink:
          continue
        self._mux.update({len(self._mux): [[f"{u}.{p}" for p in ch] for ch in pos]})
      for grp in linked:
        comb = []
        for u in grp:
          pos = _mux[u]
          comb += [[f"{u}.{p}" for p in ch] for ch in pos]
        self._mux.update({len(self._mux): comb})
    elif isinstance(by, list):
      # todo
      self._mask = {}
      self._mux = {i: [[f"{u}.{p}" for p in ch] for ch in pos] for i, (u, pos) in enumerate(_mux.items())}

    return self._mux

  def get_on_mux(self, pins, n):
    """report which path the swithes are on for nth combination"""
    combs = (range(len(pos)) for _, pos in self._mux.items())
    ch = next(itertools.islice(itertools.product(*combs), n, n + 1), None)
    if ch is not None:
      visited = {p: False for p in pins}
      paired = []

      for p in filter(lambda k: visited[k] is False, visited):
        for i, v in self._mux.items():
          if p in v[ch[i]]:
            j = v[ch[i]].index(p)
            if j % 2:
              p1, p2 = v[ch[i]][j - 1], p
            else:
              p1, p2 = p, v[ch[i]][j + 1]
            paired += [(p1, p2)]
            visited.update({p1: True, p2: True})
      detached = [x for (x, yes) in visited.items() if not yes]
      return paired, detached

  def base_graph(self, node=None, edge=None):
    G = nx.Graph()
    nodes = [x for x in self.nodes] if node is None else node
    G.add_nodes_from(nodes)
    edges = self.edges if edge is None else edge
    G.add_edges_from(edges)
    return G

  def add_mux(self, mux_nodes, bind="left_right"):
    """pick true mux pins, can help reduce size of graph"""

    self.switch["masked"] = None
    mux = {}  # {'soc.U14322':{'1': ['A2', 'D1'], '2': ['A3', 'D1']}}
    true_mux = {}

    for ln in self.switch.itertuples():
      cardU = f"{ln.card}.{ln.refdes}"
      pin_numbers = list(filter(None, re.split(r"[\(\s\)]+", ln.connection)))
      mask = [True if f"{cardU}.{p}" in mux_nodes else False for p in pin_numbers]
      masked = []
      for i in range(0, len(mask), 2):
        if all(mask[i : i + 2]):
          masked.extend(pin_numbers[i : i + 2])
      self.switch.at[ln.Index, "masked"] = masked
      # before masking mux['mb.U1493']= {'1': ['A1', 'C1', 'A3', 'C3'], '2': ['D1', 'C1', 'D3', 'C3']}
      # after masking: mux['mb.U1493']= {'1': ['A1', 'C1'], '2': ['D1', 'C1', 'D3', 'C3']}
      if masked:
        if cardU in mux:
          mux[cardU].update({ln.position: masked})
        else:
          mux[cardU] = {ln.position: masked}
      # a whole channel position eliminated if no pins found in mux_nodes
      else:
        pass
    spst_tuple = []
    n_combinations = 1
    for cardU, channels in mux.items():
      # mux can end up being spst, make edges for graph
      if len(channels) < 2:
        pin_numbers = list(channels.values())[0]
        for p1, p2 in zip(pin_numbers[::2], pin_numbers[1::2]):
          spst_tuple += [(f"{cardU}.{p1}", f"{cardU}.{p2}")]
      # true mux channel counts in total combinations
      else:
        true_mux.update({cardU: [v for v in channels.values()]})
        n_combinations *= len(true_mux[cardU])

    # apply spst edges
    if spst_tuple:
      self.G.add_edges_from(tuple(spst_tuple))
    # true mux
    # self._mux = true_mux
    # self._mask = self._contest_mask(by=contention)
    self._mux = self._bind_mux(true_mux, by=bind)

    # _comb for itertools islice and products
    # self._comb = [range(len(y)) for _,y in true_mux.items()]
    n_combinations = 1
    for n, ch in self._mux.items():
      n_combinations *= len(ch)
    return n_combinations

  def walk_block(self, n, chunck):
    """Return chuncks of paths found instead of one walk for multiprocessing."""

    logme.info(f"Computing {n}-{n+chunck}")
    print(f"Computing {n}-{n+chunck}")

    # Cache self.nodes and self._mux to local variables
    nodes = self.nodes
    mux = self._mux

    # Generate combinations efficiently
    combs = (range(len(pos)) for _, pos in mux.items())
    comb_slice = itertools.islice(itertools.product(*combs), n, n + chunck)

    # pins = {}
    labels = {}

    # Iterate over the generated channels
    for cnt, ch in enumerate(comb_slice):
      if ch is None:
        break
      edges = [
          (p[ch[i]][j], p[ch[i]][j + 1]) for i, (_, p) in enumerate(mux.items()) for j in range(0, len(p[ch[i]]), 2)
      ]
      # G = nx.Graph(self.G)  # This is a shallow copy
      G = copy.deepcopy(self.G)  # need this for compiled exe
      G.add_edges_from(edges)

      # Process connected components
      for vertices in nx.connected_components(G):
        rows = sorted(v for v in vertices if isinstance(v, int))
        if len(rows) > 1 and any(nodes[i][0] != "~mux" for i in rows):
          k = tuple(rows)
          if k not in labels:
            # pin = ':'.join(f'{nodes[i][1]};{nodes[i][2]}' for i in rows)
            labels[k] = tuple(v for v in vertices if isinstance(v, str)) + (n + cnt,)
            # pins[pin] = True

    return labels

  def walk_block_obselete(self, n, chunck):
    """return chuncks of paths found instead of one walk for multiprocessing"""
    # channels=list(itertools.islice(itertools.product(*self._comb), n, n+chunck))
    combs = (range(len(pos)) for _, pos in self._mux.items())
    channels = list(itertools.islice(itertools.product(*combs), n, n + chunck))
    # logme.info(f"Computing {n}-{n+chunck}")
    # pins = []
    pins = {}
    labels = {}
    for this_channel, this_iteration in enumerate(range(n, n + chunck)):
      ch = channels[this_channel]
      if ch is None:
        break
      # if self._contends(ch): break
      edge = []
      # for i,(cardU,p) in enumerate(self._mux.items()):
      # 	for p1, p2 in zip( p[ch[i]][0::2],p[ch[i]][1::2]):
      # 		edge.append((f'{cardU}.{p1}',f'{cardU}.{p2}'))
      for i, (n, p) in enumerate(self._mux.items()):
        for p1, p2 in zip(p[ch[i]][0::2], p[ch[i]][1::2]):
          edge.append((p1, p2))
      G = copy.deepcopy(self.G)
      G.add_edges_from(tuple(edge))
      for vertices in nx.connected_components(G):
        rows = sorted(v for v in vertices if isinstance(v, int))
        if len(rows) > 1 and any(self.nodes[n][0] != "~mux" for n in rows):
          pin = ":".join([f"{self.nodes[i][1]};{self.nodes[i][2]}" for i in rows])
          if pin not in pins:
            this_label = ",".join([v for v in vertices if isinstance(v, str)])
            labels.update({f"{rows}": (this_label, this_iteration)})
            # pins += [pin]
            pins.update({pin: True})
    return labels

  def walk_path(self, n):
    """the islice and product functinn calcuate comb directly,
    aving stroge for list of combinations which can be huge
    """
    combs = [range(len(pos)) for _, pos in self._mux.items()]
    ch = next(itertools.islice(itertools.product(*combs), n, n + 1), None)
    if ch is None:
      print("multiplex.walk_path(): combinations exhausted")
      return
    edge = []
    for i, (u, p) in enumerate(self._mux.items()):
      for p1, p2 in zip(p[ch[i]][0::2], p[ch[i]][1::2]):
        edge.append((f"{u}.{p1}", f"{u}.{p2}"))
    G = copy.deepcopy(self.G)
    logme.info(f"Computing {n}")
    G.add_edges_from(tuple(edge))
    connected = list(nx.connected_components(G))
    lines, attached, pins = [], [], []
    for i, v in enumerate(connected):
      rows = sorted([x for x in v if isinstance(x, int)])
      if len(rows) < 2 or all(self.nodes[x][0] == "~mux" for x in rows):
        continue
      lines += [rows]
      attached += [[x for x in v if isinstance(x, str)]]
      # the joining and f-string actually takes a lot of time!
      pins += [":".join([f"{self.nodes[i][1]};{self.nodes[i][2]}" for i in rows])]
    d = {"rows": lines, "pins": pins, "attached": attached}
    return d  # multiprocess need function returns primitives, no df
    # df = pd.DataFrame()
    # return df


def worker(mux, n, size):
  # multiprocessing  worker must be in __main__ namespace
  return mux.walk_block(n, size)


class nexxim:
  keep_lastest_snp_only = True
  def __init__(self, cwd, topo=None, ibs=None, copyby=None):
    self.cwd = cwd
    # self.parts = {}
    # self.nets = {}
    self.ibs = pd.DataFrame()
    self.topo = pd.DataFrame()
    self.switches = pd.DataFrame()

    self.catalog = {}  # ibis models for each ibis file
    self.param = {}  # .param lines in nexxim cir file
    self.smodel = {"count": 0, "file": []}  # snp models in nexxim cir  file

    # load topology(or later as arg to build_deck()), ibis and copy board s-parameters
    if cwd:
      os.makedirs(cwd, exist_ok=True)
      for p in [cwd, cwd + "/.snp", cwd + "/.ibs", cwd + "/.log"]:
        os.makedirs(p, exist_ok=True)
    if topo is not None and isinstance(topo, pd.DataFrame):
      self.topo = topo
    if ibs is not None and isinstance(ibs, pd.DataFrame):
      print("nexxim.__init__(): reading ibis files ...")
      self.ibs = self.load_ibs(ibs)
    if copyby is not None and isinstance(copyby, pd.DataFrame):
      self.load_snp(by=copyby, copys=True)

    # copy nexxim_gui.py, w/ modified IronPython path from user ansys installtion
    src = pstr(os.path.dirname(__file__).replace("\\", "/") + "/support/nexxim_gui.py")
    if src.isfile:
      logme.info(f"copying 'nexxim_gui.py' to {cwd}")
      ansysroot = re.compile(r"ANSYSEM_ROOT\w+")
      ansysenv = {key: val for key, val in os.environ.items() if ansysroot.match(key)}
      tgt = pstr(cwd + "/nexxim_gui.py")
      nexxim_gui_py = src.read()
      if ansysenv:
        ansysver = ansysenv[sorted(list(ansysenv.keys()))[-1]]
        src_dll = r"C:/Tools/AnsysEM/v241/Win64/common/IronPython/DLLs"
        tgt_dll = fwd_slash(ansysver) + "/common/IronPython/DLLs"
        nexxim_gui_py = re.sub(src_dll, tgt_dll, nexxim_gui_py)
      else:
        print("ansysem path not found, please check line5/nexxim_gui.py")
      tgt.write(nexxim_gui_py)

  def set(self, k, v):
    exec(f"self.{k} = {v}")

  def _rate(self, clk):
    if not clk:
      return 0
    d = {"g": 1e9, "m": 1e6, "k": 1e3}
    rate = rm_pattern(r"hz$",clk.lower())
    v, u = (rate[:-1], rate[-1])
    if u not in d:
      v, unit = rate, 1
    else:
      unit = d[u]
    return float(v) * unit * 2

  def _bstr(self, type, node, n, isTx):
    """buffer string for nexxim b-element"""
    port = node.replace(".", "_")
    if type == "input":
      bline = f"b_{port} nc{n:03d} nc{n+1:03d} {port} nc{n+2:03d}"
      strtype = f"buffer={type}"
    elif type in ["output", "open_drain", "open_sink", "open_srouce"]:
      bline = f"b_{port} nc{n:03d} nc{n+1:03d} {port} nc_{n+2:03d} nc{n+3:03d} nc{n+4:03d}"
      strtype = f"buffer={type}"
    elif type in ["tristate", "3-state"]:
      bline = f"b_{port} nc{n:03d} nc{n+1:03d} {port} nc{n+2:03d} nc{n+3:03d} nc{n+4:03d} nc{n+5:03d}"
      strtype = f"buffer=three_state buffer_mode=__output__ use_buffer_mode=true"
    elif type == "terminator":
      bline = f"b_{port} nc{n:03d} nc{n+1:03d} {port}"
      strtype = f"buffer={type} buffer_mode=__input__ use_buffer_mode=true"
    elif type in [
        "i/o",
        "input_output",
        "i/o_open_drain",
        "i/o_open_sink",
        "i/o_open_srouce",
    ]:
      bline = f"b_{port} nc{n:03d} nc{n+1:03d} {port} nc{n+2:03d} nc{n+3:03d} nc{n+4:03d} nc{n+5:03d} nc{n+6:03d}"
      strtype = f'buffer={type.replace("i/o","input_output")} buffer_mode=__input__ use_buffer_mode=true'
      if isTx:
        strtype = f'buffer={type.replace("i/o","input_output")} buffer_mode=__output__ use_buffer_mode=true'
    return bline, strtype

  def _binfo(self, pin, isTx=True):
    """buffer information"""
    bfile, bcomp, bpin, bmodel, btype = "file", "comp", "A1", ["model"], "output"
    card, refdes, bpin = pin.split(".")
    # if refdes.upper()[0] in 'J P'.split():
    #     return '', '', '', [''], 'RX'
    ib = self.ibs.index[self.ibs["cardu"] == f"{card}.{refdes}"].tolist()
    if not bool(ib):  # not listed in ibis table, assuming RX at connector
      return "", "", "", [""], "RX"
    i = ib[0]
    bfile = self.ibs.at[i, "file"]
    bcomp = self.ibs.at[i, "component"]
    if bcomp.lower().startswith("c="):  # the io is specified as C in ibis table
      return "", bcomp[2:], "", [""], "CAP"
    if not bool(bfile):
      logme.warning(f"  {pin} has no ibis file specified!")
      return "", "", "", [""], "RX"
    if not pstr(bfile).isfile:
      logme.warning(f"  {pin} failure find ibis file{bfile}")
      return "", "", "", ["NC"], ""
    if not bfile in self.catalog:
      self.catalog.update({bfile: ibis(bfile)})
    ibs = self.catalog[bfile]
    if not bcomp in ibs.comp():
      logme.warning(f"  {pin} failure find component {bcomp} in {bfile}")
      return "", "", "", ["NC"], ""
    ibspin = ibs.pin(bpin, bcomp)
    if not bool(ibspin):
      logme.warning(f"  {pin} failure find  pin {bpin} in {bfile}")
      return "", "", "", [""], "RX"
    buffers = ibspin["model_name"]
    if not bool(buffers):
      logme.warning(f"  {pin} failure find buffer {ibspin['model_name']} in {bfile}")
      return "", "", "", [""], "RX"
    bmodel = list(list(buffers.values())[0].keys()) if isinstance(buffers, dict) else [buffers]
    if "NC" in bmodel:
      logme.info(f"  Buffer model is NC for {pin} in {bfile}")
      return "", "", "", ["NC"], ""
    logme.info(f"  {pin} assigned with buffer ")
    btypes = {b: ibs.spec(b).get("io") for b  in bmodel}
    
    #filter models if multiple for Tx and Rx
    if isTx:
      bmodel = [k for k, v in btypes.items() if v!='i']
    else:
      bmodel = [k for k, v in btypes.items() if v=='i'] + [k for k, v in btypes.items() if v=='i/o']
      bmodel = bmodel[:1]
      # todo: what rx selector has dual rails of i/o buffers
    btype = ibs.spec(bmodel[0]).get("type")
      
    return bfile, bcomp, bpin, bmodel, btype

  def load_snp(self, by=None, copys=True):
    if self.topo.empty or by is None:
      return
    cwd = self.cwd + "/.snp"
    # clear /.snp folder if new copy
    # if copys:
    # 	for ts in glob.glob(f'{cwd}/*.s*p'):
    # 		os.remove(ts)
    # copy board snp files if specified by snp table
    if True:
      sfile, s = ({}, {})
      for dsn in by.itertuples():
        src = pstr(dsn.simpath + "/" + dsn.file)
        if re.search(r"\.s\d+p$", str(src), re.I):
          if src.isfile:
            tgt = pstr(cwd + "/" + dsn.file)
            tgt.copyfrom(src=src)
        else:
          p = pstr(os.path.dirname(dsn.simpath))
          dir = folders(p.path(), p.base())
          # dir.copy_snp(copys = True)
          # dir.archive()
          dcfitted = pstr(dir.Result).glob("*_DCfitted*.s*p") + pstr(dir.Result).glob("*_FIT.s*p")
          sfile[dsn.design] = dcfitted
          s[dsn.design] = [pstr(x).base() for x in dcfitted]
    # copy sigrity extraction snp
    if copys:
      copied, nofound = [], []
      logme.info(f"copying board snp files to {cwd}")
      for ln in self.topo.itertuples():
        i = ln.Index
        # i, design in enumerate(self.topo['design']):
        thisnet = rm_pattern(r"/(\d*)$", ln.xnet)
        found = [x for x in s[ln.design] if re.search(rf"^{thisnet}_", x)]

        if len(found) > 1:
          with_date = [x for x in found if re.search(rf"^{thisnet}_\d+_\d+_\d+_", x)]
          if with_date:
            found = with_date
        if len(found) > 1:
          logme.warning(f"more than one touchstone found for {thisnet}: {','.join(found)}")

        if len(found) < 1:
          nofound.append(thisnet)
        else:
          srcs = [sfile[ln.design][s[ln.design].index(e)] for e in found]
          srcs.sort(key=lambda f: os.path.getmtime(f))
          src = pstr(srcs[-1])
          tgt = pstr(f"{cwd}/{ln.design}.{src.base()}")
          if not (bool(copied) and (tgt in copied)):
            lines, pound = self.s_header(src)
            tgt.write("\n".join(lines))
            copied.append(tgt)
      if len(nofound):
        pass

    # list all snp files in /.snp folder to 'sfile column in self.topo
    if copys:
      tsfiles = pstr(cwd).glob("*_DCfitted*.s*p") + pstr(cwd).glob("*_FIT.s*p")
    else:
      tsfiles = []
      for dsn, flist in sfile.items():
        for f in flist:
          tsfile = pstr(f).path() + f"/{dsn}." + pstr(f).base()
          tsfiles.append(tsfile)

    bynet = {}
    for s in tsfiles:
      base = pstr(s).base()
      timed = re.findall(r"^(.*)_\d+_\d+_\d+_dcfitted", base, re.I)
      notimed = re.findall(r"^(.*)_dcfitted", base, re.I)
      clarity = re.findall(r"^(.*)_FIT\.s\d+p$", base, re.I)
      if timed:
        bynet[timed[0]] = bynet.get(timed[0], []) + [s]
      elif notimed:
        bynet[notimed[0]] = bynet.get(notimed[0], []) + [s]
      elif clarity:
        bynet[clarity[0]] = bynet.get(clarity[0], []) + [s]

    self.topo["sfile"] = ""
    for ln in self.topo.itertuples():
      tag = f"{ln.design}." + rm_pattern(r"/(\d*)$", ln.xnet)
      if tag in bynet:
        srcs = bynet[tag]
        if copys is False:
          srcs = [pstr(x).path() + "/" + rm_pattern(rf"^{ln.design}\.", pstr(x).base(), f=re.I) for x in srcs]
        srcs.sort(key=lambda f: os.path.getmtime(f))
        self.topo.at[ln.Index, "sfile"] = srcs[-1]
        if len(srcs) > 1 and self.keep_lastest_snp_only and copys:
          for f in srcs[0:-1]:
            pstr(f).remove()
      else:
        print(f"nexxim.load_snp(): touchstone file not found for {tag}")
    print("nexxim.load_snp(): existing scopy();")

  def load_ibs(self, df_ibs):

    copied = {}

    df = df_ibs.reset_index(drop=True).astype(str)
    df = df[df["path"].str.cat(df["file"]) != ""]
    df = df.assign(srcfile=df["file"].apply(fwd_slash))
    df.loc[:, "file"] = ""
    df.loc[:, "component"] = df["component"].apply(lambda s: rm_pattern(r"\s", s))
    df.loc[:, "path"] = df["path"].apply(fwd_slash)

    # copy ibis file to .ibs folder
    # self.ibs['file'] = ['' for _ in self.ibs['file']]
    for row in df.itertuples():
      i, p, f = row.Index, row.path, row.srcfile
      if re.search(r"\[.+\].+\.s\d+p$", f, re.I):
        port, f = re.search(r"(\[.+\])(.+)", f).groups()
        df.at[i, "component"] = port
      src = f"{p}/{f}" if p else f
      dir = ".ibs" if re.search(r"\.ibs$", f, re.I) else ".snp"
      if pstr(src).isfile and (src not in copied):
        tgt = f"{self.cwd}/{dir}/{f}"
        open(tgt, "wb").write(open(src, "rb").read())
        copied[src] = tgt
      if src in copied:
        df.at[i, "file"] = copied[src]
      else:
        print(f'nexxim.load_ibs(): file "{src}" not found')

    # load ibis file to self.catalog
    df.pop("srcfile")
    for row in df.itertuples():
      i, (comp, file) = row.Index, (row.component, row.file)
      if not file:
        continue
      if file in self.catalog:
        continue
      ext = pstr(file).ext().lower()
      if ext == ".ibs":
        print(f'nexxim.load_ibs(): reading ibis "{file}"')
        ibs = ibis(file, "short")
        if comp not in ibs.comp():
          logme.warning(f'  "{comp}" not found in file "{file}"')
        else:
          self.catalog.update({file: ibs})
      elif re.match(r"\.s\d+p$", ext):
        pass
      elif ext in [".mod", ".sp", ".inc"]:
        with open(file, "r") as f:
          for line in f:
            text = line.strip().lower()
            if text.startswith(".subckt"):
              words = re.split(r"\s+", line.strip())
              df.at[i, "component"] = words[1]
              break
        continue

    df.pop("path")
    self.ibs = df
    return self.ibs

  def list_net(self, topo=None):
    if topo is None:
      topo = self.topo
    if not ("driver" in self.topo.columns):
      print(f"nexxim.list_net(): need the driver column in topology sheet!")
      return
    byroute = topo.groupby("route")
    routes = list(byroute.groups.keys())
    circuits = {"no": routes}
    circuits.update({"rows": [[] for _ in routes]})
    circuits.update({"bps": [0 for _ in routes]})
    circuits.update({"net": ["" for _ in routes]})
    circuits.update({"text": [{} for _ in routes]})

    def _slash(x):
      return True if re.search(r"/(\d*)$", x) else False

    for i, rn in enumerate(routes):
      df = byroute.get_group(rn)
      circuits["rows"][i] = df.index[df["route"] == rn].tolist()
      ix = df.index[df["xnet"].apply(_slash)].to_list()
      if len(ix):
        circuits["bps"][i] = self._rate(df.at[ix[0], "speed"])
        circuits["net"][i] = re.sub(r"/(\d*)$", lambda m: f"_{m.group(1)}" if m.group(1) else "", df.at[ix[0], "xnet"])
      else:
        circuits["bps"][i] = self._rate(df.at[df.index[0], "speed"])
        circuits["net"][i] = df.at[df.index[0], "xnet"]

      # for no, rows in enumerate(circuits['rows']):
      if circuits["bps"][i] in ["0", 0]:
        print(f"nexxim.list_net(): {circuits['net'][i]} skipped with bps=0")
        continue

      stropt = f"*multiboard clock net: {circuits['net'][i]}\n\n" + ".option PARHIER='local'\n.option max_messages=1\n"
      self.param = {
          "bps": circuits["bps"][i],
          "iTxBuffer": "0",
          "iRxCLoad": "0",
          "iCorner": "0",
          "Corner": '{"typ","fast","slow"}',
          "RxCLoad": "2e-12",
      }
      self.smodel = {"count": 0, "file": []}

      drivers, probes, attached = [], [], []
      for row in df.itertuples():
        attached += row.attached.split(",") if row.attached else []
        probes.extend(f"{row.card}.{x}" for x in row.io.split(","))
      probes = list(set(probes) - set(attached))

      joint = eval(df.at[df.index[0], "joint"]) if attached else {}  # python: cannot eval('')
      sline, jline = self.j_lines((joint, attached), "")
      # sline, jline = self.j_lines(attached, '')
      strchn = "*board connection\n" + jline + "\n"
      for row in df.itertuples():
        ports, sline = self.s_lines(row, sline)
        if not sline:
          print(f"nexxim.list_net(): {circuits['net'][i]}")
          print(f" sfile not found for topology row {row.Index}(+2)")
          continue

        # ports, tline = self.t_lines(row, ports) # may have mux components
        tline = ""
        ports, xline = self.x_lines(row, ports)  # may have sub circuit (e.g.FB*)
        ports, cline = self.c_lines(row, ports)  # c_lines() may add param to self.param
        ports, rline = self.r_lines(row, ports)  # r_lines() may add param to self.param
        if bool(ports):
          print(f"nexxim.list_net(): {circuits['net'][i]}")
          print(f" port {','.join(ports)} left NC in {pstr(row.sfile).base()}")
          # raise ValueError(f'Nodes not consistence for row {row.Index}')
        strchn += f"\n*** board {row.card} ***\n" + sline + tline + xline + cline + rline

      # dirvers
      for row in df.itertuples():
        if bool(row.driver):
          for u in row.driver.split(","):
            drivers.extend([y for y in probes if y.startswith(f"{row.card}.{u}")])
      for tx in drivers:
        strbuf, strcmd = self.b_lines(tx, probes)  # b_lines() may add param to self.param
        strparam = "\n".join([f".param {k} = {v}" for k, v in self.param.items()]) + "\n"
        text = "\n".join([stropt, strparam, strbuf, strchn, strcmd])
        circuits["text"][i].update({tx: text})

    if True:
      for i, tx_text in enumerate(circuits["text"]):
        if not bool(text):
          logme.info(f"route {i} .cir not generated")
          continue
        for tx, lines in tx_text.items():
          # may have duplicate ele card due to mulitplexing
          txt = lines.split("\n")
          ele = {}  # { ele: repeat_count}
          for j, ln in enumerate(txt):
            # z = re.search(r'^([SRcbv]_\S+)\s',ln)
            z = re.search(r"^([S]_\S+)\s", ln)
            if z is None:
              continue
            e = z.group(1)
            if e in ele:
              ele[e] += 1
              txt[j] = re.sub(rf"^{e}", f"{e}{ele[e]}", ln)
            else:
              ele[e] = 0
          lines = "\n".join(txt)
          # end checking dulicates
          netname = re.sub(r"/(\d*)$", r"\1", circuits["net"][i])
          outfile = f"{netname}_{tx}"
          outdir = pstr(self.cwd) + outfile.replace(".", "_")
          outdir.mkdir()
          with open(f"{outdir}/{outfile}.cir", "w") as f:
            f.write(lines)

    return circuits

  def s_lines(self, row, previous):
    # touchstone circuit elements
    port, lines = ({}, "")
    sfile = row.sfile
    if not pstr(sfile).isfile:
      return port, lines

    card = row.card
    with open(sfile, "r") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        if line.startswith("#"):
          break
        line = line.replace("-", "_")
        z = re.match(r"^! Port(\d+)_(.+)::.+", line)
        if bool(z):
          port.update({z.group(1): z.group(2).split()[0].upper()})
        z = re.match(r"^! Port (\d+) = (.+)", line)
        if bool(z):
          port.update({z.group(1): z.group(2).split()[0].upper()})
    ports = [f"{card}_{x}" for x in list(port.fromkeys(port.values()))]

    # s line
    sn = 1 + len(self.smodel["file"])
    if sfile not in self.smodel["file"]:
      lines += f'* touchstone {os.path.basename(sfile)}\n.model sModel{sn} S TSTONEFILE="{sfile}"\n'
      lines += f"+ INTERPOLATION=LINEAR INTDATTYP=MA HIGHPASS=10 LOWPASS=10 convolution=0 enforce_passivity=0 Noisemodel=External\n"
      self.smodel["file"].append(sfile)
    sn = 1 + self.smodel["file"].index(sfile)
    lines += f"S_{card} {' '.join(ports)} " + f'FQMODEL="sModel{sn}"\n'
    self.smodel["count"] += 1
    # check the ports in sfile agrees with topology table
    for io in row.io.split(","):
      p = card + "_" + io.replace(".", "_").upper()
      if p not in ports:
        logme.info(f"io {io} in topology row {row.Index} not found in touchstone {sfile}")
      else:
        ports.remove(p)

    return ports, lines

  def s_header(self, tsfile):
    lines = pstr(tsfile).read().splitlines()
    ln_pound = 0
    for i, ln in enumerate(lines):
      ln = ln.strip()
      if ln.startswith("#"):
        ln_pound = i
        break
      lines[i] = re.sub(r"\s+", " ", ln)
      if not lines[i].startswith("! Port"):
        continue
      lines[i] = rm_pattern("_sbump", lines[i], f=re.I)
      lines[i] = rm_pattern("_sball", lines[i], f=re.I)

    return lines, ln_pound

  def p_value(self, part, value_string=""):
    value_string = str(value_string)
    # part = '_'.join(part.split('_')[0:2])

    if "," in value_string:
      value, net, volt = re.search("(.*),(.*)=(.*)", value_string).groups()
    else:
      value = value_string
    card, comp = re.search("(.+)_(.+)", part).groups()  #'a_b_c' to (a_b,c)
    discrete = dict(zip("R L C Q D FB XW".split(), "R L C R C L R".split()))
    des = rm_pattern(r"\d+.*", comp)
    # if not des in discrete:
    # 	raise ValueError(f'component {part} value hard to find')
    # use model in .ibs if found
    hit = self.ibs.index[self.ibs["cardu"] == f"{card}.{comp}"].tolist()
    if len(hit):
      i = hit[0]
      file = self.ibs.at[i, "file"]
      subcir = self.ibs.at[i, "component"]
      if not file:
        return "", ""
      if re.search(r"\.ibs$", file):
        pass
      # touchstone
      elif re.search(r"\.s\d+p$", file):
        return f"S_{part}", "|".join([subcir, file])
      # sub circuit
      else:
        return f"X_{part}", ",".join([file, subcir])
    elif des in discrete:
      return f"{discrete[des]}_{part}", f"{value}"
    else:
      if value_string:  # '' in case of connector pairs, no print
        logme.warning(f"component {part} value hard to find")
      return "", ""

  def x_lines(self, row, ports):
    """subcircuit lines"""

    lines = ""
    if not bool(ports):
      return ports, lines
    card = row.card
    series = eval(row.series)
    if len(series):
      for u, value_string in series.items():
        if not any(u in x for x in ports):
          continue
        part = f"{card}_{u}"
        rdes, value = self.p_value(part, value_string)
        if not rdes.startswith("X_"):
          continue
        port = [x for x in ports if part in x]
        nodes = " ".join(port) if len(port) > 1 else f"{port[0]} 0"
        # if len(port) != 1:
        # 	logme.info(f'port not found for series :{part}')
        # 	continue
        finc, subcir = value.split(",")
        lines += f'* series {part}\n.inc "{finc}"\n{rdes} {nodes} "{subcir}"\n'
        #ports.remove(port[0])
        for p in port:
          ports.remove(p)
    shunts = eval(row.shunt)
    if len(shunts):
      for pin, value_string in shunts.items():
        part = f"{card}_{pin.split('.')[0]}"
        # if not re.search('(.+),(.+)=(.*)', value_string):	print('ho')
        (value, net, volt) = re.search("(.*),(.*)=(.*)", value_string).groups()
        rdes, value = self.p_value(part, value_string)
        port = [x for x in ports if part in x]
        nodes = " ".join(port) if len(port) > 1 else f"{port[0]} 0"
        # if len(port) != 1:
        # 	logme.info(f'port not found for shunt :{part}')
        # 	continue
        if not rdes.startswith("X_"):
          continue
        if rex_gnd.search(net):
          finc, subcir = value.split(",")
          lines += f"* pulldown {part}\n"
          lines += f'.inc "{finc}"\n'
          lines += f'{rdes} {nodes} "{subcir}"\n'
        else:
          vnet = f"{card}_{net}"
          finc, subcir = value.split(",")
          lines += f"* pullup {part}\n"
          lines += f'.inc "{finc}"\n'
          lines += f'{rdes} {nodes} "{subcir}"\n'
          lines += f"v_{vnet} {vnet} 0 '{vnet}'\n"
          self.param.update({vnet: volt})
        for p in port:
          ports.remove(p)

    return ports, lines

  def r_lines(self, row, ports):
    # series circuit elements, not just r*
    lines = ""
    if not bool(ports):
      return ports, lines
    card = row.card
    series = eval(row.series) if row.series else {}
    for u, value_string in series.items():
      if not any(u in x for x in ports):
        continue
      u = u.upper()
      part = f"{card}_{u}"
      rdes, value = self.p_value(part, value_string)
      if rdes.startswith("X_"):
        continue
      port = [x for x in ports if part in x]
      if False:  # old way setting ports on series components
        if len(port) != 1:
          logme.info(f"port not find for series :{part}")
        else:
          lines += f"* series {part}\n{rdes} {port[0]} 0 '{part}'\n"
          self.param.update({part: value})
          ports.remove(port[0])
      else:  # 2(more?) ports for series components
        if len(port) != 2:
          logme.warning(f"port not find for series :{part}")
        else:
          lines += f"* series {part}\n{rdes} {' '.join(port)} '{part}'\n"
          self.param.update({part: value})
          for p in port:
            ports.remove(p)

    return ports, lines

  def c_lines(self, row, ports):
    # shunt circuit elements, not just c*
    lines = ""
    card = row.card
    shunts = eval(row.shunt) if row.shunt else {}
    for pin, value_string in shunts.items():
      part = f"{card}_{pin.split('.')[0]}"
      (value, net, volt) = re.search("(.*),(.*)=(.*)", value_string).groups()
      rdes, value = self.p_value(part, value_string)
      if rdes.startswith("X_"):
        continue
      port = [x for x in ports if part in x]
      if len(port) != 1:
        logme.warning(f"port not find for series :{part}")
        continue
      if rex_gnd.search(net):
        lines += f"* pulldown {part}\n{rdes} {port[0]} 0 '{part}'\n"
      else:
        # vnet, volt = self.net_voltage(f'{card}.{net}')
        vnet = f"{card}_{net}"
        self.param.update({vnet: volt})
        lines += f"* pullup {part}\n{rdes} {port[0]} {vnet} '{part}'\n"
        lines += f"v_{vnet} {vnet} 0 '{vnet}'\n"
      self.param.update({part: value})
      ports.remove(port[0])

    return ports, lines

  def j_lines(self, jatuple, s=""):
    # board connectoin R=1e-5 if no s model
    USE_SMODEL = True
    # no connectors:
    joint, attached = jatuple
    if not ("".join(attached)):
      return s, ""
    joint_flat = list(itertools.chain.from_iterable(joint))
    paired = [x.replace(".", "_") for x in joint_flat]
    disjoint = [x.replace(".", "_") for x in attached if x not in joint_flat]

    lines = ""
    # paired connector:
    if bool(paired):
      for x, y in zip(paired[0::2], paired[1::2]):
        if USE_SMODEL is False:
          lines += f"R_{x}_{y} {x} {y} 1e-5\n"
          continue

        p1, sstring = self.p_value("_".join(x.split("_")[:-1]))
        if not p1:
          p1, sstring = self.p_value("_".join(y.split("_")[:-1]))
        if not p1:
          lines += f"R_{x}_{y} {x} {y} 1e-5\n"
          continue

        sn = 1 + len(self.smodel["file"])
        number, sfile = sstring.split("|")
        nports = int(sfile.split(".")[-1][1:-1])
        ports = [f"ns{100*self.smodel['count']+x:03d}" for x in range(nports)]
        if number:
          i = [int(e) - 1 for e in re.findall(r"\D*(\d+)\D*", number)]
          ports[i[0]], ports[i[1]] = (x, y)
        else:  # take first 2 nodes as default connection?
          print(f"?? port number unknown connecting {x}/{y} by {sfile}?")
          pass  # no. rather no connection
        if sfile not in self.smodel["file"]:
          self.smodel["file"].append(sfile)
          lines += f'* touchstone {os.path.basename(sfile)}\n.model sModel{sn} S TSTONEFILE="{sfile}"\n'
          lines += f"+ INTERPOLATION=LINEAR INTDATTYP=MA HIGHPASS=10 LOWPASS=10 convolution=0 enforce_passivity=0 Noisemodel=External\n"
        sn = 1 + self.smodel["file"].index(sfile)
        lines += f"S_{x}_{y} {' '.join(ports)} " + f'FQMODEL="sModel{sn}"\n'
        self.smodel["count"] += 1
    # lone connector pin as RX:
    if bool(disjoint):
      lines += "\n".join([f"C_{node} {node} 0 'RxCLoad'" for node in disjoint])
    return s, lines

  def b_lines(self, tx, iopin):
    # buffer circuit elements
    if not bool(tx):
      return "", "", ""
    if not tx in iopin:
      logme.info(f"driver {tx} not in listed inout- {','.join(iopin)}")
    else:  # move tx to first place in io list
      iopin.remove(tx)
      iopin.insert(0, tx)

    # iopin.remove(txpin)
    cir, cmd = "", ".tran 10p '3.25/bps'\n"
    probes = []
    n_drive_strength = 1
    for i, io in enumerate(iopin):
      i_o = io.replace(".", "_")
      bfile, bcomp, bpin, bmodels, btype = self._binfo(io, bool(io == tx) )
      bmodel = bmodels[0]
      cir += f"*** io: {io} ***\n"
      if btype == "RX":
        cir += f"c_{i_o} {i_o} 0 'RxCLoad'\n"
        probes.append(i_o)
        continue
      if btype == "CAP":
        cir += f"c_{i_o} {i_o} 0 c{i_o}\n"
        probes.append(i_o)
        self.param.update({f"c{i_o}": bcomp})
        continue
      if bmodel.upper() == "NC":
        cir += f"c_{i_o} {i_o} 0 'RxCLoad'\n"
        probes.append(i_o)
        continue

      b_node, b_mode = self._bstr(btype, io, i * 10, bool(io == tx))
      cir += f"{b_node}\n"
      if bool(io == tx):
        n_drive_strength = len(bmodels)
        buffer_selector = "{" + ",".join([f'"{x}"' for x in bmodels]) + "}"
        self.param.update({"TxBuffer": buffer_selector})
        cir += f"+ file=\"{bfile}\"\n+ model='TxBuffer[iTxBuffer]'\n"
        cir += f"+ typ='Corner[iCorner]'\n+ {b_mode}\n"
        cir += f"+ comp_name={bcomp} pin_name={bpin}\n"
        cir += f"+ pkg_selector=2 die_side_node_provided=true\n"
        cir += f"+ use_eyesrc_parameters=true TRISE=2e-12 TFALL=2e-12 UI='1/bps' bitlist=#0101\n"
      else:
        cir += f'+ file="{bfile}"\n+ model="{bmodel}"\n'
        cir += f"+ typ='Corner[iCorner]'\n+ {b_mode}\n"
        cir += f"+ comp_name={bcomp} pin_name={bpin}\n"
        cir += f"+ pkg_selector=2 die_side_node_provided=true\n"
        probes.append(i_o)
    # cmd += '\n'.join([f"*.tran 10p '3/bps' sweep {x} lin0 0 1" for x in 'iTxBuffer iCorner iRxLoad'.split()])
    cmd += f"*.tran 10p '3.25/bps' sweep iTxBuffer lin {n_drive_strength} 0 {n_drive_strength-1}\n"
    cmd += f"*.tran 10p '3.25/bps' sweep iCorner 3 0 2\n"
    cmd += f"*.tran 10p '3.25/bps' sweep RxCLoad POI 5 1e-12 2e-12 5e-12 10e-12 20e-12\n"

    cmd += "\n\n.probe tran" + "".join([f" v({x.replace('.','_')})" for x in probes]) + "\n\n.end\n"
    return cir, cmd


class clocks:
  
  cols_setup_table = {
      "Board Database": "design stackup file simpath netlist component topology tool nondns dns xpin",
      "Board Instance": "card design exnet",
      "Board Connection": "Aboard Aconn Bboard Bconn model",
      "Board NetGroup": "group speed regex design refdes",
      "Multiplexer": "partno type position connection",
  }
  cols_netlist_page = "net voltage pins"
  cols_component_page = "refdes partno pin_number pin_name net_name voltage"
  cols_xnets_page = "simulate grp speed xnet loads io pins nets ground series shunt dns"
  cols_switch_page = "design refdes partno type position use connection nets"
  cols_ibis_page = "design refdes partno nickname component file path note"

  def __init__(self, proj="proj_settings.json"):
    self.prj = None  # poject json
    self.gxls = None  # gxls anchor
    self.tables = {}  # mainly the setup page

    if proj and pstr(proj).isfile:
      with open(proj) as json_file:
        try:
          self.prj = self._json(json_file) or None
        except json.JSONDecodeError:
          print("clocks.__init__(): json file error")
          return None
    if self.prj is None:
      return None
    if "gsheet_url" not in self.prj:
      print("clocks: need gsheet url in json file!")
      return None

    ck_path = pstr(self.prj["prj_folder"] + "/Clocks")
    ck_path.mkdir()
    if not ck_path.isdir:
      print(f"clocks: cannot create folder {ck_path}!")
      return None
    logme(ck_path + "/clocks.log")
    ck_xlsx = os.path.basename(self.prj["prj_folder"]) + "_clocks.xlsx"
    cache_xls = f"{ck_path}/{ck_xlsx}" if self.prj.get('multiplex',None) else None
    self.gxls = gservices.gsheet(url=self.prj["gsheet_url"], xls=cache_xls)

    self._setup()

  def _json(self, json_file):
    """do some basic check to json files"""
    errors = {}

    def check_key(key, value):
      # valid keys
      if any(re.match(r"[a-zA-Z]", k[0]) is None for k in key.split(".")):
        return
      if isinstance(value,int):
        return
      
      value = value.replace("\\", "/")
      # https link
      if value.startswith("http"):
        if pstr(value).isurl is False:
          errors[key] = f"invalid url - {value}"
      # drive letters
      elif re.search("^[A-Za-z]:/", value):
        match = re.match(r"^([^\.]+\.\S+)\s*(.*)$", value)
        if match:
          path_to_file = match.group(1)
          if not pstr(path_to_file).isfile:
            errors[key] = f"invalid file - {value}"
        else:
          if not pstr(value).isdir:
            errors[key] = f"invalid directory - {value}"

    def iterate_json_keys(data, parent_key=""):
      if isinstance(data, dict):
        for k, v in data.items():
          new_key = f"{parent_key}.{k}" if parent_key else k
          if isinstance(v, (dict, list)):
            iterate_json_keys(v, new_key)
          else:
            check_key(new_key, v)  # Or do whatever you need with the key
      elif isinstance(data, list):
        for i, item in enumerate(data):
          new_key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
          if isinstance(item, (dict, list)):
            iterate_json_keys(item, new_key)
          else:
            check_key(new_key, item)  # Or do whatever you need with the key
      else:
        print(parent_key)

    data = json.load(json_file)
    if "spdlinks" not in data:
      if "powersi" in data:
        data["spdlinks"] = data["powersi"]
    if "spdlinks" not in data:
      errors["spdlinks"] = "need spdlinks or powersi to genearte spd file"
    if "allegro" not in data:
      errors["allegro"] = "need allegro executable to extract logic and etc"
    if "amm_library" not in data:
      errors["amm_library"] = "need amm_library file to run spd tcl"

    iterate_json_keys(data)
    if len(errors):
      for k, v in errors.items():
        print(f"{k}: {v}")
      return False

    return data

  def _setup(self, tab_prompt={}, tab_rows="all"):
    global rex_gnd

    if self.gxls is None:
      return False

    tabs= self.gxls.table("setup", skip_blank_col=0)
    if len(tabs) < 1:
      print("clocks: error reading setup page")
      return False
    
    if "Board Database" not in tabs:
      print("board database not find, existing...")
      return None

    # read multiplex csv, or gsheet, if any
    if "multiplex" in self.prj and "position" in self.prj["multiplex"]:
      where = self.prj["multiplex"]["position"].strip()
      if re.search(r"\.csv$", where, re.I) and pstr(where).isfile:
        tabs["Multiplexer"] = pd.read_csv(where, dtype=str)
      elif z := re.search(r"^http.*#gid=(\d+)", where):
        lib_gxls = gservices.gsheet(where)
        page = {f"{v}": k for (k, v) in lib_gxls.pages().items()}[z.group(1)]
        tabs["Multiplexer"] = lib_gxls.df(page)

    # check requried columns are there
    columns_there = True

    for t, df in tabs.items():
      cols = self.cols_setup_table[t].split()
      if not all(x in df.columns for x in cols):
        print(f"?? setup table `{t}` should have cols: ({','.join(cols)})")
        columns_there = False
    if not columns_there:
      return None

    if "Board NetGroup" in tabs:
      gnd_rex = []
      for ln in tabs["Board NetGroup"].itertuples():
        if re.search(r"dc\s*=\s*0", ln.speed.strip(), re.I):
          gnd_rex.append(ln.regex)
      if len(gnd_rex):
        rex_gnd = re.compile(rf"{'|'.join(gnd_rex)}", re.I)

    # create folders, update link to folders for the board database table
    df= tabs["Board Database"]
    df= df[df['simpath'].str.strip() != '']
    if df.duplicated(subset='design').any():
      print('duplicated designs found, keeping first design only')
      df = df.drop_duplicates(subset='design',keep = 'first')
    col_simpath = ord("A") + df.columns.get_loc("simpath")
    for row in filter(row_active, df.itertuples()):
      new_simpath = folders.setup(row, self.prj, tag='ck')
      if new_simpath:
        cell = f"{chr(col_simpath)}{row.Index+1}"
        self.gxls.set("setup", cell, new_simpath)
        readback  = self.gxls.cells(f"setup!{cell}")  # readback
        if new_simpath == readback[0][0]:
          tabs["Board Database"].at[row.Index, "simpath"] = new_simpath
    
    # check missing board if any
    missing_dsn = {}
    for ln in filter(row_active, tabs["Board Database"].itertuples()):
      if ln.simpath and ln.file:
        f = pstr(ln.simpath) + ln.file
        if not f.isfile:
          missing_dsn[str(f)] = True
    if len(missing_dsn):
      print(f"boards missing: {','.join(x for x in missing_dsn)}")
      return False
    # no nans, but blanks
    for t, df in tabs.items():
      df.fillna("", inplace=True)

    # make the tables pd.dataframe so later dont' have to check it is None
    for k in self.cols_setup_table:
      if k not in tabs or tabs[k] is None:
        tabs[k] = pd.DataFrame()

    self.tables = tabs

    for k, prompt in tab_prompt.items():
      for x in k.split():
        t = f"Board {x}"
        if self.tables[t].empty:
          print(f"{prompt}: aborted due to empty `{t}` table")
          return False

    return True

  def _pair_connectors(self, pairby="pin_name"):
    """pair connection pins by pin_name or pin_number"""
    print(f"clocks._pair_connectors():")

    # hidden rows in Board Instance table do not participate in pairing
    df_crd = self.tables["Board Instance"][self.tables["Board Instance"]["hide"] == False]
    b = dict(zip(df_crd["card"], df_crd["design"]))

    df_conn = self.tables["Board Connection"][self.tables["Board Connection"]["hide"] == False]
    i_missing_cards = []
    for r in df_conn.itertuples():
      cards = [r.Aboard, r.Bboard]
      if any(x not in b for x in cards):
        missing_cards = [x for x in cards if x not in b]
        logme.warning(f"_pair_connectors(): cards {missing_cards} not find in Board Instance")
        i_missing_cards.append(r.Index)
    if i_missing_cards:
      df_conn = df_conn.copy()
      df_conn.drop(i_missing_cards, inplace=True)
    if len(df_conn) < 1:
      logme.warning(f"_pair_connectors(): board connection table empty")
      return

    # seperate df_conn to comp only (df_cmp) and pin_specific only (df_pin)
    pin_cell = re.compile(r"^[A-Z0-9]+\.[A-Z0-9]+$", re.I)
    pin_row = lambda r: bool(pin_cell.match(r['Aconn'])) and bool(pin_cell.match(r['Aconn']))
    df_pin = df_conn[df_conn.apply(pin_row, axis=1)]  # Rows where both columns match the pattern
    df_cmp = df_conn.copy()
    idx = df_cmp.index.isin(df_pin.index)
    if any(idx):
      df_cmp.loc[idx, 'Aconn'] = df_cmp.loc[idx, 'Aconn'].str.rpartition('.')[0]
      df_cmp.loc[idx, 'Bconn'] = df_cmp.loc[idx, 'Bconn'].str.rpartition('.')[0]
      df_cmp = df_cmp.drop_duplicates()

    # valid design connector combinations
    connectors = {}
    for r in df_conn.itertuples():
      b1, b2 = (b[r.Aboard], b[r.Bboard])
      connectors[b1] = connectors.get(b1, []) + [r.Aconn]
      connectors[b2] = connectors.get(b2, []) + [r.Bconn]
    connectors = {k: v for (k, v) in connectors.items() if "".join(v).strip()}

    # read deigns metioned in conenction table
    df_dsn = self.tables["Board Database"]
    df_dsn = df_dsn[df_dsn["design"].isin(connectors)]

    bom = {}
    for dsn in df_dsn.itertuples():
      if dsn.component in self.gxls.pages():
        print(f"  reading components {dsn.component}")
        c = self.gxls.scan(dsn.component, "visible=all", cached=True)
        zipped = zip(c["refdes"], c["pin_number"], c["pin_name"])
        bom[dsn.design] = {f"{x}.{y}": f"{x}.{z}" for (x, y, z) in zipped}
      else:
        logme.warning(f"{dsn.simpath}/{dsn.file} not loaded, run parse_xnet() first")
    for k, v in filter(lambda kv: len(kv[1]) < 1, bom.items()):
      print(f"bom for {k} is empty")

    # group pins by components
    def rex_pins(bom, rex):
      numbers = {k: v for (k, v) in bom.items() if rex.match(k)}
      components = {}
      for number, name in numbers.items():
        refdes = number.split(".")[0]
        components.setdefault(refdes, {}).update({number: name})
      return components

    # return commmon (ending is same) portion of 2 dictionaries
    def common_key(d1, d2, ending="."):  #
      e1 = {k.split(ending)[-1]: k for k in d1}
      e2 = {k.split(ending)[-1]: k for k in d2}
      e = sorted(e1.keys() & e2.keys())
      return ({e1[k]: d1[e1[k]] for k in e}, {e2[k]: d2[e2[k]] for k in e})

    # connector pin pairs in ppair
    ppair = {"pin1": [], "pin2": [], "model": []}
    logme.info("  pairing conenctors ...")
    for r in df_cmp.itertuples(index=False):
      # Aboard	Aconn	Bboard	Bconn
      by = r.pairing if getattr(r, "pairing", None) else pairby
      bom1 = bom.get(b[r.Aboard], {})
      bom2 = bom.get(b[r.Bboard], {})
      if len(bom1) < 1 or len(bom2) < 1:
        continue
      j1 = r.Aconn.split(".")[0]
      conn1 = rex_pins(bom1, re.compile(rf"{j1}\.", re.I))
      if len(conn1) < 1:
        logme.warning(f"connector not found: {r.Aboard}.{j1}")
        continue
      j2 = r.Bconn.split(".")[0]
      conn2 = rex_pins(bom2, re.compile(rf"{j2}\.", re.I))
      if len(conn2) < 1:
        logme.warning(f"connector not found: {r.Bboard}.{j2}")
        continue
      if len(conn1) > 1 or len(conn2) > 1:  # only same refdes if multiple connectors
        conn1, conn2 = common_key(conn1, conn2)
      if len(conn1) < 1 or len(conn2) < 1:
        logme.warning(f"pairing error: {r.Aboard}.{j1} - {r.Bboard}.{j2}")
        continue

      # process for each connector refdes
      for (j1, number1), (j2, number2) in zip(conn1.items(), conn2.items()):
        if by == "pin_number":
          num1, num2 = common_key(number1, number2)
        elif by == "pin_name":
          name1 = {v: k for (k, v) in number1.items()}
          name2 = {v: k for (k, v) in number2.items()}
          nym1, nym2 = common_key(name1, name2)
          num1 = {v: k for (k, v) in nym1.items()}
          num2 = {v: k for (k, v) in nym2.items()}
        elif by == "odd_even":
          oxe = lambda x: x + 1 if x % 2 else x - 1
          flipped = {k: f"{j2}.{oxe(int(k.split('.')[-1]))}" for k in number2}
          num1, flip = common_key(number1, flipped)
          num2 = {v: k for (k, v) in flip.items()}
        else:
          logme.info("pair(): method of mapping not defined")
          continue
        if len(num1) < 1 or len(num2) < 1:
          logme.warning(f"pairing error: {r.Aboard}.{j1} - {r.Bboard}.{j2}")
          continue

        ppair["pin1"] += [f"{r.Aboard}.{x}" for x in num1]
        ppair["pin2"] += [f"{r.Bboard}.{x}" for x in num2]
        ppair["model"] += [r.model for _ in range(len(num1))]

    # overwrite ppair if pins explicitly defined( e.g. J25001.3)
    for r in df_pin.itertuples():
      b1, j1, b2, j2 = r.Aboard, r.Aconn, r.Bboard, r.Bconn
      if not (b1 in b and b2 in b and b[b2] in bom and b[b1] in bom):
        continue
      if j1 in bom[b[b1]] and j2 in bom[b[b2]]:
        pin1 = f"{b1}.{j1}"
        pin2 = f"{b2}.{j2}"
        i1 = ppair["pin1"].index(pin1) if pin1 in ppair["pin1"] else None
        i2 = ppair["pin2"].index(pin2) if pin2 in ppair["pin2"] else None
        if i1 is not None:
          ppair["pin2"][i1] = pin2
          ppair["model"][i1] = r.model
        if i2 is not None:
          ppair["pin1"][i2] = pin1
          ppair["model"][i2] = r.model
        if i1 is None and i2 is None:
          ppair["pin1"] += [pin1]
          ppair["pin2"] += [pin2]
          ppair["model"] += [r.model]
        if i1 is not None and i2 is not None and i1 != i2:
          del ppair["pin1"][i2]
          del ppair["pin2"][i2]
          del ppair["model"][i2]

    return ppair if len(ppair["pin1"]) else {}

  def _g2g_prts(self, grounds, cpn):
    head, tail = (grounds.split(":", 1) + [""])[:2]
    gnets = head.split(",")
    gparts = {k: True for k in tail.split(",")} if tail else {}

    dsc_cpn = {k: v for (k, v) in cpn.items() if v["type"] == "dsc" and v["thru"] is not None and v["net"] in gnets}
    for k, v in dsc_cpn.items():
      part = k.split(".")[0]
      if part in gparts:
        continue
      j = part + "." + v["thru"]
      net_thru = dsc_cpn[j]["net"] if j in dsc_cpn else None
      if net_thru is None:
        continue
      if v["net"] != net_thru:
        gparts.update({part: True})
    parts = ",".join(gparts.keys())
    nets = ",".join(gnets)
    return f"{nets}:{parts}" if parts else nets

  def _grp_nets(self, df, xnets, fuzzy):
    d = {e: [] for e in self.cols_xnets_page.split()}
    df = df.iloc[::-1]  # revserse row order
    nodf = df[df["group"] == "EXCLUDE"]
    nonet = {k: v for (k, v) in zip(nodf["regex"], nodf["refdes"]) if k.strip()}
    exc = re.compile("|".join(list(nonet.keys())), re.I) if len(nonet) else None
    is_clk_grp = lambda row: row.group not in ["POWER", "GROUND", "EXCLUDE"]
    for row in filter(is_clk_grp, df.itertuples()):
      prt = re.compile(rf"(,?{row.refdes}\.)") if bool(row.refdes) else None
      inc = re.compile(row.regex, re.I)
      fltred = xnets.regex(inc, exc, prt, exact=not fuzzy)
      for e, pinlst in fltred.items():
        if len(d["nets"]) and (e in d["nets"]):  # already in another group
          continue
        # xpins = xnets.xnet[e].split(',')
        pins = pinlst.split(",")
        k = xnets.kinds(pins)
        d["simulate"].append("TRUE")
        d["grp"].append(row.group)
        d["speed"].append(row.speed)
        d["xnet"].append(min(e.split(","), key=lambda s: (len(s), s)))
        d["loads"].append(len(k["io"]))
        d["io"].append(",".join(k["io"].keys()))
        # d['driver'].append( list(k['io'].keys())[0] )
        d["pins"].append(pinlst)
        d["nets"].append(e)
        # d['power'].append( ','.join(sorted(k['pwr'].keys())) )
        d["ground"].append(",".join(sorted(k["gnd"].keys())))
        d["series"].append(",".join(sorted(k["ser"].keys())))
        d["shunt"].append(str(k["par"]))
        d["dns"].append(",".join(k["dns"].keys()))
        if k["g2g"]:
          d["ground"][-1] += ":" + ",".join(k["g2g"].keys())
          # d['tp'].append(','.join(k['tp'] ) )
      d = {k: v[::-1] for (k, v) in d.items()}
    return pd.DataFrame(d)

  def _prt_nets(self, df, xnets, multiplexers, grpName="~mux"):
    d = {e: [] for e in self.cols_xnets_page.split()}
    df = df.iloc[::-1]
    nodf = df[df["group"] == "EXCLUDE"]
    nonet = {k: v for (k, v) in zip(nodf["regex"], nodf["refdes"]) if k.strip()}
    exc = re.compile("|".join(list(nonet.keys())), re.I) if len(nonet) else None
    inc = re.compile(r".+", re.I)
    for refdes in multiplexers:
      prt = re.compile(rf"(,?{refdes}\.)") if bool(refdes) else None
      fltred = xnets.regex(inc, exc, prt, exact=True)
      for e, pinlst in fltred.items():
        # for e in s:
        if len(d["nets"]) and (e in d["nets"]):  # already in another group
          continue
        pins = pinlst.split(",")
        k = xnets.kinds(pins)
        d["simulate"].append("TRUE")
        d["grp"].append(grpName)
        d["speed"].append("Hz=NAN")
        d["xnet"].append(min(e.split(","), key=lambda s: (len(s), s)))
        d["loads"].append(len(k["io"]))
        d["io"].append(",".join(k["io"].keys()))
        # d['driver'].append( list(k['io'].keys())[0] )
        d["pins"].append(pinlst)
        d["nets"].append(e)
        # d['power'].append( ','.join(sorted(k['pwr'].keys())) )
        d["ground"].append(",".join(sorted(k["gnd"].keys())))
        d["series"].append(",".join(sorted(k["ser"].keys())))
        d["shunt"].append(str(k["par"]))
        d["dns"].append(",".join(k["dns"].keys()))
        if k["g2g"]:
          d["ground"][-1] += ":" + ",".join(k["g2g"].keys())
        # d['tp'].append(','.join(k['tp'] ) )
    d = {k: v[::-1] for (k, v) in d.items()}
    return pd.DataFrame(d)

  def _yn_slash(self, df):
    """aske user confirmation for duplicate/missing master nets"""
    if df is None or df.empty:
      return True
    if "route" not in df.columns:
      return True

    tagged = {}  # {card.xnet: [rn,..]} or {int(rn):[]}
    byroute = df.groupby("route")
    for rn in list(byroute.groups.keys()):
      tagged.update({int(rn): []})
      df1 = byroute.get_group(rn)
      for row in df1.itertuples():
        if re.search(r"/(\d*)$", row.xnet) or len(df1) == 1:
          tagged.pop(int(rn))
          tag = f"{row.card}.{row.xnet}"
          tagged[tag] = tagged[tag] + [rn] if tag in tagged else [rn]
          break

    repeat, blank = [], []
    for k, v in tagged.items():
      if isinstance(k, int):
        blank.append(f"{k}")
      elif len(v) > 1:
        repeat.append("(" + " ".join(v) + ")")

    repeating, missing = ",".join(repeat), ",".join(blank)
    if repeating:
      print(rf"? duplicate master net in routes: {repeating}")
    if missing:
      print(rf"? missing master net in routes: {missing}")
    if repeating or missing:
      while True:
        yes = input("? continue(y/n):")
        if yes in list("yYnN"):
          break
      if yes in list("nN"):
        return False

    return True

  def _list_mux(self, df_tab="~switch"):
    """add card columns to ~swith table as usage of mux on individual card can vary
    in - switch datafrem or the ~switch page
    out - page ~swith with card column
    """
    if self.gxls is None:
      return  # not initiated
    if isinstance(df_tab, pd.DataFrame):
      switch = df_tab
    elif isinstance(df_tab, str) and df_tab in self.gxls.pages():
      switch = self.gxls.df(df_tab)
    else:
      print(f"list_mux(): {df_tab} not a df or page in gsheet")
      return

    if len(switch):
      if "card" in switch.columns:
        switch = switch.drop(columns=["card"])
        switch = switch.drop_duplicates()
      if "Board Instance" in self.tables:
        cards = self.tables["Board Instance"]
        switch.insert(loc=0, column="card", value=switch["design"])
        allcards = pd.DataFrame(columns=switch.columns)
        for row in cards.itertuples():
          onecard = switch[switch["design"] == row.design].copy()
          onecard["card"] = row.card
          allcards = pd.concat([allcards, onecard], ignore_index=True)
        self.gxls.update("~switch", allcards)
      else:
        self.gxls.update("~switch", switch)
    if isinstance(df_tab, str):
      print("hola clocks.list_mux()!")

  def parse_xnet(self, fuzzy_hit=True, redo=False):
    """list netlist, bom and xnets, netgroup can be loose if fuzzy_hit
    ~switch page created, 'card' column added if board instance table found
    """
    chk = {"Database NetGroup": "clocks.parse_xnet()"}
    if self._setup(chk) is False:
      return  # not initiate

    def _refer(x, y):
      return [f"{k}={y[v]}" for k, v in x.items() if v in y] if y else []

    df_dsn = self.tables["Board Database"]
    df_grp = self.tables["Board NetGroup"].copy()
    df_grp = df_grp[df_grp["hide"] == False].drop(columns=["hide"])

    # seperate cmc and true mux from mux table
    mux = self.tables.get("Multiplexer", {})
    if len(mux):
      cmc = mux[mux["type"] == "cmc"].copy()
      mux = mux[mux["type"] != "cmc"]
      cmc_dict = dict(zip(cmc["partno"], cmc["connection"]))
    # cmc will be as aditional xpin pairs for xnet dectection
    else:
      cmc_dict = {}
    switch = pd.DataFrame(columns=self.cols_switch_page.split())
    design = df_dsn["design"].to_list()
    if isinstance(redo, str):
      design = re.split(r"\s*,\s*", redo)
      redo = True
    dsn_visible = lambda row: (row.hide is False) and (row.design in design)

    for dsn in filter(dsn_visible, df_dsn.itertuples()):
      # if not 'qua' in dsn.design: continue
      simd = pstr(dsn.simpath)
      srcd = pstr(simd.path()).path() + "/Boards"
      file = simd + dsn.file
      src = pstr(srcd) + dsn.file
      if file.isfile and file.after(src):
        print(f"parse_xnet(): {file}")
      elif src.isfile:
        print(f"parse_xnet(): copying {file} from ../../Boards", end="")
        file.copyfrom(src)
        print("...copied" if file.isfle else "...failed")
      if not file.isfile:
        print(f"{src} skipped")
        continue

      parsed = pstr(file.file() + ".xnet")
      if parsed.after(file) and redo is False:
        continue
      ext = file.ext().lower()
      stk = self.gxls.text(dsn.stackup) if dsn.stackup else []
      lib = self.prj.get("amm_library", "")
      exe = self.prj[dsn.tool] if dsn.tool in self.prj else ""
      if re.match(r"^\.s\d+p$", ext, re.I):
        snp = touchstone(str(file))
        comp = component(snp.data, nondns=dsn.nondns, dns=dsn.dns)
        xpin = _refer(comp.gpn, cmc_dict) + [dsn.xpin]
        xnets = functionpin(snp.data["comppin"], comp.dns, ",".join(xpin))
      elif ext in [".brd", ".spd", ".tgz"]:
        lnk = spdlinks(self.prj["spdlinks"], app=exe, raw=file, amm=lib, stk=stk)
        if redo or lnk.translated is False:
          print(f"  translating {dsn.file} to spd")
          lnk.translate()
        if ext in [".brd"]:  # brd
          brd = allegro(brd=file, editor=self.prj["allegro"])
          comp = component(brd.cmp, nondns=dsn.nondns, dns=dsn.dns)
          xpin = _refer(comp.gpn, cmc_dict) + [dsn.xpin]
          xnets = functionpin(brd.fpn, comp.dns, ",".join(xpin))
        else:  # sigrity
          spd = sigrity(spd=lnk.spd)
          comp = component(spd.data, nondns=dsn.nondns, dns=dsn.dns)
          xpin = _refer(comp.gpn, cmc_dict) + [dsn.xpin]
          xnets = functionpin(spd.data["comppin"], comp.dns, ",".join(xpin))
      else:
        print(f"file not recogonized: {file}")
        continue

      dc = {}  # dc nets
      mask_dsn = df_grp["design"].apply(lambda x: x == "" or dsn.design in re.split(r"[,\s]+", x))
      d_grp = df_grp[mask_dsn]
      x = d_grp["speed"].apply(lambda x: re.match(r".*dc=.*", x)).values.nonzero()
      if x is not None:
        dc = {d_grp.iloc[i]["regex"]: re.search("dc=([^,]+)", d_grp.iloc[i]["speed"]).group(1) for i in x[0]}
      xnets.trace(dc=dc, inc_pwr=False)

      # netlist
      d = {e: None for e in self.cols_netlist_page.split()}
      d["net"] = [e for e in xnets.nets]
      d["pins"] = [",".join(v) for v in xnets.nets.values()]
      d["voltage"] = [xnets.vnet[e] if e in xnets.vnet else "" for e in xnets.nets]
      self.gxls.update(dsn.netlist, d, sort=["voltage"], hide=True)
      print(f"  nets hide in gsheet page `{dsn.netlist}`")
      # components
      multiplexers = {}
      d = {e: [] for e in self.cols_component_page.split()}
      for pin, p in xnets.cpn.items():
        refdes, pin_number = pin.split(".")
        gpn = comp.gpn[refdes] if refdes in comp.gpn else ""
        if gpn and len(mux) and gpn in mux["partno"].values:
          multiplexers.update({refdes: gpn})
        d["refdes"] += [refdes]
        d["partno"] += [gpn]
        d["pin_number"] += [pin_number]
        d["pin_name"] += [p["name"]]
        d["net_name"] += [p["net"]]
        d["voltage"] += [xnets.vnet[p["net"]] if p["net"] in xnets.vnet else ""]
      self.gxls.update(dsn.component, d, sort=["voltage"], hide=True)
      print(f"  components hide in gsheet page `{dsn.component}`")
      # switches
      for refdes, partno in multiplexers.items():
        df = mux[mux["partno"] == partno].copy()
        df = df.assign(use=df["position"])
        df = df.assign(refdes=refdes)
        df = df.assign(design=dsn.design)
        df = df.assign(nets="")
        for i, row in df.iterrows():
          pins = re.findall(r"\((\S+)\s+(\S+)\)", row["connection"])
          for p1, p2 in pins:
            pin1, pin2 = f"{refdes}.{p1}", f"{refdes}.{p2}"
            net1 = [k for (k, v) in xnets.nets.items() if pin1 in v]
            net2 = [k for (k, v) in xnets.nets.items() if pin2 in v]
            df.at[i, "nets"] += f"({net1[0] if net1 else ''} {net2[0] if net2 else ''})"
        switch = pd.concat([switch, df], ignore_index=True)

      # conenctors
      connectors = {}
      inst = self.tables.get("Board Instance", None)
      conn = self.tables.get("Board Connection", None)
      if fuzzy_hit and all(x is not None for x in (inst, conn)):
        df_card = inst[(inst["hide"] == False) & (inst["design"] == dsn.design)]
        cards = df_card["card"].to_list()
        df_conn = conn[conn["hide"] == False]
        for c in df_conn.itertuples():
          if c.Aboard in cards:
            connectors.update({c.Aconn.split(".")[0]: True})
          if c.Bboard in cards:
            connectors.update({c.Bconn.split(".")[0]: True})

      # toplogy
      df = self._grp_nets(d_grp, xnets, fuzzy_hit)
      df1 = self._prt_nets(d_grp, xnets, connectors, grpName="~con") if bool(connectors) else None
      df2 = self._prt_nets(d_grp, xnets, multiplexers, grpName="~mux") if bool(multiplexers) else None
      df = pd.concat([df, df1, df2])
      df = df.sort_values(by=["grp"])
      df.drop_duplicates(subset="pins", inplace=True, keep="first")
      df = df.reset_index(drop=True)
      # now do some sanity check 1)series have odd pins, 2)#loads>10
      for ln in df.itertuples():
        if ln.series:
          for x in ln.series.split(","):
            pins = re.findall(rf"{x}\.(\w+)", ln.pins)
            if len(pins) % 2:
              print(f"  ? series part {x} has {len(pins)} pin(s) in xnet {ln.xnet} ")
        if int(ln.loads) > 10:
          print(f"  ?? xnet {ln.xnet} at row {ln.Index}(+2) has large number({ln.loads}) of loads")
        if ln.ground and len(ln.ground.split(",")) > 1:
          df.at[ln.Index, "ground"] = self._g2g_prts(ln.ground, xnets.cpn)
      self.gxls.update(dsn.topology, df)
      parsed.write(f"{dsn.topology} at {time.time()}")
      print(f"  xnets shown in gsheet page {dsn.topology}")
    # ~switch page
    if len(switch):
      self._list_mux(switch)

    print("hola clocks.parse_xnet()!")

  def list_mux(self, page_in="~switch", page_out="switch"):
    print("place holder for mannual step 2. see:")
    print(
        "https://docs.google.com/document/d/1BTvaRC3c_gJUi9-sVTEFp-4wAWHW2jtR_uDU8vsHFFE/edit?resourcekey=0-xJz_s2OG2xdpDWneNeh1mg&tab=t.fj2mc5dqwfc1#heading=h.1xe6gbskfpi3"
    )
    if self._setup() is False:
      return  # not initiate
    if page_out and page_in and page_in in self.gxls.pages():
      yes = input(f"copy page '{page_in}' to '{page_out}'? Y for yes:")
      if yes == "Y":
        self.gxls.copy(page_in, page_out)
        # df = self.gxls.df(page_in)
        # self.gxls.update(page_out,df)
        print(f"'{page_in}' duplicated to '{page_out}'")
      print(f"make sure unused mux channels hidden or assigned 0 in use cell on page '{page_out}'")

  def connect_brd(self, mux_by="switch", conn_by="pin_name", trail_run=False):
    chk = {"Database Instance Connection": "clocks.connect_brd()"}
    if self._setup(chk) is False:
      return  # not initiated

    FUZZY_NETS = True  # this alows connecting nets not in regex

    use_mux = True if mux_by else False
    if mux_by and mux_by not in self.gxls.pages():
      print(f"clocks.connect_brd(): switch tab '{mux_by}' not found in gsheet, connecting w/o mux")
      use_mux = False

    design = self.tables["Board Database"]
    cards = self.tables["Board Instance"]
    cards_visible = cards[cards["hide"] == False]
    # filter switch table visible and pisitoin is same as use
    binding = None

    if use_mux and "multiplex" in self.prj:
      switch = self.gxls.df(mux_by, visible=True)
      switch = switch.loc[switch["design"].isin(cards_visible["design"].values)].reset_index(drop=True)
      switch = switch[switch["position"] == switch["use"]]
      switch["channels"] = 0  # append channels column
      if "binding" in self.prj["multiplex"]:
        how = self.prj["multiplex"]["binding"]
        if isinstance(how, str):
          if how in "none left right left_right".split():
            binding = None if how == "none" else how
          else:
            print(f"clocks.connect_brd(): mux binding {self.prj['multiplex']['binding']} not valid")
        elif isinstance(how, list):
          binding = eval(how)
    else:
      if use_mux:
        print("clocks.connect_brd(): no 'multiplex' entry in json file, connecting w/o mux")
      switch = pd.DataFrame()
    # mux may use fewer channel on card than on board design.
    # number of channels will be supperset of all cards, or appreared on the design
    channels = {}  # {brd.refdes: [pos1, pos2]}
    switch_pins = {}  # {card.refdes.pin: #channels}
    for row in switch.itertuples():
      brdU, pos = f"{row.design}.{row.refdes}", row.position
      channels[brdU] = list(set(channels[brdU] + [pos])) if brdU in channels else [pos]

    for row in switch.itertuples():
      crdU = f"{row.card}.{row.refdes}"
      # nch = len(channels[f'{c2b[row.card]}.{row.refdes}'])
      nch = len(channels[f"{row.design}.{row.refdes}"])
      switch.at[row.Index, "channels"] = nch
      for pin_number in filter(None, re.split(r"[\(\s\)]+", row.connection)):
        id = f"{crdU}.{pin_number}"
        switch_pins.update({id: nch})
    # seperate switch to jumper and mux
    jumper_pins = {k: "jumper" for (k, v) in switch_pins.items() if v == 1}
    x = [k for k in jumper_pins]
    jumper = dict(zip(x[::2], x[1::2]))  # {card1.mux1.pin1: card1.mux1.pin2}
    mux_pins = {k: "mux" for (k, v) in switch_pins.items() if v > 1}

    # jumpers merged to conn as it provides 1:1 connection just like connectors
    pin1pin2 = self._pair_connectors(pairby=conn_by)
    conn = dict(zip(pin1pin2["pin1"], pin1pin2["pin2"])) if len(pin1pin2) else {}  # {card1.con1.pin1: card2.con2.pin2}
    conn_pins = {}
    for k, v in conn.items():
      conn_pins.update({k: "conn", v: "conn"})
    conn_pins.update(jumper_pins)
    conn.update(jumper)

    # cerate lists of xnets for all cards called routes,
    # 'card' column added to distinguish same design
    routes = None
    dsn_read = {}  # {brd: topo_of_brd}, save gxls read effort in case of multiple cards w/ same design
    for crd in filter(row_active, cards.itertuples()):
      dsn_name = crd.design
      if dsn_name not in design["design"].values:
        print(f"clocks.connect_brd(): design {dsn_name} not found for board instance {crd.card}")
        continue
      dsn = design[design["design"] == dsn_name]
      if len(dsn) < 1:
        print(f"clocks.connect_brd(): design {dsn_name} not found for board instance {crd.card}")
        continue
      tab_xnet = dsn.iloc[0]["topology"]
      if not tab_xnet in self.gxls.pages():
        print(f"clocks.connect_brd(): page `{tab_xnet}` not found for card `{crd.card}` of design `{dsn_name}`")
        continue
      if dsn_name in dsn_read:
        df = dsn_read[dsn_name]
      else:
        df = self.gxls.df(tab_xnet, visible=True)
        df = df[(df["grp"] != "") & (df["xnet"] != "") & (df["xnet"] != "...unused...")]
        if not FUZZY_NETS:
          df = df[df["grp"] != "~con"]
        df = df.assign(card=crd.card)
        dsn_read.update({dsn_name: df})
      routes = pd.concat([routes, df], ignore_index=True)
    # 'alone': False if xnet goes to conenctor or mux
    routes["alone"] = "yes"
    routes["joint"] = ""  # (mux_pin1, mux_pin2), ... [con1_pin, con2_pin]
    # create the xnet tuple: edge between node(=row.Index) of xnet and nodes of mux/conn ids
    xnt_nodes, xnt_tuple = {}, []
    mux_nodes, con_nodes = {}, {}
    for row in routes.itertuples():
      n = row.Index
      pins = row.pins.split(",")
      ids = [f"{row.card}.{p}" for p in pins]  # card.refdes.pin
      d1 = dict((i, n) for i in ids if i in mux_pins)
      if d1:
        mux_nodes.update(d1)
        xnt_tuple += [(n, i) for (i, n) in d1.items()]
      d2 = dict((i, n) for i in ids if i in conn_pins)
      if d2:
        con_nodes.update(d2)
        xnt_tuple += [(n, i) for (i, n) in d2.items()]
      if d1 or d2:
        xnt_nodes.update({n: (row.grp, row.card, row.pins)})
        routes.at[n, "alone"] = "no"

    # create 1:1 edge tuple. limiting edge numbers by connector pins in xnets
    # mate = conn | {v: k for (k, v) in conn.items()}
    mate = {v: k for (k, v) in conn.items()}
    mate.update(conn)

    con_tuple = {}
    for pin in con_nodes:
      k = ",".join(sorted([pin, mate[pin]]))
      con_tuple[k] = (pin, mate[pin])
    con_tuple = list(con_tuple.values())

    # mux edge tuples will be created later in muxGraph by mux_dict
    muxGraph = multiplex(xnt_nodes, xnt_tuple + con_tuple, switch)
    n_combs = muxGraph.add_mux(mux_nodes, bind=binding)
    if trail_run:
      n_combs = min(n_combs, 1 * 2 * 64)  # 1*n_threads*chunck

    chunck = min(n_combs, 64 if trail_run else 2048)
    n_chuncks = (n_combs + chunck - 1) // chunck  # ceiling number, total jobs to run
    n_threads = min(2 if trail_run else 48, multiprocessing.cpu_count() // 2, n_chuncks)  # size of job queque
    next_iter = n_threads * chunck

    print(f"iterating {n_combs} combinations ({n_chuncks} chuncks of size {chunck}) in {n_threads} threads")

    # copies of multiplex class for parallel processing
    jobs = [copy.deepcopy(muxGraph) for _ in range(n_threads)]  # job queque

    # create signal so user can interrupt with control+C
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = multiprocessing.Pool(n_threads)
    signal.signal(signal.SIGINT, original_sigint_handler)

    walked = {}  # {(0, 61):((mb.J1.E6,soc.J1.E6),0)}
    try:
      # launch initial jobs by apply_async
      time_start = time.time()
      results = {i: pool.apply_async(worker, args=(jobs[i], i * chunck, chunck)) for i in range(n_threads)}
      collected = {i: False for i in range(n_threads)}  # true if job i has result collected
      collecting = n_chuncks  # count down to 0 then all results collected
      while collecting:
        for j, res in results.items():
          if res.ready() and not collected[j]:
            walked.update(res.get())
            time_spent = seconds_to_hms(time.time() - time_start)
            print(
                f"collecting chunck {collecting} from thread {j}, time elapsed: {time_spent}, total {len(walked)} paths found"
            )
            collected[j] = True
            collecting -= 1
            if next_iter < n_combs:
              results[j] = pool.apply_async(worker, args=(jobs[j], next_iter, chunck))
              collected[j] = False
              next_iter += chunck
    except KeyboardInterrupt:
      pool.terminate()
      print("clocks.connect_brd(): interupted")
      return
    # straight = routes.index[(routes['alone'] == 'yes')
    # 											& (routes['grp'] != '~mux')
    # 											& (~routes['speed'].str.endswith('NAN'))].tolist()
    straight = routes.loc[
        (routes["alone"] == "yes") & (routes["grp"] != "~mux") & (~routes["speed"].str.endswith("NAN"))
    ].index.tolist()
    routes["route"] = routes.pop("alone")
    # routes.drop(columns=['alone'],inplace=True)
    routes["attached"] = ""
    diffstr = lambda s1, s2: set(s1.split(",")) - set(s2.split(","))
    routes.loc[straight, "route"] = [x + len(walked) for x in range(len(straight))]
    single = routes.loc[straight].reset_index(drop=True)
    joined = None
    for rn, (row_numbers, vertices_ch) in enumerate(walked.items()):
      idx = list(row_numbers)
      idy = vertices_ch[:-1]
      on_mux, others = muxGraph.get_on_mux(idy, vertices_ch[-1])
      on_conn = []
      for y in others:
        on_conn += [[y, mate[y]]] if ((y in mate) and (y not in str(on_conn))) else []
      terminated = False  # just assume a route has no tx/rx to filter out nets among mux
      for i in idx:
        labeled_pins = [f"{routes['card'][i]}.{e}" for e in routes["pins"][i].split(",")]
        attached_pins = [y for y in idy if y in labeled_pins]
        io_pins = [routes.at[i, "card"] + x for x in routes.at[i, "io"].split(",")]
        if set(io_pins) != set(attached_pins):
          terminated = True
        routes.at[i, "attached"] = ",".join(attached_pins)
        routes.at[i, "route"] = rn
        routes.at[i, "joint"] = str(on_conn + on_mux)
        if on_conn:  # move conn pins to io cell
          missing_io = []
          for e in diffstr(routes.at[i, "pins"], routes.at[i, "io"]):
            crdpin = routes.at[i, "card"] + "." + e
            if any(crdpin in sublst for sublst in on_conn):
              missing_io.append(e)
          if len(missing_io):
            routes.at[i, "io"] += ",".join([""] + missing_io)
            routes.at[i, "loads"] = str(int(routes.at[i, "loads"]) + len(missing_io))
      if terminated:  # collect routes that are temrinated
        joined = pd.concat([joined, routes.loc[idx]], ignore_index=True)
    routed = pd.concat([joined, single], ignore_index=True)
    routed = routed[routed["route"].apply(lambda x: isinstance(x, int))]
    routed = routed.sort_values(by=["speed", "route", "card"])

    # routed = routed.sort_values(by=['route', 'xnet', 'card'])
    # self.gxls.update('routed',routed)
    # now append /(\d*) to master nets for cell in xnet columns
    exnet_cards = cards.loc[cards["exnet"].str.lower() == "true", "card"].to_list()
    master_nets = {}
    for ln in routed.itertuples():
      if ln.card in exnet_cards:
        master_nets.update({ln.xnet: (master_nets[ln.xnet] + [ln.Index] if ln.xnet in master_nets else [ln.Index])})
    for _, lines in master_nets.items():
      for i, n in enumerate(lines):
        routed.at[n, "xnet"] += f"/{i}" if len(lines) > 1 else "/"
    routed["simulate"] = routed["io"].apply(lambda x: "TRUE" if x else "FALSE")

    # filter out routes hitting nopower but not any signal regex
    byroute = routed.groupby("route")
    topology = None
    for rn in list(byroute.groups.keys()):
      df = byroute.get_group(rn)
      if len(df["io"].str.cat(sep=",").split(",")) < 2:
        df.loc[:, "simulate"] = "FALSE"
      nan_index = df.index[df["speed"].str.contains("=NAN")].tolist()
      num_index = [i for i in df.index if i not in nan_index]
      if num_index:
        df.loc[:, "speed"] = df.loc[num_index[0], "speed"]
        topology = pd.concat([topology, df])

    # renumber route numbers, zebra color rows
    if not (topology is None or topology.empty):
      rowcolor = {}
      for i, rn in enumerate(topology["route"].values):
        rows = rowcolor[rn] + [i] if (bool(rowcolor) and rn in rowcolor) else [i]
        rowcolor.update({rn: rows})
      for i, rn in enumerate(rowcolor):
        topology.loc[topology["route"] == rn, "route"] = i
      self.gxls.update("~topology", topology, zebra=rowcolor)

    print("hola clocks.connect_brd()!")

  def assign_ibs(self, worksheet="~topology", log=False):
    """create the ibis page assuming catalog in the parts page
    wirte back to libary page if new devices find
    in - page 'topology', link to partdb( a gsheet, a page, or None)
    out- page 'ibis', 'topology' with updated driver info
    """
    chk = {"Database Instance": "clocks.assign_ibs()"}
    if self._setup(chk) is False:
      return

    # toplogy topo
    if worksheet not in self.gxls.pages():
      print(f"assign_ibs(): page `{worksheet}` not found in gsheet")
      return
    topo = self.gxls.df(worksheet)
    if topo.empty:
      print(f"assign_ibs(): page `{worksheet}` is empty")
      return

    today = datetime.today().strftime("%Y-%m-%d")

    log_sheet, lib_gxls = "", None  # write back page and url

    _u = lambda x: [e.split(".")[0] for e in x.split(",")]
    _g = lambda x: re.sub(r'^G','', x)

    # libary data  lib
    lib = None
    catalog = self.prj.get("ibs_library", "")
    if catalog and catalog.startswith("https:"):
      log_sheet, lib_gxls = "~log", gservices.gsheet(catalog)
      print("clocks.assign_ibs(): loading ibis catalag from gsheet...", end="")
      for page in lib_gxls.pages():
        if page != log_sheet:
          lib = pd.concat([lib, lib_gxls.df(page)])
      print("done")
    elif catalog and catalog in self.gxls.pages():
      lib = self.gxls.df(catalog)
      log_sheet, lib_gxls = catalog, None
    if isinstance(lib, pd.DataFrame) and lib.empty:
      lib = None
    if lib is not None:
      lib.drop_duplicates(subset=["partno"], inplace=True, keep="first")
      lib.reset_index(drop=True)
      lib.fillna("", inplace=True)
      lib['rexpart'] = lib['partno'].apply(_g)

    
    cmc = re.compile(r"^(L|FB|D|XW)\d+", re.I)
    buf = re.compile(r"^(U|Y|X)(\d+)?", re.I)

    b = self.tables["Board Database"]
    ins = self.tables["Board Instance"]
    c2b = dict(zip(ins["card"], ins["design"]))
    components = {}
    ibs = pd.DataFrame(columns=self.cols_ibis_page.split())
    for row in topo.itertuples():
      dshunt = eval(row.shunt) if row.shunt else {}
      io, series, shunts = (
          _u(row.io),
          _u(row.series),
          _u(",".join(list(dshunt.keys()))),
      )
      devices = [x for x in re.split(r"\s*,\s*", row.driver) if x] if hasattr(row, "driver") else []
      devices += [x for x in io if buf.match(x)]
      devices += [x for x in (series + shunts) if cmc.match(x)]
      devices = list(set(devices))
      dsn_name = c2b[row.card]
      component_sheet = b.loc[b["design"] == dsn_name, "component"].tolist()[0]
      if component_sheet and (component_sheet not in components):
        cell = self.gxls.read(component_sheet)
        if len(cell):
          components[component_sheet] = dict(zip(cell["refdes"], cell["partno"]))
        else:
          print(f"?? components page {component_sheet} missing or empty")
      gpn = components[component_sheet] if component_sheet in components else {}
      for refdes in devices:
        if not refdes:
          print('clocks.assign_ibs(): refdes==""? ')
          continue
        partno = gpn[refdes] if refdes in gpn else "gpn-not-found"
        if partno in ibs["partno"]:
          continue
        ribs = pd.DataFrame(columns=ibs.columns)
        rlib = pd.DataFrame(columns=lib.columns)
        common_cols = ribs.columns.intersection(rlib.columns)
        if lib is not None:
          found = lib[lib["rexpart"] == _g(partno)].reset_index(drop=True)
          if len(found):
            ribs.loc[0, common_cols] = found.loc[0, common_cols]
          else:
            rlib.loc[0, ["partno", "description"]] = [partno, today]
        ribs.loc[0, ["partno", "design", "refdes"]] = [partno, dsn_name, refdes]
        ibs = pd.concat([ibs, ribs], ignore_index=True)
        if len(rlib):
          lib = pd.concat([lib, rlib], ignore_index=True)

    # update ibis page
    ibs.fillna("", inplace=True)
    ibs.sort_values(by=["partno", "design", "refdes"], inplace=True)
    ibs = ibs.groupby(["partno", "design", "refdes"], as_index=False).first()
    self.gxls.update("ibis", ibs)

    # update part page
    if log and lib is not None:
      lib.drop_duplicates(inplace=True)
      lib.drop(columns=['rexpart'], inplace=True)
      lib.fillna("", inplace=True)
      if catalog.startswith("https:"):
        rlib = lib[(lib["description"] == today) & (lib["partno"] != "")]
        if log_sheet in lib_gxls.pages():
          writeback = lib_gxls.df(log_sheet)
          writeback = pd.concat([writeback, rlib], axis=0, join="outer", ignore_index=True)
        else:
          writeback = rlib
        writeback.drop_duplicates(inplace=True)
        writeback = writeback.groupby("partno", as_index=False).first()
        lib_gxls.update(log_sheet, writeback)
      # lib.sort_values(by=['partno'], inplace=True)
      else:
        self.gxls.update(log_sheet, lib)

    # update topology page with drivers column: assuming first comp in ibis used
    ibs_all = {}
    for ln in filter(lambda row: row.file.endswith("ibs"), ibs.itertuples()):
      ibs_file = ln.path + "/" + ln.file
      if not pstr(ibs_file).isfile:
        print(f"file {ibs_file} not found!")
        continue
      if ibs_file not in ibs_all:  # first seen of the ibis
        b = ibis(ibs_file, "short")
        if ln.component in b.component:
          pinTypes = {}
          for pin, pin_attr in b.component[ln.component]["Pin"].items():
            tag = ln.design + "." + ln.refdes + "." + pin
            pinTypes.update({tag: pin_attr["type"]})
          ibs_all.update({ibs_file: pinTypes})
      else:  # other chances seeing same ibis file
        d = ibs_all[ibs_file]
        pinTypes = {f"{ln.design}.{ln.refdes}.{k.split('.')[-1]}": v for (k, v) in d.items()}
        ibs_all[ibs_file].update(pinTypes)

    if "driver" not in topo.columns:
      topo["driver"] = ""
    for row in topo.itertuples():
      drivers = {k: True for k in re.split(r"[\s,;]+", row.driver)}  # keep user input, if anything there
      dsn_name = c2b[row.card]
      for io in row.io.split(","):
        tag = dsn_name + "." + io
        for ibs_file, iotypes in ibs_all.items():
          if tag in iotypes and iotypes[tag] in ["o", "i/o"]:
            drivers.update({tag.split(".")[1]: True})
            break
      topo.at[row.Index, "driver"] = ",".join([x for x in drivers if x])
    topo.fillna("", inplace=True)

    # move master net tag (/) to the rows with drivers if a multi-row route has any
    slash_drivers = True
    if len(topo) and slash_drivers:
      byroute = topo.groupby("route")
      for rn in list(byroute.groups.keys()):
        df = byroute.get_group(rn)
        tx = df["driver"].str.len() > 0
        if tx.any():
          topo.loc[df[tx].index, "xnet"] = df.loc[tx, "xnet"].apply(lambda x: re.sub(r"(/\d*)/$", r"\1", x + "/"))
          topo.loc[df[~tx].index, "xnet"] = df.loc[~tx, "xnet"].apply(lambda s: rm_pattern(r"/(\d*)$", s))
        if len(df) < 2:  # no need to slash for single row routes
          topo.loc[df.index, "xnet"] = df["xnet"].apply(lambda s: rm_pattern(r"/+$", s))

    # sort by real netgroups not( nopower, ~con, ~mux)
    if len(topo):
      grp = self.tables["Board NetGroup"]
      clkgrp = grp[(~grp["group"].isin(["EXCLUDE"])) & (~grp["speed"].str.startswith("dc="))]["group"].to_list()
      topo["grp1"] = topo["grp"]
      byroute = topo.groupby("route")
      for rn in list(byroute.groups.keys()):
        df = byroute.get_group(rn)
        grp1 = df[df["grp"].isin(clkgrp)]["grp"].to_list()
        if len(grp1):
          topo.loc[df.index, "grp1"] = grp1[0]
      bygrp = topo.groupby("grp1")
      topo1 = None
      i = 0
      for grp1 in sorted(list(bygrp.groups.keys())):
        df1 = bygrp.get_group(grp1)
        byrn = df1.groupby("route")
        for rn in byrn.groups.keys():
          df = byroute.get_group(rn)
          df.loc[:, "route"] = i
          topo1 = pd.concat([topo1, df]).reset_index(drop=True)
          i += 1
      topo = topo1.drop(columns="grp1")
      # self.gxls.update('topo1',topo)

    # set simualte=False for routes masked by longer path
    supperset_only = True
    if len(topo) and supperset_only:
      route_sets = {
          rn: set(topo.loc[topo["route"] == rn, "xnet"].apply(lambda s: rm_pattern(r"/\d*$", s)))
          for rn in set(topo["route"].unique())
      }
      # any route as a subset of others is FALSE
      for i, set_i in route_sets.items():
        for j, set_j in route_sets.items():
          if i != j and set_i.issubset(set_j):
            topo.loc[topo["route"] == i, "simulate"] = "FALSE"
            break
      # stack false sims to end of table
      sim_false = topo[topo["simulate"] == "FALSE"]
      sim_true = topo[topo["simulate"] == "TRUE"]
      topo = pd.concat([sim_true, sim_false]).reset_index(drop=True)
      # reorder route number incrementally from 0
      unique_rn = pd.Series(topo["route"]).drop_duplicates().reset_index(drop=True)
      route_to_rn = unique_rn.reset_index().set_index("route")["index"].to_dict()
      topo.loc[:, "route"] = topo["route"].map(route_to_rn)
      # self.gxls.update('topo',topo)

    # reset route numbers and zebra color toplogy sheet
    if len(topo):
      rowcolor = {rn: [] for rn in topo["route"].unique()}
      for i, rn in enumerate(topo["route"].values):
        rowcolor[rn].append(i)
      self.gxls.update(worksheet, topo, zebra=rowcolor)

    print("hola clocks.assign_ibs()!")

  def pick_driver(self, page_in="~topology’", page_out="topology’"):
    print("place holder for mannual step 5. see:")
    print(
        "https://docs.google.com/document/d/1BTvaRC3c_gJUi9-sVTEFp-4wAWHW2jtR_uDU8vsHFFE/edit?resourcekey=0-xJz_s2OG2xdpDWneNeh1mg&tab=t.fj2mc5dqwfc1#heading=h.1xe6gbskfpi3"
    )
    if self._setup() is False:
      return  # not initiate
    if page_out and page_in and page_in in self.gxls.pages():
      yes = input(f"copy page '{page_in}' to '{page_out}'? Y for yes:")
      if yes == "Y":
        self.gxls.copy(page_in, page_out)
        print(f"{page_in} duplicated to {page_out}")
      print(f"make sure master nets marked and drivers assigned in '{page_out}'")

  def extract_brd(self, n_licenses=4, design="topology", redo=False):
    """do spd extraction to designs of
    1: design=='topology', a page with a 'route' column meaning final topolgy
    2. design=='brd1' if 'brd1' is in self.tables['Board Database']
    3. design=='' all desgins as '' matchs any string
    """
    chk = {"Database Instance": "clocks.extract_brd()"}
    if self._setup(chk) is False:
      return  # not initiated

    n_licenses = int(n_licenses)

    df_dsn = self.tables["Board Database"]
    inst = self.tables["Board Instance"]
    crd2dsn = dict(zip(inst["card"], inst["design"]))

    RUN_MERGED = False

    df = pd.DataFrame()
    if design in self.gxls.pages():
      df = self.gxls.df(design, visible=True)
      # 1. design = '~toplology'
      if "route" in df.columns:
        # remove appending /(\d)*
        if self._yn_slash(df) is False:
          return
        df.loc[:, "xnet"] = df["xnet"].apply(lambda s: rm_pattern(r"/(\d*)$", s))
        df = df.assign(design=df["card"].apply(lambda x: crd2dsn[x] if x in crd2dsn else x))
    else:
      # 2. design = 'mlb'
      # 3. design = ''
      for dsn in df_dsn.itertuples():
        if not re.match(design, dsn.design):
          continue
        if not dsn.topology in self.gxls.pages():
          continue
        df1 = self.gxls.df(dsn.topology, visible=True)
        if df1.empty:
          continue
        df1 = df1.assign(design=dsn.design)
        df = pd.concat([df, df1])
    if isinstance(redo, str):
      df = df[df["design"].isin(re.split(r"\s*,\s*", redo))]
      redo = True

    if df.empty:
      print("extract_brd(): no designs to extract")
      return
    df = df[(df["simulate"] == "TRUE") & (df["io"] != "")].reset_index(drop=True)
    if df.empty:
      print("extract_brd(): find no extractions true")
      return

    grouped = df.groupby("design")
    design_names = list(grouped.groups.keys())
    unrouted = {}

    for dsn in filter(lambda row: row.design in design_names, df_dsn.itertuples()):
      # if dsn.hide: continue
      topology = grouped.get_group(dsn.design)
      if topology.empty:
        continue
      cell = topology.copy()
      cell.drop("design", axis=1, inplace=True)  # no use column as well
      bom_none = [x for x in cell["dns"] if x]
      bom_none += [x for x in dsn.dns.split(",") if x]

      file = pstr(dsn.simpath + "/" + dsn.file)
      p = os.path.dirname(dsn.simpath)
      dir = folders(os.path.dirname(p), os.path.basename(p))

      if file.isfile:
        print(f"clocks.extract_brd(): {file}")
      else:
        print(f"? clocks.extract_brd(): {file} skipped, not found")
        continue

      stk = self.gxls.text(dsn.stackup) if dsn.stackup else []
      lib = self.prj["amm_library"]
      exe = self.prj[dsn.tool]
      ext = file.ext().lower()
      if re.search(r"\.s\d+p$", ext):  # do not solve for snp files
        continue
      elif ext in [".brd", ".spd", ".tgz"]:  # extract these files
        lnk = spdlinks(self.prj["spdlinks"], app=exe, raw=file, amm=lib, stk=stk)
        if redo or lnk.translated is False:
          print(f"  translating {dsn.file} to spd")
          lnk.translate()
        spd = sigrity(app=exe, spd=lnk.spd)
        if ext == ".brd":
          brd = allegro(brd=file, editor=self.prj["allegro"])
          # mark poor routings False for simulation
          nc = brd.unrouted(topology)
          if len(nc):
            unrouted.update({nc["rpt"]: True})
            cell.loc[nc["ind"], "simulated"] = "FALSE"
          comp = component(brd.cmp, nondns=dsn.nondns, dns=",".join(bom_none))
        else:
          comp = component(spd.data, nondns=dsn.nondns, dns=",".join(bom_none))
      else:  # other files not supported yet
        print(f"file not recogonized: {file}")
        continue

      # check license
      feature = re.findall(r"\s+-PS(\w+)", exe)
      if feature:
        free_lics = license("CDS_LIC_FILE").free(feature[0])
        if free_lics < 1:
          print(f"? clocks.extract_brd(): no license {feature[0]}")
          continue
        elif free_lics < n_licenses:
          n_licenses = min(n_licenses, free_lics)
          print(f"? clocks.extract_brd(): reduced to {n_licenses} parrallel run due to lack of license {feature[0]}")

      tcl = tmpl.tcl(brd=None, spd=spd)
      tcl.set("dns", comp.dns)
      tcl.clocks(cell)
      if RUN_MERGED:
        dir.run_spd(spd, licenses=n_licenses)
      else:
        spd.run(licenses=n_licenses, overwrite=redo)

    if len(unrouted):
      print(f'  check following files for unrouted nets:\n{":".join(unrouted.keys())}')
      for f in unrouted:
        ftxt = "file:///" + os.path.abspath(f).replace("\\", "/")
        webbrowser.open(ftxt)
    print("hola clocks.extract_brd()")

  def write_ckt(self, worksheet="topology", copys=True):
    """reveal_worksheet write to the 'circuit' page for debugging purpose"""
    chk = {"Database Instance": "clocks.write_ckt()"}
    if self._setup(chk, tab_rows="visible") is False:
      return  # not initiated

    reveal_worksheet = False
    cwd = self.prj["prj_folder"] + "/Clocks"

    ws = None
    if not (worksheet and worksheet in self.gxls.pages()):
      print("clocks.write_ckt(): need the worksheet page in gsheet")
      return
    else:
      ws = self.gxls.df(worksheet, visible=True)
    if ws is None or ws.empty:
      return
    else:
      ws = ws[(ws["simulate"] == "TRUE") & (ws["io"] != "")]

    if ws is None or ws.empty:
      return
    elif self._yn_slash(ws) is False:
      return

    # df table has mux/conn connection iformation already
    # attach more attributes: component value( value), net voltage(shunt column)
    df_dsn = self.tables["Board Database"]
    df_inst = self.tables["Board Instance"]
    crd2dsn = dict(zip(df_inst["card"], df_inst["design"]))

    df_conn = self.tables["Board Connection"]
    if len(df_conn):
      df_conn["cardu"] = df_conn["Aboard"].str.cat(df_conn["Aconn"].str.upper(), sep=".")

    df_mux = self.tables.get("Multiplexer", {})
    df_sw = self.gxls.df("switch", visible=True)

    if len(df_sw) and len(df_mux):
      df_sw["cardu"] = df_sw["card"].str.cat(df_sw["refdes"].str.upper(), sep=".")
      df_sw["model"] = ""
      for ln in df_mux.itertuples():
        model = df_mux.loc[df_mux["partno"] == ln.partno, "model"].tolist()
        df_sw.at[ln.Index, "model"] = model[0]

    # load parts
    components, nets, dsn_in_ws = {}, {}, {}
    for dsn in df_dsn.itertuples():
      brdfile = pstr(dsn.simpath)+ dsn.file
      spdfile = brdfile.ext('.spd')
      dsn_ext = brdfile.ext().lower()

      if brdfile.isfile is False:
        logme.info(f"clocks.wirte_ckt(): file {brdfile} not found")
        continue
      nets[dsn.design] = self.gxls.df(dsn.netlist)
      if re.search(r"\.s\d+p$", dsn_ext):
        continue
      elif dsn_ext == ".brd":
        brd = allegro(brd=brdfile)
        components[dsn.design] = component(brd.cmp, nondns=dsn.nondns, dns=dsn.dns)
      elif spdfile.isfile:
        spd = sigrity(spd=spdfile)
        components[dsn.design] = component(spd.data, nondns=dsn.nondns, dns=dsn.dns)
      else:
        print(f"clocks.write_ckt(): file {spdfile} skipped iterating designs")

    ws["design"] = ""
    for row in ws.itertuples():
      n = row.Index
      shunt = eval(row.shunt) if row.shunt else {}
      series = re.split(r"[\s,]+", row.series.strip()) if row.series else []
      dsn = crd2dsn[row.card]
      ws.at[n, "design"] = dsn
      if dsn in dsn_in_ws:
        dsn_in_ws[dsn].update({row.card: True})
      else:
        dsn_in_ws[dsn] = {row.card: True}
      if len(series):
        d = {u: components[dsn].value.get(u,"nan") for u in series}
        ws.at[n, "series"] = str(d)
      else:
        ws.at[n, "series"] = str({})
      if len(shunt):
        for pin, net in shunt.items():
          u = pin.split(".")[0]
          value = components[dsn].value.get(u,"nan")
          voltage = nets[dsn].loc[nets[dsn]["net"] == net, "voltage"].to_list()[0]
          shunt[pin] = f"{value},{net}={voltage}"
        ws.at[n, "shunt"] = str(shunt)

    def _flat(x):
      return [e for y in x for e in y]

    conn_snp = {}  # {'card.u': '[1,2]afile.snp'} for conn/mux
    if "joint" in ws.columns:
      row_joint = ""
      for row in ws.itertuples():
        # same route has has paired list/tuple, true if row.joint==''
        if row.joint == "" or row.joint == row_joint:
          continue
        row_joint = row.joint
        joint = eval(row.joint)
        for paired_pins in joint:
          paired_refdes = [".".join(x.split(".")[:2]) for x in paired_pins]
          if len(df_conn) and isinstance(paired_pins, list):
            if "model" not in df_conn.columns:
              continue
            m1 = _flat([df_conn.loc[df_conn["cardu"] == p, "model"].tolist() for p in paired_pins])
            m2 = _flat([df_conn.loc[df_conn["cardu"] == p, "model"].tolist() for p in paired_refdes])
            if len(m1):  # index by pin if specially specified in 'Board Connection' table
              conn_snp.update({k: m1[0] for k in paired_pins})
            elif len(m2):  # index by refdes to save size
              conn_snp.update({k: m2[0] for k in paired_refdes})
          elif len(df_sw) and isinstance(paired_pins, tuple):
            if "model" not in df_mux.columns:
              continue
            m2 = _flat([df_sw.loc[df_sw["cardu"] == p, "model"].tolist() for p in paired_refdes])
            if len(m2):  # conn_snp may grow large if index by pin than refdes
              # conn_snp.update({k: m[0] for k in paired_pins } )
              conn_snp.update({k: m2[0] for k in paired_refdes})
    if reveal_worksheet:
      self.gxls.update("circuit", ws)

    # append card column to ibis sheet
    df_ibs = self.gxls.df("ibis", visible=True)
    if df_ibs.empty:
      print("clocks.write_ckt(): aborted due to empty `ibis` page")
      return
    df_ibs.pop("note")
    df_ibs.assign(card="", cardu="")
    df_ibis = None
    for dsn, cards in dsn_in_ws.items():
      dsn_ibs = df_ibs[df_ibs["design"] == dsn].reset_index(drop=True)
      if dsn_ibs.empty:
        continue
      for card in cards:
        dsn_ibs.loc[:, "card"] = card
        dsn_ibs.loc[:, "cardu"] = dsn_ibs["refdes"].apply(lambda x: f"{card}.{x}")
        df_ibis = pd.concat([df_ibis, dsn_ibs])
    # add sparameters of conn/mux if any
    d_conn = []
    for cardu, s in filter(lambda kv: bool(kv[1]), conn_snp.items()):
      card, u = cardu.split(".")[:2]
      bracket, filename = s.split("]") if "]" in s else ("", s)
      path, file = os.path.split(filename)
      d = {k: "" for k in df_ibis.columns}
      d["component"] = bracket + "]" if bracket else "[1,2]"
      d.update(
          {
              "path": path,
              "file": file,
              "card": card,
              "refdes": u,
              "cardu": f"{card}.{u}",
          }
      )
      d_conn.append(d)
    if len(d_conn):
      df_ibis = pd.concat([df_ibis, pd.DataFrame.from_records(d_conn)])

    # copy individual snp from /SimFiles to /Results
    if copys:
      for dsn in df_dsn.itertuples():
        simdate = pstr(pstr(dsn.simpath).path())
        simdir = folders(simdate.path(),simdate.base())
        simdir.copy_snp()
    
    # check if nan components has value in ibis page
    _rlc = lambda e: re.match(r'^[rlc]\s*=',e.strip(), re.I)
    _dict = {f"{r.design}.{r.refdes}": r.component for r in df_ibs.itertuples() if _rlc(r.component)}
    if len(_dict):
      for ln in ws.itertuples():
        d = eval(ln.series)
        for u, _ in filter(lambda e: e[1]=='nan', d.items()):
          k = f"{ln.design}.{u}"
          d[u] = _dict.get(k,'=nan').rpartition('=')[-1].strip()
        ws.at[n, "series"] = str(d)
        d = eval(ln.shunt)
        for pin, expr in filter(lambda e: e[1].partition[','][0]=='nan', d.items()):
          u = pin.split(".")[0]
          k = f"{ln.design}.{u}"
          d[pin]= re.sub(r'^nan', _dict.get(k,'=nan').rpartition('=')[-1].strip(), expr)
        ws.at[n, "shunt"] = str(d)


    cir = nexxim(cwd, ws)
    cir.load_snp(by=df_dsn, copys=copys)  # if copys is true then remove snp first under ./snp
    cir.load_ibs(df_ibis)  # may also add .mod( e.g FB ) and .snp(e.g Connectors) to ./snp
    cir.list_net()
    print("hola clocks.write_ckt()!")

  def run_nexxim(self, sweep="iTxBuffer", skip_simed=True):
    """create .cir folder with modified netlist and call ansysedt"""
    # def run_nexxim(self, iTxBuffer=(0,-1), iCorner=(0,)):
    sweeps = ["iTxBuffer", "iCorner", "RxCLoad"]
    if not "ansysedt" in self.prj:
      return
    cwd = self.prj["prj_folder"].replace("\\", "/") + "/Clocks"
    cir = pstr(cwd + "/.cir")
    cir.mkdir()
    if not cir.isdir:
      logme.warning(f"error createing folder {cwd}/.cir")
      return
    for net in os.listdir(cwd):
      if net.startswith("."):
        continue
      if pstr(cwd + "/" + net).isdir:
        words = re.split(r"[_\.]+", net)
        b_driver = "b_" + "_".join(words[-3:])
        netlist = "_".join(words[:-3] + [".".join(words[-3:])])
        src = pstr(cwd + "/" + net + "/" + netlist + ".cir")
        tgt = pstr(cwd + "/.cir/" + netlist + ".cir")
        if not src.isfile:
          continue

        text = src.read()
        # comment out .tran analysis
        text = re.sub(r"\n\.tran ", "\n*.tran ", text)
        cmd = "sweep iTxBuffer" if sweep not in sweeps else f"sweep {sweep}"
        # circuit w/o driver still have all .trans commented out
        if re.search(rf"\n{b_driver} ", text):
          text = re.sub(rf"\n\*\.tran(\s.*{cmd}.*)\n", rf"\n.tran\1\n", text)

        tgt.write(text)
        if tgt.recently() is False:
          print(f"error copying to {tgt}")
    scripts = pstr(cwd + "/" + "nexxim_gui.py")
    if scripts.isfile and cir.isdir:
      ansys = self.prj["ansysedt"]
      # 			command = f'{ansys} -scriptargs "{cir}" -RunScript ../{os.path.basename(scripts)}'
      ipython_args = f"skip_simed={skip_simed} cwd={cir}"
      command = f'{ansys} -scriptargs "{ipython_args}" -RunScript ../{scripts.base()}'
      batch_file = pstr(f"{cir}/" + re.sub(r"\s", "_", cmd) + ".bat")
      batch_file.write(f"{command}\npause\n")
      print(f"{batch_file} generated with {command}")
      dos(command, cir, -1)
    print("hola clocks.run_nexxim()!")


commandLines = r"""	
0: json=
1: parse_xnet(fuzzy_hit=True, redo=False)
2: list_mux(page_in='~switch', page_out='switch')
3: connect_brd(mux_by='switch', conn_by='pin_name', trail_run=False)
4: assign_ibs(worksheet='~topology', log=False)
5: pick_driver(page_in='~topology', page_out='topology')
6: extract_brd(n_licenses=4, design='topology', redo=False)
7: write_ckt(worksheet='topology', copys=True)
8: run_nexxim(sweep='iTxBuffer', skip_simed=True)
"""

if __name__ == "__main__":

  # need this line for compiled exe, otherwise infinite loop calling exe
  multiprocessing.freeze_support()
  flow = workflow(clocks, "clocks", commandLines)
  flow.run("clocks.exe", "./json/ck_p26_grizzly.json")
  #flow.run("clocks.exe", "./json/ck_p25_bz5_evt.json")

# sigrity::spdif_option convertStaticShape {0} | {1}
# 4) add RX buffer slector on gui
# 5) simulated() must have same pins in tcl and snp
