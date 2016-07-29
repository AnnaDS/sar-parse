# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:08:26 2016

@author: anna
"""
import os
import re
import traceback
import pandas as pd
import mmap
import itertools
import numpy as np
import datetime
from os import listdir
from os.path import isfile, join
from sar import PART_CPU, PART_MEM, PART_SWP, PART_IO, \
    PATTERN_CPU, PATTERN_MEM, PATTERN_SWP, PATTERN_IO, PATTERN_RESTART, \
    FIELDS_CPU, FIELD_PAIRS_CPU, FIELDS_MEM, FIELD_PAIRS_MEM, FIELDS_SWP, \
    FIELD_PAIRS_SWP, FIELDS_IO, FIELD_PAIRS_IO
    
def _split_file(self, data=''):
    '''
    Splits SAR output or SAR output file (in ASCII format) in order to
    extract info we need for it, in the format we want.
        :param data: Input data instead of file
        :type data: str.
        :return: ``List``-style of SAR file sections separated by
            the type of info they contain (SAR file sections) without
            parsing what is exactly what at this point
    '''

    # Filename passed checks through __init__
    if ((self and os.access(self, os.R_OK))
        or data != ''):

        fhandle = None

        if (data == ''):
            try:
                fhandle = os.open(self, os.O_RDONLY)
            except OSError:
                print(("Couldn't open file %s" % (self.__filename)))
                fhandle = None

        if (fhandle or data != ''):

            datalength = 0
            dataprot = mmap.PROT_READ

            if (data != ''):
                fhandle = -1
                datalength = len(data)
                dataprot = mmap.PROT_READ | mmap.PROT_WRITE

            try:
                sarmap = mmap.mmap(
                    fhandle, length=datalength, prot=dataprot
                )
                if (data != ''):
                    sarmap.write(data)
                    sarmap.flush()
                    sarmap.seek(0, os.SEEK_SET)

            except (TypeError, IndexError):
                if (data == ''):
                    os.close(fhandle)
                traceback.print_exc()
                # sys.exit(-1)
                return False

            # Here we'll store chunks of SAR file, unparsed
            searchunks = []
            oldchunkpos = 0

            dlpos = sarmap.find(b"\n\n", 0)
            size = 0

            if (data == ''):
                # We can do mmap.size() only on read-only mmaps
                size = sarmap.size()
            else:
                # Otherwise, if data was passed to us,
                # we measure its length
                len(data)

            # oldchunkpos = dlpos

            while (dlpos > -1):  # mmap.find() returns -1 on failure.

                tempchunk = sarmap.read(dlpos - oldchunkpos)
                searchunks.append(tempchunk.strip())

                # We remember position, add 2 for 2 DD's
                # (newspaces in production). We have to remember
                # relative value
                oldchunkpos += (dlpos - oldchunkpos) + 2

                # We position to new place, to be behind \n\n
                # we've looked for.
                try:
                    sarmap.seek(2, os.SEEK_CUR)
                except ValueError:
                    print(("Out of bounds (%s)!\n" % (sarmap.tell())))
                # Now we repeat find.
                dlpos = sarmap.find(b"\n\n")

            # If it wasn't the end of file, we want last piece of it
            if (oldchunkpos < size):
                tempchunk = sarmap[(oldchunkpos):]
                searchunks.append(tempchunk.strip())

            sarmap.close()

        if (fhandle != -1):
            os.close(fhandle)

        if (searchunks):
            return searchunks
        else:
            return False

    return False


def get_values(s, di):
    df = {}
    if s is None:
        return df
    s = s.split()
    if s[0] == 'Average:':
        df['time'] = s[0]
        for k, v in di.items():
            df[k] = s[(int(v) - 1)]
    else:
        df['time'] = ' '.join([':'.join([s[0].split(':')[0], s[0].split(':')[1], "00"]), s[1]])

        for k, v in di.items():
            pattern_re = re.compile(k)
            if (pattern_re.search(s[v])):
                df[k] = np.nan
            else:
                df[k] = s[v]

    return (df)


def data_ind(part_parts, FIELDS_CPU):
    return_dict = {}
    counter = 0
    for piece in part_parts:
        for colname in FIELDS_CPU:
            pattern_re = re.compile(colname)
            if (pattern_re.search(piece)) and piece == colname.replace("\\", ""):
                return_dict[colname.replace("%", "").replace("\\", "")] = counter
                # print(piece)
                break
        counter += 1
    return return_dict


def get_data(S):
    cpu_pattern = re.compile('.*CPU.*(usr|user).*nice.*sys.*')
    mem_pattern = re.compile(PATTERN_MEM)
    swp_pattern = re.compile(PATTERN_SWP)
    io_pattern = re.compile(PATTERN_IO)
    #PATTERN_IO_P = ['DEV', '^tps', '^rd_sec\\/s', '^wr_sec\\/s', 'avgrq-sz', 'avgqu-sz', 'await', 'svctm', '\\%util']
    PATTERN_IO_P = '.*tps.*rd_sec\/s.*wr_sec\/s.*avgrq-sz.*avgqu-sz.*await.*svctm.*\%util.*'
    io_pattern_p=re.compile(PATTERN_IO_P)
    restart_pattern = re.compile(PATTERN_RESTART)
    iface_pattern = re.compile('.*IFACE.*rxerr.*txerr.*coll.*rxdrop.*txdrop.*txcarr.*rxfram.*rxfifo.*txfifo.*')
    di = {}
    dm = {}
    df = {}
    io = {}
    iop={}
    FIELDS_MEM= ['kbmemfree', 'kbmemused', '\\%memused', 'kbbuffers', 'kbcached', 'kbcommit','\\%commit']
    FIELDS_CPU = ['CPU', '\\%usr', '\\%nice', '\\%sys', '\\%iowait', '\\%idle']
    FIELDS_IFACE = ['IFACE', 'rxerr/s', 'txerr/s', 'coll/s', 'rxdrop/s', 'txdrop/s', 'txcarr/s', 'rxfram/s', 'rxfifo/s',
                    'txfifo/s']
    FIELDS_IOPS = ['tps', 'rtps', 'wtps', 'bread/s', 'bwrtn/s']
    FIELDS_IOPS_PERC = ['DEV', 'tps', 'rd_sec/s', 'wr_sec/s', 'avgrq-sz', 'avgqu-sz', 'await', 'svctm','\\%util']

    CPU = list()
    MEM = list()
    IF = list()
    IO = list()
    IOP = list()
    for k in range(len(S)):

        D = (S[k].decode().split('\n'))
        part = D[0]
        if (cpu_pattern.search(part)):
            if len(di) == 0:
                part_parts = part.split()
                di = data_ind(part_parts, FIELDS_CPU)
            CPUn = []
            for dd in D:
                CPUn.append(get_values(dd, di))
            # CPUn=list(map(get_values, D, itertools.repeat(di)))
            CPU.extend(CPUn)
        elif (mem_pattern.search(part)):
            if len(dm) == 0:
                part_parts = part.split()
                dm = data_ind(part_parts, FIELDS_MEM)
            MEMn = []
            for dd in D:
                MEMn.append(get_values(dd, dm))
            MEM.extend(MEMn)
        elif (iface_pattern.search(part)):
            if len(df) == 0:
                part_parts = part.split()
                df = data_ind(part_parts, FIELDS_IFACE)
            IFn = []
            for dd in D:
                IFn.append(get_values(dd, df))
            IF.extend(IFn)
        elif (io_pattern.search(part)):
            if len(io) == 0:
                part_parts = part.split()
                io = data_ind(part_parts, FIELDS_IOPS)
            IOn = []
            for dd in D:
                IOn.append(get_values(dd, io))
            IO.extend(IOn)
        elif (io_pattern_p.search(part)):
            if len(iop) == 0:
                part_parts = part.split()
                iop = data_ind(part_parts, FIELDS_IOPS_PERC)
            IOPn = []
            for dd in D:
                IOPn.append(get_values(dd, iop))
            IOP.extend(IOPn)

    return (CPU, MEM, IF, IO, IOP)

def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)

def get_resource_tables(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    pattern_res = re.compile('sar\\d{2}')
    DFR = pd.DataFrame()
    DMR = pd.DataFrame()
    IFR = pd.DataFrame()
    IOR = pd.DataFrame()
    IOPR = pd.DataFrame()
    dr=re.compile('(\d+-\d+-\d+)')
    for o in onlyfiles:
        if (o.replace("sar", "")).isdigit():
            sarfile = join(mypath, o)
            with open(sarfile, 'r') as f:
                first_line=f.readline()
            dd=dr.findall(first_line)    
            if len(dd)==1:
                dd=str(dd[0])
            else:
                raise ValueError("Cannot define the date in line  ", s)
            S = _split_file(sarfile)
            FF, MEM, IF, IO, IOP = get_data(S)
            DF = pd.DataFrame(FF)
            DM = pd.DataFrame(MEM)
            IF = pd.DataFrame(IF)
            IO = pd.DataFrame(IO)
            IOP = pd.DataFrame(IOP)
            d = datetime.date.today()
                            
            DF.time = DF.time.apply(lambda x: ' '.join([dd, x]))
            DM.time = DM.time.apply(lambda x: ' '.join([dd, x]))
            IF.time = IF.time.apply(lambda x: ' '.join([dd, x]))
            IO.time = IO.time.apply(lambda x: ' '.join([dd, x]))
            IOP.time = IOP.time.apply(lambda x: ' '.join([dd, x]))
            
            DF = DF.dropna()
            DM = DM.dropna()
            IF = IF.dropna()
            IO = IO.dropna()
            IOP = IOP.dropna()

            DFR = DFR.append(DF, ignore_index=True)
            DMR = DMR.append(DM, ignore_index=True)
            IFR = IFR.append(IF, ignore_index=True)
            IOR = IOR.append(IO, ignore_index=True)
            IOPR = IOPR.append(IOP, ignore_index=True)

    import time
    DFR=DFR[(~DFR.idle.str.contains('idle'))]
    DFR['idle'] = DFR['idle'].apply(float)
    DFR['used_cpu'] = 100-DFR['idle']
    
    l = 'Average:'
    DFR_Average = DFR[DFR.time.apply(lambda v: True if l in v else False)]#= DFR.loc[DFR.index.map(lambda v: True if l in v else False)]
    DFR = DFR[DFR.time.apply(lambda v: False if l in v else True)]

    DMR['memused'] = DMR['memused'].apply(float)
    DMR_Average = DMR[DMR.time.apply(lambda v: True if l in v else False)]
    DMR = DMR[DMR.time.apply(lambda v: False if l in v else True)]

    IFR['rxerr/s'] = IFR['rxerr/s'].apply(float)
    IFR_Average = IFR[IFR.time.apply(lambda v: True if l in v else False)]
    IFR = IFR[IFR.time.apply(lambda v: False if l in v else True)]
    IFR.columns =['Network_interface','CollisionN','Received_dropped_packets','Bad_packets','FIFO_overrun_errors', 'Frame_alignment_errors', 'time','Carrier_errors', 'Transmitted_dropped_packets', 'Errors_transmit_packets', 'FIFO_overrun_errors']


    IOR['tps'] = IOR['tps'].apply(float)
    IOR_Average = IOR[IOR.time.apply(lambda v: True if l in v else False)]
    IOR = IOR[IOR.time.apply(lambda v: False if l in v else True)]

    IOPR = IOPR[IOPR.time.apply(lambda v: False if l in v else True)]

    DFR.to_csv(mypath + "/cpu_data.csv", index=False)
    DMR.to_csv(mypath + "/mem_data.csv", index=False)
    IFR.to_csv(mypath + "/net_errors_data.csv", index=False)
    IOR.to_csv(mypath + "/io_data.csv", index=False)
    IOPR.to_csv(mypath + "/io_perc_data.csv", index=False)

