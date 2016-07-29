# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:01:27 2016

@author: anna
"""
import time
import paramiko
import os
import logging
logging.getLogger("paramiko").setLevel(logging.WARNING)
# The function to parse the input script arguments
def parse_args(argv):
    user = ''
    password = ''
    host = ''
    param = ''
    minsize=''
    try:
        opts, args = getopt.getopt(argv,"u:p:h:v:s:",["user=","password=", "host=", "param=", "size="])
    except getopt.GetoptError:
        print 'argparse_my.py -u <user> -p <password> -h <host>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-N':
            print 'argparse_my.py -u <user> -p <password> -h <host>'
            sys.exit()
        elif opt in ("-u", "--user"):
            user = arg
        elif opt in ("-p", "--password"):
            password = arg
        elif opt in ("-h", "--host"):
            host = arg
        elif opt in ("-v", "--param"):
            param = arg
        elif opt in ("-s", "--size"):
            minsize = arg
    return ([user, password, host, param, minsize])

#Copy sar data from remote host
def get_host_data (host, user, password):
    #Create the directory to store data
    if not os.path.exists('/home/'+user+'/tmp'):
        os.makedirs('/home/'+user+'/tmp')
    if not os.path.exists('/home/'+user+'/tmp/'+host):
        os.makedirs('/home/'+user+'/tmp/'+host)
    host=host
    user = user
    password = password
    files_stat=['sar']
    k=1
    paramiko.util.log_to_file('ssh.log') # sets up logging
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host+'.etadirect.com', username=user, password=password)
    cmd='uname'
    stdin, stdout, stderr = client.exec_command(cmd)
    out=[x.replace('\n','') for x in stdout]
    print ("OS is:", out)
    if out[0]!=u'Linux':
        return 0
    cmd='ls /var/log/sa/'
    stdin, stdout, stderr = client.exec_command(cmd)
    sars=[x.replace('\n','') for x in stdout if 'sar' in x]
    #print ("Response was:", sars)
    time.sleep(1)
    #print ('Done')

        #Copy file from remote host to local
    port=22
    transport = paramiko.Transport((host+'.etadirect.com', port))
    transport.connect(username = user, password = password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    # Download
    filepath = '/var/log/sa/'
    localpath = '/home/'+user+'/tmp/'+host+'/'
    for sar in sars:
        sftp.get(filepath+sar, localpath+sar)
    client.close()
