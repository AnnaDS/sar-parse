import sys
import os
import re
import imaplib
import email
from email.utils import parsedate_tz, mktime_tz
import datetime


def get_week_act_users(imap_login, imap_passwd, search_folder, subject_expr, leter_path, new_folder_mail=''):
    True=not False 
    imap_server='stbeehive.oracle.com'
    mailbox = imaplib.IMAP4_SSL(imap_server)
    mailbox.login(imap_login, imap_passwd)

    responce,result = mailbox.list() # list of all folders
    if responce=='OK':
        print "login to E-mail : " + responce
    else:
        print "Login failed"
        sys.exit(0)

    res, data = mailbox.select(search_folder)
    if res=="NO":
        print "wrong E-mail folder"

    resp, data = mailbox.search(None, 'SUBJECT', subject_expr)

    for i in reversed(data[0].split()):
        resp, message = mailbox.fetch(i, '(RFC822)');
        f_temp = open("temp","w+")
        f_temp.write(message[0][1])
        f_temp.close();
        f_temp=open('temp', 'r');
        mail=email.message_from_file(f_temp);
        subject = mail.get('Subject')
        #print subject
        sender = mail.get('From')
        msg_date = mail.get('Date')
        try:
            dt = datetime.datetime.strptime(msg_date, "%a, %d %b %Y %H:%M:%S %Z")
        except ValueError:
            timestamp = mktime_tz(parsedate_tz(msg_date))
            dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=timestamp)
        file_name = subject.replace("_active_users_for_all_companies.","") + "__"  + dt.strftime("%Y-%m-%d - %H%M");
        #Weekly_active_users_for_all_companies.2016-06-05_2016-06-11.ed1db20 - ED Reports <noreply@toatech.com> - 2016-06-13 - 0750
        try:
            f_message = open(leter_path + file_name + ".eml", 'w')
        except IOError:
            print "WRONG FOLDER"
            sys.exit(0)
        h = email.Header.decode_header(msg_date)
        msg_date = h[0][0].decode(h[0][1]) if h[0][1] else h[0][0]
        f_message.write("Date: "+msg_date)
        
        h = email.Header.decode_header(sender)
        sender = h[0][0].decode(h[0][1]) if h[0][1] else h[0][0]
        temp = sender.encode('utf-8')
        f_message.write("\nSender: "+temp)
        
        h = email.Header.decode_header(subject)
        subject = h[0][0].decode(h[0][1]) if h[0][1] else h[0][0]
        temp = subject.encode('utf-8')
        f_message.write("\nSubject: "+temp)
        
        msg_parts = [(part.get_filename(), part.get_payload(decode=True)) for part in mail.walk() if part.get_content_type() == 'text/plain']
        for name,data_temp in msg_parts:
            msg_body = str(data_temp)
            p_msg_body = msg_body.decode('koi8-r').encode('utf-8')
            f_message.write("\nBody:\n"+p_msg_body)

        f_temp.close()
        f_message.close()
        os.remove(r'temp')

      ## move to other folder
        if new_folder_mail!="":
            resp, data_msg = mailbox.fetch(i, '(UID)')
            pattern_uid = re.compile('\d+ \(UID (?P<uid>\d+)\)')
            msg_uid = match.group('uid')
            result_msg = mailbox.uid('COPY', msg_uid, new_folder_mail)
            if result_msg[0] == 'OK':
                mov, data_temp = mailbox.uid('STORE', msg_uid , '+FLAGS', '(\Deleted)')
                mailbox.expunge()
        #else:
        #    print "no folder"
    mailbox.close()
    mailbox.logout()
    
    subject_expr="Weekly"
    emails=os.listdir(leter_path)
    D=[]
    for f in emails:
        if f.startswith(subject_expr):
            with open(leter_path+f) as fr:
                content = fr.readlines()
                db=[x.replace('Subject:','') for x in content if x.startswith('Subject:') ]
                date = (db[0].split(".")[1]).split("_")[1]
                db=db[0].split(".")[2]
                if 'ed' in db:
                    dc="ed"+db.replace("\n", "").split("ed")[1][:1]
                else:
                    dc=""
                good_ind=[not (x.startswith('Date:') or x.startswith('Sender:') or x.startswith('Subject:') or x.startswith('Body:') or x.startswith("\"Company")) for x in content]
                lines = [a for (a, True) in zip(content, good_ind) if True]
                lines=[x.replace("\r\n", "").replace("\"", "") for x in lines]
                lines=[db.replace("\n", "")+";"+x+";"+dc+";"+date for x in lines]
                [D.append(x) for x in lines]
    with open(leter_path+'result.csv', 'wb') as thefile:
        thefile.write("%s\n" % "DB;Company;users_number;week;DC;Date")
        for item in D:
            thefile.write("%s\n" % item)
