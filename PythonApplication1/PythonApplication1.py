
from html.parser import HTMLParser
from string import *
import glob
import os
import gensim
import numpy as np



def remove_lines(istr):
    #remove lines in html and make it one long string
    htmlstr = str(istr)
    fullstr = "".join(htmlstr.split('\n'))
    return(fullstr)

def compact(istr):
  ##  if more than one \n then replace with just one
    while(istr.find("\n\xa0") != -1):
        istr = istr.replace("\n\xa0","\n")
    while(istr.find("\n ") != -1):
        istr = istr.replace("\n ", "\n")
    while(istr.find("\r\n") != -1):
        istr = istr.replace("\r\n","\n")
    while(istr.find("\n\n") != -1):
        istr = istr.replace("\n\n","\n")
    return(istr)


class MyHTMLParser(HTMLParser):
    mstr =""

    def handle_starttag(self, tag, attrs):
        if tag == 'p':
            self.mstr += "\n"
        if tag == 'br':
            self.mstr += "\n"
        if tag == 'td':
            self.mstr += " "
        if tag == 'span':
            self.mstr += " "
        if tag == 'tr':
            self.mstr += "\n"

    def handle_data(self, data):
        nme = self.lasttag
       # if(nme != 'style' and nme != '???' and nme != 'script' and nme != 'table' and nme != 'div'):
        if(nme != 'style' and nme != '???' and nme != 'script' and nme != 'table'):
            if nme == 'p':
                self.mstr += "\n"
            if nme == 'br':
                self.mstr += "\n"
            if nme == 'td':
                self.mstr += " "
            if nme != '\n' and nme != '\xa0':
                self.mstr += data

    def handle_comment(self, data):
        return

    def handle_decl(self, decl):
        return


  



parser = MyHTMLParser()
ipath='E:/Data/Flights/GB/UK_Bravofly_100_20160602'
opath = 'E:/Data/Flights/GB/UK_Bravofly_100_20160602/processed'

files = glob.glob(ipath)
for file in files:
    f = open(file,"r")
    #fo = open(path+file,"w")
    #str1 = f.read()
    #str1 = remove_lines(str1)
    #parser.feed(str1)
    #str1 = parser.mstr
    #tstr = compact(str1)
    #fo.write(tstr)
    #f.close()
    #fo.close()
    print(path.)


