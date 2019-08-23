#!/usr/bin/env python3
import ssl,pickle,glob,os
context = ssl._create_unverified_context()
import urllib.request
from bs4 import BeautifulSoup

def getNames(wmo_sensor_id, cacheDir='', useWeb=True):
    """
    for a given wmo sensor id, get the short, and long name of the sensor. 
    """
    if(useWeb):
        fp = urllib.request.urlopen('https://www.wmo-sat.info/oscar/instruments/view/{}'.format(wmo_sensor_id), context=context)
    else:
        f = glob.glob(os.path.join(cacheDir,'{:04d}*.html'.format(wmo_sensor_id)))[0]     
        fp = open(f).read()
    soup = BeautifulSoup(fp,features="html.parser")
    t = soup.findAll("table",{"class":"smalltable"})
    tr = t[0].findAll('tr')
    tt = soup.findAll("title")
    # if you REAALLY want the short name to be what is on the webpage....
    #shortName = " ".join(tt[0].text.split(' ')[6::])
    shortName = tt[0].text.split(' ')[6]
    fullName = tr[1].text.replace('Full name','').replace("\n","").lstrip().rstrip()
    return shortName, fullName


def lookupSensorFrequencyTable(wmo_sensor_id, cacheDir='',useWeb=True):
    """
    For a given WMO sensor ID return a table of frequencies/wavelengths provided by the WMO website.
    """
    if(useWeb):
        fp = urllib.request.urlopen('https://www.wmo-sat.info/oscar/instruments/view/{}'.format(wmo_sensor_id), context=context)
    else:
        f = glob.glob(os.path.join(cacheDir,'{:04d}*.html'.format(wmo_sensor_id)))[0]        
        fp = open(f).read()

    fp = urllib.request.urlopen('https://www.wmo-sat.info/oscar/instruments/view/{}'.format(wmo_sensor_id), context=context)
    soup = BeautifulSoup(fp,features="html.parser")
    t = soup.findAll("div",{"class":"frequencytable"})
    tt = t[0].findAll("table")
  
    table = []
    for tr in tt[0].findAll('tr'):
        col = tr.findAll('td')
        tr = []
        for c in col:
            tr.append(c.text)
        # if there is a V,H append two channels so we have you know...A table that has matching frequency numbers?
        if(len(tr[2].split(','))>1):
            tr_tmp = tr
            for tt in list(tr[2].split(',')):
                tmp = []
                for i,v in enumerate(tr_tmp):
                    if(i==2):
                        tmp.append(tt)
                    else:
                        tmp.append(v)
                table.append(tmp)
        else: 
            table.append(tr)
    return table

if __name__ == "__main__":
    instruments = list(range(1210)) # largest instrument number in OSCAR database right now...
    pwd = os.path.dirname(os.path.abspath(__file__))
    # Lazy way to get sensors with a table of information (not the greatest coding practice, but whatevs). 
    # If it fails at table parsing, catch expception, pass, and move on to next sensor.
    for i in list(instruments):
        try: 
            shortName, fullName = getNames(i+1, cacheDir=pwd)
            table = lookupSensorFrequencyTable(i+1,cacheDir=pwd)
            print( 'Saving {} {}'.format(shortName,fullName) )
            pkl = '{:04d}_{}.pkl'.format(i+1,shortName)
            print('Saving {}'.format(pkl)) 
            with open(os.path.join(pwd,'pkl',pkl),'wb') as f:
                pickle.dump(shortName,f)
                pickle.dump(fullName,f)
                pickle.dump(table,f)
            html = '{:04d}_{}.html'.format(i+1,shortName)
            print('Saving {}'.format(html))
            with open( os.path.join(pwd,'html', html),'wb') as f:
                fp = urllib.request.urlopen('https://www.wmo-sat.info/oscar/instruments/view/{}'.format(i+1), context=context)
                r = fp.read()
                f.write(r)
        except:pass
