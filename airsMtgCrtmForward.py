#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import configparser, os, sys, h5py
import numpy as np
from lib.graphics.profile import plotContour, plotLines
from lib.graphics.linePlots import basicLine, linePlotWithVerticalLines
pathInfo = configparser.ConfigParser()
# Stuff to get the installed rttov path, and import pyrttov interface
pathInfo.read('rttov.cfg')
rttovPath = pathInfo['RTTOV']['rttovPath']
pyRttovPath = os.path.join(rttovPath,'wrapper')
if not pyRttovPath in sys.path:
    sys.path.append(pyRttovPath)
import pyrttov
from lib.pycrtm.pyCRTM import pyCRTM, profilesCreate
from lib.pycrtm.crtm_io import readTauCoeffODPS 
from lib.pycrtm.units import  waterPpmvDry2GmKgDry, gasPpmvMoistToDry
from lib.pycrtm.interpolation import profileInterpolate 
from matplotlib import pyplot as plt
def wv2um(wv):
    return(10000.0/wv)

def readProfileItemsH5( filename, additionalItems = []):
    """
    Read an RTTOV-style atmosphere profile.
    In: filename to hdf5
    Out: Pressure, Temperature, CO2, O3 [nprofiles,nlevels]
    Out: Gas_Units (mass, ppmv dry, ppmv moist)
    """
    items = ['T','Q','O3']
    if(len(additionalItems)>0):
        for i in additionalItems:
            items.append(i)
    h5 = h5py.File( filename )
    groups = list(h5['PROFILES'].keys())
    nprofiles = len(groups)
    nlevs, = np.asarray( h5['PROFILES'][groups[0]]['P'] ).shape 
    P = np.zeros([nprofiles,nlevs])
    itemsOut = {}
    for i in items: itemsOut[i] = np.zeros([nprofiles,nlevs])
    for i,g in enumerate(groups):
        P[i,:] = np.asarray(h5['PROFILES'][g]['P'])
        for ii in items:
            itemsOut[ii][i,:] = np.asarray(h5['PROFILES'][g][ii])
        GasUnits = int(np.asarray(h5['PROFILES'][g]['GAS_UNITS']))
    
    return P, itemsOut, GasUnits 

def setProfilesCRTM(h5_mass, h5_ppmv, layerPressuresCrtm, additionalItems = [], method='average'):
    nprofiles = 6
    profilesCRTM = profilesCreate( 6, 100, nAerosols=0, nClouds=0, additionalGases = additionalItems )
   
    Pi, profileItems, gas_units = readProfileItemsH5(h5_ppmv, additionalItems = additionalItems)
    interpOb = profileInterpolate(layerPressuresCrtm, Pi, profileItems)
    interpOb.interpProfiles(method=method) 
    profilesCRTM.P[:,:], profileItems = interpOb.get() 

    for i in list(profileItems.keys()):
        exec( "profilesCRTM.{}[:,:] = profileItems['{}']".format(i,i) )
    for i in list(profileItems.keys()):
        if ( i != 'T' ):
            exec('profilesCRTM.{}[:,:] = gasPpmvMoistToDry(profilesCRTM.{}[:,:], profilesCRTM.Q[:,:])'.format(i,i))
        
    profilesCRTM.Q[:,:] = waterPpmvDry2GmKgDry(profilesCRTM.Q[:,:]) 
    profilesCRTM.Pi[:,:] = Pi

    profilesCRTM.Angles[:,:] = 0.0
    profilesCRTM.Angles[:,2] = 100.0  # Solar Zenith Angle 100 degrees zenith below horizon.

    profilesCRTM.DateTimes[:,0] = 2015
    profilesCRTM.DateTimes[:,1] = 8
    profilesCRTM.DateTimes[:,2] = 1

    # Turn off Aerosols and Clouds
    #profilesCRTM.aerosolType[:] = -1
    #profilesCRTM.cloudType[:] = -1

    profilesCRTM.surfaceFractions[:,:] = 0.0
    profilesCRTM.surfaceFractions[:,1] = 1.0 # all water!
    profilesCRTM.surfaceTemperatures[:,:] = 293.0 
    profilesCRTM.S2m[:,1] = 35.0 # just use salinity out of S2m for the moment.
    profilesCRTM.windSpeed10m[:] = 0.0
    profilesCRTM.windDirection10m[:] = 0.0 

    # land, soil, veg, water, snow, ice
    profilesCRTM.surfaceTypes[:,3] = 1 
    
    return profilesCRTM
if __name__ == "__main__":
    #################################################################################################
    # Get installed path to coefficients from pycrtm submodule install (crtm.cfg in pycrtm directory)
    # load stuff we need for CRTM coefficients
    #################################################################################################
    pathToThisScript = os.path.dirname(os.path.abspath(__file__))
    pathInfo = configparser.ConfigParser()
    pathInfo.read( os.path.join(pathToThisScript,'lib','pycrtm','crtm.cfg') )
    coefficientPathCrtm = pathInfo['CRTM']['coeffs_dir']
    #################################################################################################
    # Get CRTM coefficient interface levels, and pressure layers 
    # Pull pressure levels out of coefficient
    # get pressures used for profile training in CRTM.
    #################################################################################################
    crtmTauCoef, _ = readTauCoeffODPS( os.path.join(coefficientPathCrtm,'airs_aqua.TauCoeff.bin') )
    coefLevCrtm = np.asarray(crtmTauCoef['level_pressure'])
    layerPressuresCrtm = np.asarray(crtmTauCoef['layer_pressure'])

    ##########################
    # Set Profiles
    ##########################
    h5_mass =  os.path.join(rttovPath,'rttov_test','profile-datasets-hdf','standard101lev_allgas_kgkg.H5')
    h5_ppmv =  os.path.join(rttovPath,'rttov_test','profile-datasets-hdf','standard101lev_allgas.H5')

    profilesCRTM  = setProfilesCRTM( h5_mass, h5_ppmv, layerPressuresCrtm, additionalItems=['CO2','CO'] )

    print("Now on to CRTM.")
    # get AIRS channels
    h5 = h5py.File(os.path.join(pathToThisScript,'etc','airs_wavenumbers.h5') )
    chans =np.asarray( h5['idxBufrSubset'])
    idx = np.arange(0,len(chans)) 
    idxOz = np.asarray(h5['idxEcmwfOzoneInInstrument']).astype(np.int)
    idxOz2 = np.array([1012,1024,1088,1111,1120])-1
    idxOz3 = np.array([1012,1088,1111,1120])-1
    wavenumbersInstrument = np.asarray(h5['wavenumbers'])
    #########################
    # Run CRTM
    #########################
    crtmOb = pyCRTM()
    crtmOb.profiles = profilesCRTM
    crtmOb.coefficientPath = coefficientPathCrtm 
    crtmOb.sensor_id = 'airs_aqua' 
    crtmOb.nThreads = 6

    crtmOb.loadInst()
    crtmOb.runDirect()

    #########################
    # End Run CRTM
    #########################
    print(np.asarray(idxOz))

    # RT is done! Great! Let's make some plots!

    wv = np.asarray(crtmOb.Wavenumbers)
    profileNames = ['1 Tropical','2 Mid-Lat Summer', '3 Mid-Lat Winter', '4 Sub-Arctic Summer', '5 Sub-Arctic Winter', '6 US-Standard Atmosphere' ]
    for i,n in enumerate(profileNames): 
        key = n.replace(" ","_")+'_'
        linePlotWithVerticalLines(wv, crtmOb.Bt[i,:], wavenumbersInstrument[idxOz-1],'Wavenumber [cm$^{-1}$]', 'Brightness Temperature [K]', '', 'spectrum_{}.pdf'.format(n), xvals2=(wv2um,wv2um), xvals2Label='Wavelength [$\mu$m]')
        linePlotWithVerticalLines(wv, crtmOb.Bt[i,:], wavenumbersInstrument[idxOz-1],'Wavenumber [cm$^{-1}$]', 'Brightness Temperature [K]', '', 'spectrum_{}_zoom.pdf'.format(n),xlim=(990,1100), xvals2=(wv2um,wv2um), xvals2Label='Wavelength [$\mu$m]')
        linePlotWithVerticalLines(wv, crtmOb.Bt[i,:], wavenumbersInstrument[idxOz2],'Wavenumber [cm$^{-1}$]', 'Brightness Temperature [K]', '', 'spectrum_{}_zoom2.pdf'.format(n),xlim=(990,1100), xvals2=(wv2um,wv2um), xvals2Label='Wavelength [$\mu$m]')
        linePlotWithVerticalLines(wv, crtmOb.Bt[i,:], wavenumbersInstrument[idxOz3],'Wavenumber [cm$^{-1}$]', 'Brightness Temperature [K]', '', 'spectrum_{}_zoom3.pdf'.format(n),xlim=(990,1100), xvals2=(wv2um,wv2um), xvals2Label='Wavelength [$\mu$m]')
