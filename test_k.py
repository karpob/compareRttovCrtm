#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import configparser, os, sys, h5py
import numpy as np
from lib.graphics.profile import plotContour, plotLines
from lib.graphics.linePlots import basicLine
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

def setRttovProfiles( h5ProfileFileName, additionalItems=[] ):
    nlevels = 101  
    nprofiles = 6
    myProfiles = pyrttov.Profiles(nprofiles, nlevels)
    myProfiles.P, profileItems, myProfiles.GasUnits = readProfileItemsH5( h5ProfileFileName, additionalItems)
    for item in list(profileItems.keys()):
        exec("myProfiles.{} = profileItems['{}']".format(item,item))
       
    # View/Solar angles
    # satzen, satazi, sunzen, sunazi
    #nprofiles, nvar (4)
    myProfiles.Angles = 0.0*np.zeros([nprofiles,4])
    # set solar zenith angle below horizon +10 deg
    myProfiles.Angles[:,2] = 100.0 #below horizon for solar

    # s2m surface 2m variables
    # surface : pressure, temperature, specific humidity, wind (u comp), wind (v comp), windfetch 
    # nprofiles, nvar (6)
    s2m = []

    for i in list(range(nprofiles)):
        Ps = myProfiles.P[:,-1]
        Ts = myProfiles.T[:,-1]
        Qs = myProfiles.Q[:,-1]
        
        s2m.append([Ps[i], Ts[i] , Qs[i], 0, 0., 10000.])
    myProfiles.S2m = np.asarray(s2m)
    
    #Skin variables
    #skin%t, skin%salinity, skin%snow_fraction, skin %foam_fraction, skin%fastem(1:5), skin%specularity
    #nprofiles, nvar (10)
    myProfiles.Skin = np.zeros([nprofiles,10])
    myProfiles.Skin[:,0] = 270.0
    myProfiles.Skin[:,1] = 35.0

    #Surface Type info
    # surftype (land = 0, sea = 1, seacice =2), watertype (fresh = 0, ocean = 1)
    #nprofiles, nvar (2)
    myProfiles.SurfType = np.ones([nprofiles,2])

    #Surface geometry
    # latitude, longitude, elevation (lat/lon used in refractivity and emis/BRDF atlases, elevation used for refractivity).
    #nprofiles, nvar (3)
    myProfiles.SurfGeom = np.zeros([nprofiles,3])

    #Date/times 
    #nprofiles, nvar(6)
    datetimes = []
    for i in list(range(nprofiles)):
        datetimes.append( [2015 ,   8,    1,    0,    0,    0]) 
    myProfiles.DateTimes = np.asarray(datetimes)
    return myProfiles

def setProfilesCRTM(h5_mass, h5_ppmv, layerPressuresCrtm, additionalItems = [], method='average'):
    nprofiles = 6
    profilesCRTM = profilesCreate( 6, 100, additionalGases = additionalItems )
   
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
    profilesCRTM.aerosolType[:] = -1
    profilesCRTM.cloudType[:] = -1

    profilesCRTM.surfaceFractions[:,:] = 0.0
    profilesCRTM.surfaceFractions[:,1] = 1.0 # all water!
    profilesCRTM.surfaceTemperatures[:,:] = 270.0 
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
    crtmTauCoef, _ = readTauCoeffODPS( os.path.join(coefficientPathCrtm,'iasi616_metop-b.TauCoeff.bin') )
    coefLevCrtm = np.asarray(crtmTauCoef['level_pressure'])
    layerPressuresCrtm = np.asarray(crtmTauCoef['layer_pressure'])

    ##########################
    # Set Profiles
    ##########################
    h5_mass =  os.path.join(rttovPath,'rttov_test','profile-datasets-hdf','standard101lev_allgas_kgkg.H5')
    h5_ppmv =  os.path.join(rttovPath,'rttov_test','profile-datasets-hdf','standard101lev_allgas.H5')

    profilesCRTM  = setProfilesCRTM( h5_mass, h5_ppmv, layerPressuresCrtm, additionalItems=['CO2','CO'] )
    myProfiles = setRttovProfiles( h5_mass, additionalItems=['CO2','CO'])

    print("Now on to CRTM.")
    # get the 616 channel subset for IASI
    h5 = h5py.File('iasi_wavenumbers.h5')
    chans =np.asarray( h5['idxBufrSubset'])
    idx = np.arange(0,len(chans)) 
   

    #########################
    # Run CRTM
    #########################
    crtmOb = pyCRTM()
    crtmOb.profiles = profilesCRTM
    crtmOb.coefficientPath = coefficientPathCrtm 
    crtmOb.sensor_id = 'iasi616_metop-a' 
    crtmOb.nThreads = 6

    crtmOb.loadInst()
    crtmOb.runK()

    #########################
    # End Run CRTM
    #########################

 
    #########################
    # Run RTTOV
    #########################

    # Create Rttov object
    rttovObj = pyrttov.Rttov()
    # set options.
    rttovObj.Options.AddInterp = False
    rttovObj.Options.InterpMode = 3
    rttovObj.Options.FixHgpl = True 
    rttovObj.Options.RegLimitExtrap = False
    rttovObj.Options.Spacetop = False 
    rttovObj.Options.Lgradp = False
    rttovObj.Options.StoreTrans = True
    rttovObj.Options.StoreRad = True
    rttovObj.Options.StoreRad2 = True
    rttovObj.Options.StoreEmisTerms = True
    rttovObj.Options.VerboseWrapper = True
    rttovObj.Options.CO2Data = True
    rttovObj.Options.OzoneData = True
    rttovObj.Options.COData = True
    rttovObj.Options.IrSeaEmisModel = 2
    rttovObj.Options.UseQ2m = False
    rttovObj.Options.DoNlteCorrection = True
    rttovObj.Options.AddSolar = True
    rttovObj.Options.FastemVersion = 6
    rttovObj.Options.Nthreads = 6
    rttovObj.Options.Switchrad = True
    rttovObj.FileCoef = os.path.join(rttovPath, 'rtcoef_rttov12','rttov9pred101L','rtcoef_metop_2_iasi.H5')
    #load the coefficient
    rttovObj.loadInst(channels=chans)

    rttovObj.printOptions()
    #load the profiles
    rttovObj.Profiles = myProfiles

    # have rttov calculate surface emission.
    #rttovObj.SurfEmisRefl = -1.0*np.ones((2,myProfiles.P.shape[0],rttovObj.Nchannels), dtype=np.float64)
    
    # When we can't duplicate, CHEAT! Use CRTM's emissivity!
    rttovObj.SurfEmisRefl = crtmOb.surfEmisRefl[:,:]
    # run it 
    rttovObj.runK()
    
    ########################
    # End Run RTTOV
    ########################

    # RT is done! Great! Let's make some plots!

    wv = np.asarray(crtmOb.Wavenumbers)
    profileNames = ['1 Tropical','2 Mid-Lat Summer', '3 Mid-Lat Winter', '4 Sub-Arctic Summer', '5 Sub-Arctic Winter', '6 US-Standard Atmosphere' ]
    sensitivities = ['O3','Q','T','CO2','CO']
    for i,n in enumerate(profileNames): 
        key = n.replace(" ","_")+'_'
        for s in sensitivities:
            exec('sValCrtm = crtmOb.{}K[i,:,:]*profilesCRTM.{}[i,:]'.format(s,s)) 
            exec('sValRttov = rttovObj.{}K[i,:,:]*myProfiles.{}[i,:]'.format(s,s)) 
            maxS = max(sValCrtm.max().max(),sValRttov.max().max())
            minS = max(sValCrtm.min().min(),sValRttov.min().min())
            symMaxS = max(abs(minS),abs(maxS))
            symMinS = -1.0*symMaxS
            plotContour(wv, profilesCRTM.P[i,:], sValCrtm,\
                        'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Jacobian [K]',\
                        profileNames[i]+' CRTM {} Jacobian'.format(s.replace('O3','O$_3$').replace('Q','H$_2$O').replace('T','Temperature').replace('CO2','CO$_2$').replace('N2O','N$_2$O')),\
                        key+'{}k_crtm.png'.format( s.lower() ),\
                        zlim = [symMinS, symMaxS] )    
            plotContour(wv, myProfiles.P[i,:], sValRttov,\
                        'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Jacobian [K]',\
                        profileNames[i]+' RTTOV {} Jacobian'.format(s.replace('O3','O$_3$').replace('Q','H$_2$O').replace('T','Temperature').replace('CO2','CO$_2$').replace('N2O','N$_2$O')),\
                        key+'{}k_rttov.png'.format( s.lower() ),\
                        zlim = [symMinS, symMaxS] )    

        wfCRTM =-1.0*np.diff(crtmOb.TauLevels[i,idx,:])/np.diff(np.log(profilesCRTM.P[i,:]))
        wfRTTOV =-1.0* np.diff(rttovObj.TauLevels[i,idx,:])/np.diff(np.log(myProfiles.P[i,:]))

        maxWf = max(wfCRTM.max().max(),wfRTTOV.max().max())
        minWf = min(wfCRTM.min().min(),wfRTTOV.min().min())
        mmWf= max(abs(minWf),maxWf)
        maxWf = mmWf
        minWf = -1.0*mmWf

        plotContour(wv,profilesCRTM.P[i,:],wfCRTM[:,:],'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Weighting Function',profileNames[i]+' CRTM Weighting Function',key+'WF_crtm.png', zlim = [minWf, maxWf])    
        plotContour(wv, myProfiles.P[i,:], wfRTTOV[:,:],'Wavenumber [cm$^{-1}$]','Pressure [hPa]','Weighting Function',profileNames[i]+' RTTOV Weighting Function',key+'WF_rttov.png', zlim = [minWf, maxWf])    

        basicLine(wv, np.asarray([crtmOb.Bt[i,idx],rttovObj.Bt[i,idx]]).T,\
                  'Wavenumber [cm$^{-1}$]', 'Brightness Temperature [K]',\
                  'IASI Brightness Temperature Profile'+n, key+'iasi_crtm_rttov.png', legendItems = ('CRTM','RTTOV'), cmap='bwr')
        basicLine(wv, crtmOb.Bt[i,idx]-rttovObj.Bt[i,idx],\
                  'Wavenumber [cm$^{-1}$]', 'CRTM - RTTOV Brightness Temperature [K]',\
                  'CRTM - RTTOV Brightness Temperature', key+'iasi_crtm_rttov_diff.png')
        basicLine(wv, np.asarray([crtmOb.surfEmisRefl[0,i,idx],rttovObj.SurfEmisRefl[0,i,idx]]).T,\
                  'Wavenumber [cm$^{-1}$]', 'Brightness Temperature [K]',\
                  'IASI Emissivity Profile'+n, key+'iasi_emissivity_crtm_rttov.png', legendItems = ('CRTM','RTTOV'), cmap='bwr')
        basicLine(wv, crtmOb.surfEmisRefl[0,i,idx]-rttovObj.SurfEmisRefl[0,i,idx],\
                  'Wavenumber [cm$^{-1}$]', 'CRTM - RTTOV Emissivity',\
                  'CRTM - RTTOV Emissivity', key+'iasi_crtm_rttov_diff.png')
        
    err = crtmOb.Bt[:,idx]-rttovObj.Bt[:,idx]
    sqerr = err**2
    mse = sqerr.mean(axis=0)
    rmse = np.sqrt(mse)
    basicLine(wv, rmse,\
              'Wavenumber [cm$^{-1}$]','Brightness Temperature Difference RMS [K]',\
              'IASI Brightness Temperature Difference RMSD','iasi_crtm_rttov_rms.png')  

